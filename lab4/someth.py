from torch.nn import functional as F
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Caltech101
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt

class OptimizedContext(nn.Module):
    def __init__(self, config, class_labels, model_clip):
        super().__init__()
        self.prompt_trainer = PromptLearner(config, class_labels, model_clip)
        self.token_prompts = self.prompt_trainer.tokenized_prompts
        self.encoder_image = model_clip.visual
        self.encoder_text = PromptEncoder(model_clip)
        self.scale_logit = model_clip.logit_scale
        self.dtype = model_clip.dtype

    def forward(self, img):
        img = img.to(self.dtype)
        features_img = self.encoder_image(img.type(self.dtype))
        prompts = self.prompt_trainer()
        token_prompts = self.token_prompts
        features_text = self.encoder_text(prompts, token_prompts)
        features_img = features_img / features_img.norm(dim=-1, keepdim=True)
        features_text = features_text / features_text.norm(dim=-1, keepdim=True)
        scale_logit = self.scale_logit.exp()
        logits = scale_logit * features_img @ features_text.t()
        return logits

class PromptLearner(nn.Module):
    def __init__(self, config, class_labels, model_clip):
        super().__init__()
        num_classes = len(class_labels)
        num_context = config["TRAINER"]["COOP"]["N_CTX"]
        context_init = config["TRAINER"]["COOP"]["CTX_INIT"]
        dtype = model_clip.dtype
        context_dim = model_clip.ln_final.weight.shape[0]
        clip_image_size = model_clip.visual.input_resolution
        config_image_size = config.get("INPUT", {}).get("SIZE", (224, 224))[0]
        assert config_image_size == clip_image_size, f"config_image_size ({config_image_size}) must equal to clip_image_size ({clip_image_size})"
        if context_init:
            # 使用给定的单词初始化上下文向量
            context_init = context_init.replace("_", " ")
            num_context = len(context_init.split(" "))
            prompt = clip.tokenize(context_init)
            with torch.no_grad():
                embedding = model_clip.token_embedding(prompt).type(dtype)
            context_vectors = embedding[0, 1: 1 + num_context, :]
            prompt_prefix = context_init

        else:
            # 随机初始化
            if config["TRAINER"]["COOP"]["CSC"]:
                print("初始化类别特定的上下文")
                context_vectors = torch.empty(num_classes, num_context, context_dim, dtype=dtype)
            else:
                print("初始化通用上下文")
                context_vectors = torch.empty(num_context, context_dim, dtype=dtype)
            nn.init.normal_(context_vectors, mean=0, std=0.02)
            prompt_prefix = " ".join(["X"] * num_context)
        print(f'初始上下文: "{prompt_prefix}"')
        print(f"上下文单词（标记）数量: {num_context}")
        self.context = nn.Parameter(context_vectors)  # 待优化
        class_labels = [label.replace("_", " ") for label in class_labels]
        label_lengths = [len(_tokenizer.encode(label)) for label in class_labels]
        prompts = [prompt_prefix + " " + label + "." for label in class_labels]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.to(device)
        with torch.no_grad():
            embedding = model_clip.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + num_context:, :])  # CLS, EOS
        self.num_classes = num_classes
        self.num_context = num_context
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.label_lengths = label_lengths
        self.class_token_position = config["TRAINER"]["COOP"]["CLASS_TOKEN_POSITION"]

    def forward(self):
        context = self.context
        if context.dim() == 2:
            context = context.unsqueeze(0).expand(self.num_classes, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (num_classes, 1, dim)
                    context,  # (num_classes, num_context, dim)
                    suffix,  # (num_classes, *, dim)
                ], dim=1,
            )
        elif self.class_token_position == "middle":
            half_num_context = self.num_context // 2
            prompts = []
            for i in range(self.num_classes):
                label_len = self.label_lengths[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :label_len, :]
                suffix_i = suffix[i: i + 1, label_len:, :]
                context_i_half1 = context[i: i + 1, :half_num_context, :]
                context_i_half2 = context[i: i + 1, half_num_context:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        context_i_half1,  # (1, num_context//2, dim)
                        class_i,  # (1, label_len, dim)
                        context_i_half2,  # (1, num_context//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        return prompts
