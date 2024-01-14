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
from someth import OptimizedContext,PromptLearner

# 加载 CLIP RN50 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("RN50", device=device)


class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.categories = os.listdir(directory)
        self.category_to_index = {category: index for index, category in enumerate(self.categories)}
        self.image_files = self.get_image_files()

    def get_image_files(self, max_files_per_category=1):
        image_files = []
        for category in self.categories:
            category_path = os.path.join(self.directory, category)
            file_count = 0
            for file_name in os.listdir(category_path):
                if file_count >= max_files_per_category:
                    break
                file_path = os.path.join(category_path, file_name)
                image_files.append((file_path, self.category_to_index[category]))
                file_count += 1
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        file_path, label = self.image_files[index]
        img = Image.open(file_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {"img": img, "label": label}


class PromptEncoder(nn.Module):
    def __init__(self, model_clip):
        super().__init__()
        self.transformer = model_clip.transformer
        self.positional_embedding = model_clip.positional_embedding
        self.ln_final = model_clip.ln_final
        self.text_projection = model_clip.text_projection
        self.dtype = model_clip.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


_tokenizer = _Tokenizer()

# 数据预处理和划分
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageDataset(directory='/kaggle/input/d/mytoumakazusa/caltech101/caltech-101',
                       transform=transform)
print(len(dataset))
class_names = dataset.categories
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 根据shot数量设置不同的epoch
num_epochs_1shot, num_epochs_2shot, num_epochs_4shot = 50, 100, 100

# 选择不同的数据集和epoch
shot_to_dataset = {1: dataset, 2: dataset, 4: dataset}
shot_to_epochs = {1: num_epochs_1shot, 2: num_epochs_2shot, 4: num_epochs_4shot}

# 定义 DataLoader
batch_size = 32  # 你可以根据需要进行调整
train_data = shot_to_dataset[1]  # 选择1-shot的数据集
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

cfg = {
    "MODEL": {"BACKBONE": {"NAME": "RN50"}},
    "TRAINER": {
        "COOP": {
            "N_CTX": 16,
            "CTX_INIT": None,  # 这里可以设置为你需要的初始化内容
            "CLASS_TOKEN_POSITION": "end",
            "CSC": False,
        },
    },
    "OPTIM": {
        "NAME": "SGD",
        "BASE_LR": 0.002,
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 1e-4,
    },
    "INPUT": {
        "SIZE": (224, 224),  # 请根据你的需要设置正确的输入尺寸
    },
}

# 构建模型
model = OptimizedContext(cfg, class_names, model_clip=clip_model)
model = model.to("cuda")

# 初始化 Text Embedding
nn.init.normal_(model.prompt_trainer.context, mean=0, std=0.02)

# 设置优化器和学习率调度器
optimizer = SGD(model.parameters(), lr=cfg["OPTIM"]["BASE_LR"], momentum=cfg["OPTIM"]["MOMENTUM"],
                weight_decay=cfg["OPTIM"]["WEIGHT_DECAY"])

# 使用余弦退火调整学习率
scheduler = CosineAnnealingLR(optimizer, T_max=shot_to_epochs[1], eta_min=1e-5)

# 训练循环
for epoch in range(1, num_epochs_1shot + 1):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs_1shot}"):
        image, label = batch["img"].to("cuda"), batch["label"].to("cuda")
        optimizer.zero_grad()
        output = model(image)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
    scheduler.step()

# 保存模型
torch.save(model.state_dict(), "1shot_model_.pth")

