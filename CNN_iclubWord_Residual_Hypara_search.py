import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
import datetime

# ============================
# 1. iCubWorld 数据集加载
# ============================
class iCubDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_per_class=100, selected_classes=7):
        self.image_paths = []
        self.labels = []
        class_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])[:selected_classes]

        for label_idx, class_name in enumerate(class_dirs):
            count = 0
            class_path = os.path.join(root_dir, class_name)
            for root, _, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm')):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(label_idx)
                        count += 1
                        if count >= max_per_class:
                            break
                if count >= max_per_class:
                    break

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, label

def get_loaders(batch_size):
    trainset = iCubDataset("../iCubWorld28/train_all_flat", selected_classes=7)
    testset = iCubDataset("../iCubWorld28/test_all_flat", selected_classes=7)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

# ============================
# 2. 模型结构（增强版）
# ============================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.gelu(out + identity)

class CustomCNN(nn.Module):
    def __init__(self, dropout_rate, num_classes=7):
        super(CustomCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 32)

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.res3 = ResidualBlock(64, 64)
        self.res4 = ResidualBlock(64, 64)

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )
        self.res5 = ResidualBlock(128, 128)
        self.res6 = ResidualBlock(128, 128)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.layer2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.layer3(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.classifier(x)
        return x

# ============================
# 3. 超参数搜索 + 训练测试流程
# ============================
param_grid = {
    'batch_size': [32,64,128],
    'lr': [1e-3, 1e-4],
    'dropout_rate': [0.3, 0.5],
    'optimizer': ['SGD', 'Adam']
}
combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())
EPOCHS = 20

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_root = f"results_CustomCNN_iCub_{timestamp}"
os.makedirs(result_root, exist_ok=True)

results = []
best_acc = 0.0
best_config = None
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

for combo in combinations:
    params = dict(zip(param_names, combo))
    print(f"\n🔍 Testing combination: {params}")

    trainloader, testloader = get_loaders(params['batch_size'])
    model = CustomCNN(dropout_rate=params['dropout_rate']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9) if params['optimizer'] == 'SGD' else optim.Adam(model.parameters(), lr=params['lr'])

    train_loss_history, train_acc_history = [], []
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total
        train_loss_history.append(running_loss)
        train_acc_history.append(acc)
        print(f"Epoch {epoch+1}: Loss={running_loss:.3f}, Acc={acc:.3f}")

    # 测试
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Final Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_config = params

    config_name = f"lr{params['lr']}_bs{params['batch_size']}_drop{params['dropout_rate']}_opt{params['optimizer']}"
    result_dir = os.path.join(result_root, config_name)
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "report.txt"), "w") as f:
        f.write(f"Config: {config_name}\n")
        f.write(classification_report(y_true, y_pred))
        f.write(f"\nAccuracy: {acc:.4f}\n")

    plt.figure()
    plt.imshow(confusion_matrix(y_true, y_pred), cmap='Blues')
    plt.title(f"Confusion Matrix\n{config_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history)
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history)
    plt.title("Accuracy")
    plt.suptitle(config_name)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "train_curves.png"))
    plt.close()

print("\n🏆 Best combination:")
print(best_config)
print(f"Accuracy: {best_acc:.4f}")

#  Best combination:
# {'batch_size': 32, 'lr': 0.0001, 'dropout_rate': 0.5, 'optimizer': 'Adam'}
# Accuracy: 0.85