import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import itertools
import datetime

# ============================
# 1. 参数组合定义
# ============================
learning_rates = [ 1e-3, 1e-4]
optimizers = ['SGD', 'Adam']
batch_sizes = [32, 64, 128]
freeze_backbones = [False,True]

param_combinations = list(itertools.product(learning_rates, optimizers, batch_sizes, freeze_backbones))

# ============================
# 2. 创建保存目录
# ============================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_root = f"results_CIFAR10_ResNet_{timestamp}"
os.makedirs(result_root, exist_ok=True)

# ============================
# 3. 数据加载函数
# ============================
def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), #For the input of ResNet
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

# ============================
# 4. 主搜索流程
# ============================
num_epochs = 10
best_acc = 0.0
best_config = None

for lr, opt_name, batch_size, freeze_backbone in param_combinations:
    print(f"\n🔍 Testing config: lr={lr}, opt={opt_name}, batch_size={batch_size}, freeze={freeze_backbone}")

    trainloader, testloader = get_dataloaders(batch_size)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 模型构建
    model = models.resnet18(pretrained=True)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) if opt_name == 'Adam' else \
                optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)

    train_loss_history, train_acc_history = [], []

    for epoch in range(num_epochs):
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
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = torch.max(outputs, 1)
            y_probs.extend(probs)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Final Accuracy: {acc:.4f}")

    # 保存最佳结果
    if acc > best_acc:
        best_acc = acc
        best_config = (lr, opt_name, batch_size, freeze_backbone)

    # 保存结果
    config_name = f"lr{lr}_opt{opt_name}_bs{batch_size}_freeze{freeze_backbone}"
    result_dir = os.path.join(result_root, config_name)
    os.makedirs(result_dir, exist_ok=True)

    # 保存报告
    with open(os.path.join(result_dir, "report.txt"), "w") as f:
        f.write(f"Config: {config_name}\n")
        f.write(classification_report(y_true, y_pred))
        f.write(f"\nAccuracy: {acc:.4f}\n")

    # 保存混淆矩阵图
    plt.figure()
    plt.imshow(confusion_matrix(y_true, y_pred), cmap='Blues')
    plt.title(f"Confusion Matrix\n{config_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()

    # 保存训练曲线
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

# 最佳配置
print("\n🏆 Best Configuration:")
print(f"Learning rate: {best_config[0]}")
print(f"Optimizer: {best_config[1]}")
print(f"Batch size: {best_config[2]}")
print(f"Freeze backbone: {best_config[3]}")
print(f"Accuracy: {best_acc:.4f}")

# 🏆 Best Configuration:
# Learning rate: 0.001
# Optimizer: SGD
# Batch size: 32
# Freeze backbone: True
# Accuracy: 0.8099

# Config: lr0.001_optSGD_bs32_freezeFalse
#
#
# Accuracy: 0.9539
