import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
import datetime



transform = transforms.Compose([
    #    transforms.RandomHorizontalFlip(),
    # random turn over the image,a kind of data enhance

    # transforms.RandomCrop(32, padding=4),
    # filling the 4 pixels around the image
    # and then crop a 32x32 image randomly

    transforms.ToTensor(),
    # ToTensor transfer image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # Normalize help fasten the training
# the number in Normalize is the mean and standard of the whole dataset
# 0.5 is a simple version, using the true mean and std better
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


def get_loaders(batch_size):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
# dataloader is for read data example in batch and shuffle the dataset
# dataloader also help multithreading and support the Iteration: for batch in dataloader
# dataloader only support those standard build_in dataset eg CIFAR10 MNIST
# Or use Customization Dataset which inherit the torch.utils.data.Dataset
# need to implement __init__ __len__ __getitem__



# ResidualBlock inherit the nn.Module

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Convolution Layer：adjust the input channels to output channels
        # out_channel: the number of conv kernel
        # kernel_size: the 2D size of conv kernel  3D size:(input, kernel_size,kernel_size)
        # padding: the filled pixel size
        self.relu = nn.ReLU()
        # activate layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Identity()
        # copy the input

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    #   if the input channel is not equal to out_channel, it can not be added
    # using 1x1 Conv2d to change the size = reduce/increase the dimension of tensor



    def forward(self, x):
        identity = self.shortcut(x)
        # if equal identity or conv2d
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)



class CustomCNN(nn.Module):
    def __init__(self, dropout_rate):
        super(CustomCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        #     downsampling: select the largest number in 2x2 and slide the window
        )
        self.res1 = ResidualBlock(32, 32)

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res2 = ResidualBlock(64, 64)

        # nn.sequential is a kind of module container to pack some layers
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # change the dimension from nxn to 1 x n*n
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.res2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x



param_grid = {
    'batch_size': [32, 64, 128],
    'lr': [1e-3, 1e-4],
    'dropout_rate': [0.3, 0.5],
    'optimizer': ['SGD', 'Adam']
}

combinations = list(itertools.product(*param_grid.values()))
# list for storing those parameters
# itertools.product: generating the Descartes product of parameters
# *: for unpacking those values eg *[[1,2,3],[2,3,4],[5]] = [1,2,3] [2,3,4],[5]
# because the itertools.product only accept those sub_lists but not a whole_list

param_names = list(param_grid.keys())
EPOCHS = 10

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_root = f"results_CustomCNN_Res_{timestamp}"
os.makedirs(result_root, exist_ok=True)



results = []
best_acc = 0.0
best_config = None
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

for combo in combinations:
    # trying different parameters combinations
    # combo like (64, 0.001, 0.3, 'Adam')
    # zip the pair two parameters in func by order
    # zip_combo:[('batch_size', 64), ('lr', 0.001), ('dropout_rate', 0.3), ('optimizer', 'Adam')]
    # dic:making the list to dict
    params = dict(zip(param_names, combo))
    print(f"\n🔍 Testing combination: {params}")

    trainloader, testloader = get_loaders(params['batch_size'])
    model = CustomCNN(dropout_rate=params['dropout_rate']).to(device)
    criterion = nn.CrossEntropyLoss()
    # the loss func: is a common loss func in multi_classify task
    # the output of CrossEntropyLoss() is the loss but not the probability
    # logit output->softmax->loss

    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    train_loss_history, train_acc_history = [], []
    #  epoch is the out loop means the all training dataset
    #  batch_size is the size of input
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:

            # put the x and y to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            # all training 1 need zero_grad 2 get outputs 3 calculate loss 4 backward gradient 5 update parameters
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
        print(f"Epoch {epoch + 1}: Loss={running_loss:.3f}, Acc={acc:.3f}")

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


print("\n Best combination:")
print(best_config)
print(f"Accuracy: {best_acc:.4f}")

# Best combination:
# {'batch_size': 64, 'lr': 0.001, 'dropout_rate': 0.3, 'optimizer': 'Adam'}
# Accuracy: 0.7964