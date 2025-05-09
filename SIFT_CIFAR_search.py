import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from itertools import product
import joblib


def load_cifar10_subset(selected_classes, max_per_class, size=(128, 128)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    cifar = CIFAR10(root='./data', train=True, download=True, transform=transform)

    images, labels = [], []
    class_counts = {i: 0 for i in selected_classes}

    for img, label in cifar:
        if label in selected_classes and class_counts[label] < max_per_class:
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(img)
            labels.append(label)
            class_counts[label] += 1
        if all(class_counts[i] >= max_per_class for i in selected_classes):
            break

    label_to_new = {label: i for i, label in enumerate(selected_classes)}
    labels = np.array([label_to_new[l] for l in labels])
    return np.array(images), np.array(labels)


def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in tqdm(images, desc="Extracting SIFT"):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        descriptors_list.append(desc if desc is not None else np.zeros((1, 128)))
    return descriptors_list


def build_bow_histograms(features, kmeans, k):
    histograms = []
    for desc in features:
        if desc is not None and len(desc) > 0:
            labels = kmeans.predict(desc.astype(np.float64))
            hist, _ = np.histogram(labels, bins=np.arange(k + 1))
        else:
            hist = np.zeros(k)
        histograms.append(hist)
    return np.array(histograms)


def run_sift_gridsearch(selected_classes, max_per_class, k_list, kernel_list, C_list,
                        save_dir="sift_cifar10_withoutColor_results"):
    os.makedirs(save_dir, exist_ok=True)
    images, labels = load_cifar10_subset(selected_classes, max_per_class)
    sift_features = extract_sift_features(images)

    best_acc = 0
    best_model_data = {}

    for k, kernel, C in product(k_list, kernel_list, C_list):
        setting = f"k={k}_kernel={kernel}_C={C}"
        print(f"\n🚀 正在运行配置: {setting}")

        all_desc = np.vstack([d.astype(np.float64) for d in sift_features if d is not None and len(d) > 0])
        kmeans = KMeans(n_clusters=k, random_state=42).fit(all_desc)
        bow_features = build_bow_histograms(sift_features, kmeans, k)

        X_train, X_test, y_train, y_test = train_test_split(bow_features, labels, test_size=0.2, stratify=labels)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SVC(kernel=kernel, C=C)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        setting_dir = os.path.join(save_dir, setting)
        os.makedirs(setting_dir, exist_ok=True)

        with open(os.path.join(setting_dir, "report.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report)

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap='Blues')
        plt.title(f"Confusion Matrix ({setting})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(setting_dir, "confusion_matrix.png"))
        plt.close()

        print(f" 已保存结果: {setting}")

        if acc > best_acc:
            best_acc = acc
            best_model_data = {
                'clf': clf,
                'scaler': scaler,
                'kmeans': kmeans,
                'k': k,
                'kernel': kernel,
                'C': C
            }

    # 保存最佳模型
    best_folder = f"best_SIFT_CIFAR10_k-{best_model_data['k']}_kernel-{best_model_data['kernel']}_C-{best_model_data['C']}"
    best_path = os.path.join(save_dir, best_folder)
    os.makedirs(best_path, exist_ok=True)

    joblib.dump(best_model_data['clf'], os.path.join(best_path, "svm_model.pkl"))
    joblib.dump(best_model_data['scaler'], os.path.join(best_path, "scaler.pkl"))
    joblib.dump(best_model_data['kmeans'], os.path.join(best_path, "kmeans.pkl"))

    with open(os.path.join(best_path, "best_config.txt"), "w") as f:
        f.write(f"Best Accuracy: {best_acc:.4f}\n")
        f.write(f"k={best_model_data['k']}, kernel={best_model_data['kernel']}, C={best_model_data['C']}\n")

    print(f"\n 最佳模型已保存到: {best_path}")

# ========== 示例调用 ==========
run_sift_gridsearch(
    selected_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    max_per_class=1000,
    k_list=[100, 200, 300],
    kernel_list=['linear', 'rbf'],
    C_list=[1, 10]
)