import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import joblib

# 数据加载
def load_icubworld_dataset(dataset_path, size=(128,128), max_per_class=3000, selected_classes=10):
    class_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])[:selected_classes]

    images, labels = [], []
    for label_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(dataset_path, class_name)
        image_paths = []
        for root, _, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm')):
                    image_paths.append(os.path.join(root, file))
            if len(image_paths) >= max_per_class:
                break
        image_paths = image_paths[:max_per_class]

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB').resize(size)
                images.append(np.array(img))
                labels.append(label_idx)
            except:
                continue
    return np.array(images), np.array(labels)

# SIFT 特征提取
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in tqdm(images, desc="Extracting SIFT"):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        descriptors_list.append(desc if desc is not None else np.zeros((1, 128)))
    return descriptors_list

# 构建 BoW
def build_bow_histograms(features, kmeans, k):
    histograms = []
    for desc in features:
        if desc is not None and len(desc) > 0:
            labels = kmeans.predict(desc)
            hist, _ = np.histogram(labels, bins=np.arange(k+1))
        else:
            hist = np.zeros(k)
        histograms.append(hist)
    return np.array(histograms)

# 提取颜色直方图
def extract_color_histograms(images, bins=32):
    histograms = []
    for img in images:
        hist_r = np.histogram(img[:,:,0], bins=bins, range=(0,255))[0]
        hist_g = np.histogram(img[:,:,1], bins=bins, range=(0,255))[0]
        hist_b = np.histogram(img[:,:,2], bins=bins, range=(0,255))[0]
        hist = np.concatenate([hist_r, hist_g, hist_b])
        histograms.append(hist)
    return np.array(histograms)

# 混淆矩阵保存
def save_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 网格搜索主逻辑
def run_grid_search(train_path, test_path, k_list, bins_list, kernels, C_list, selected_classes=7, save_dir="SIFT_iCubWorld_color_results"):
    os.makedirs(save_dir, exist_ok=True)

    train_images, train_labels = load_icubworld_dataset(train_path, selected_classes=selected_classes)
    test_images, test_labels = load_icubworld_dataset(test_path, selected_classes=selected_classes)

    best_acc = 0
    best_config = ""
    best_model_data = {}

    for k, bins, kernel, C in product(k_list, bins_list, kernels, C_list):
        setting = f"k={k}_bins={bins}_kernel={kernel}_C={C}"
        print(f"\n🚀 正在运行配置: {setting}")

        train_sift = extract_sift_features(train_images)
        test_sift = extract_sift_features(test_images)
        all_desc = np.vstack([d for d in train_sift if d is not None and len(d) > 0])
        kmeans = KMeans(n_clusters=k, random_state=42).fit(all_desc)
        bow_train = build_bow_histograms(train_sift, kmeans, k)
        bow_test = build_bow_histograms(test_sift, kmeans, k)

        color_train = extract_color_histograms(train_images, bins=bins)
        color_test = extract_color_histograms(test_images, bins=bins)

        X_train = np.hstack([bow_train, color_train])
        X_test = np.hstack([bow_test, color_test])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SVC(kernel=kernel, C=C)
        clf.fit(X_train, train_labels)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(test_labels, y_pred)
        report = classification_report(test_labels, y_pred)
        cm = confusion_matrix(test_labels, y_pred)

        with open(os.path.join(save_dir, f"{setting}_report.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report)

        cm_path = os.path.join(save_dir, f"{setting}_confusion_matrix.png")
        save_confusion_matrix(cm, [str(i) for i in range(selected_classes)], f"Confusion Matrix ({setting})", cm_path)
        print(f"✅ 已保存结果: {setting}")

        if acc > best_acc:
            best_acc = acc
            best_config = setting
            best_model_data = {
                'clf': clf,
                'scaler': scaler,
                'kmeans': kmeans,
                'k': k,
                'bins': bins,
                'kernel': kernel,
                'C': C
            }

    # 保存最佳模型
    best_folder = f"best_SIFT_iCubWorld_color_{best_config.replace('=', '-').replace(',', '')}"
    best_path = os.path.join(save_dir, best_folder)
    os.makedirs(best_path, exist_ok=True)

    joblib.dump(best_model_data['clf'], os.path.join(best_path, "svm_model.pkl"))
    joblib.dump(best_model_data['scaler'], os.path.join(best_path, "scaler.pkl"))
    joblib.dump(best_model_data['kmeans'], os.path.join(best_path, "kmeans.pkl"))

    with open(os.path.join(best_path, "best_config.txt"), "w") as f:
        f.write(f"Best Accuracy: {best_acc:.4f}\n")
        f.write(f"Best Config: {best_config}\n")

    print(f"\n🏆 最佳模型已保存到: {best_path}")

# 示例调用
run_grid_search(
    train_path="../iCubWorld28/train_all_flat",
    test_path="../iCubWorld28/test_all_flat",
    k_list=[100, 200, 300],
    bins_list=[16, 32],
    kernels=['linear', 'rbf'],
    C_list=[0.1,1,10],
    selected_classes=7
)