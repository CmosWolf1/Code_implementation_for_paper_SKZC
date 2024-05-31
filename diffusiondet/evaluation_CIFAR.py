import random
import os

import hdbscan
from PIL import Image
import numpy
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageNet, ImageFolder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import GridSearchCV
from torchvision import transforms as pth_transforms
import dino.vision_transformer as vits
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import top_k_accuracy_score, classification_report
import torch
from sklearn.cluster import KMeans
import ast
from sklearn import metrics
from torch.utils.data import DataLoader, Subset


def select_best_k(X, k_range):
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k+1)
        knn.fit(X)
        distances, _ = knn.kneighbors(X)

        score = sum(distances[:, -1]) / len(X)
        scores.append(score)

    best_k = k_range[scores.index(min(scores))]
    return best_k


# 定义聚类的数量
n_clusters = 10  # 假设有10个未知类别
patch_size = 16
UNKNOWN_LABEL = 100
alpha = 0.75
unknown = []
y_test_true = []
y_test_pred = []
X_train = []
y_train_true = []
radiuse = []
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# build model
model = vits.__dict__["vit_small"](patch_size=patch_size, num_classes=0)
model.eval()
model.to(device)

url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"  # 16
# url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # 8
# url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"  # 16
# url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"  # 8
state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
model.load_state_dict(state_dict, strict=True)

transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
r = ['../input/unknown1000/apple/image/', '../input/unknown1000/bear/image/', '../input/unknown1000/bed/image/',
        '../input/unknown1000/cow/image/', '../input/unknown1000/giraffe/image/', '../input/unknown1000/person/image/']
# train_dataset = STL10(root='../input/', split='train', download=True, transform=transform)
# test_dataset = STL10(root='../input/', split='test', download=True, transform=transform)
train_dataset = CIFAR10(train=True, download=True, root='../input/', transform=transform)
test_dataset = CIFAR10(train=False, download=True, root='../input/', transform=transform)
# train_dataset = CIFAR100(train=True, download=True, root='../input/', transform=transform)
# test_dataset = CIFAR100(train=False, download=True, root='../input/', transform=transform)
# dataset = ConcatDataset([train_dataset, test_dataset])
# test_dataset = ImageFolder(root=r[0], transform=transform)
# dataset010 = Subset(train_dataset, [i for i, (image, label) in enumerate(train_dataset) if label in list(range(10))])
# dataset1020 = Subset(test_dataset, [i for i, (image, label) in enumerate(test_dataset) if label in list(range(10, 100))])
data_loader_train = DataLoader(train_dataset, batch_size=20, num_workers=4)
data_loader_test = DataLoader(test_dataset, batch_size=20, num_workers=4)
# data_loader_010 = DataLoader(dataset010, batch_size=20, num_workers=4)
# data_loader_1020 = DataLoader(dataset1020, batch_size=20, num_workers=4)


def normalize_feature(features, min_value, max_value):
    normalized_feature = (features - min_value) / (max_value - min_value)
    return normalized_feature


print('start to extract train datasets feature...')
for i, (img, label) in enumerate(data_loader_train):
    # make the image divisible by the patch size
    w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - img.shape[3] % patch_size
    img = img[:, :w, :h]
    img = img.to(device)
    feature = model(img)
    feature = feature.cpu().detach().numpy()
    X_train.append(feature)
    y_train_true.append(label)
y_train_true = torch.cat(y_train_true, 0).numpy()
X_train = np.concatenate(X_train, axis=0)
# normalize
# X_train = normalize_feature(X_train, np.min(X_train, axis=0), np.max(X_train, axis=0))

print("start to cluster...")
# kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_train)
# y_train_pred = kmeans.predict(X_train)
parameters = {'n_neighbors': range(1, 20)} # 搜索空间
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, parameters, cv=5) # 5-fold cross validation
clf.fit(X_train, y_train_true)
# print(clf.best_params_['n_neighbors'])
knn = KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'])
knn.fit(X_train, y_train_true)
y_train_pred = knn.predict(X_train)
# clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
# cluster_labels = clusterer.fit_predict(X_train)
# calculate cluster centers
cluster_centers = np.array([X_train[y_train_pred == i].mean(axis=0) for i in range(n_clusters)])

# calculate radius of cluster
for i, center in enumerate(cluster_centers):
    point_in_cluster = [x for x, y in zip(X_train, y_train_pred) if y == i]
    dists = [np.linalg.norm(point - center) for point in point_in_cluster]
    dists.sort()
    index_X_percent = int(len(dists) * alpha)
    radius = dists[index_X_percent]
    radiuse.append(radius)

print("start to evaluate...")
xo = 0
for i, (img, label) in enumerate(data_loader_test):
    # make the image divisible by the patch size
    w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - img.shape[3] % patch_size
    img = img[:, :w, :h]
    img = img.to(device)
    feature = model(img)
    feature = feature.cpu().detach().numpy()

    for single_feature in feature:
        distance = {}

        # single_feature = normalize_feature(single_feature, min_value, max_value)
        # recognise unknown class
        dist_list = [np.linalg.norm(single_feature - center) for center in cluster_centers]
        if all(d > r for d, r in zip(dist_list, radiuse)):
            xo+=1
            print("xo", xo)
            unknown.append(single_feature)
            y_test_pred.append(UNKNOWN_LABEL)
            continue
        for cluster_id, center in enumerate(cluster_centers):
            distance[cluster_id] = np.linalg.norm(single_feature - center)
        y_test_pred.append(min(distance, key=distance.get))
    y_test_true.append(label)
y_test_true = torch.cat(y_test_true, 0).numpy()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

report = classification_report(y_test_true, y_test_pred, labels=list(range(n_clusters)), target_names=class_names, zero_division=0, digits=3)
print(report)

