import random
import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageNet, ImageFolder
from sklearn.neighbors import KNeighborsClassifier
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


# 定义聚类的数量
n_clusters = 10  # 假设有10个未知类别
patch_size = 16
UNKNOWN_LABEL = 100
unknown = [0] * 100
y_test_true = []
y_test_pred = []
X_train = []
y_train_true = []
radiuse = [[] for _ in range(100)]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# build model
model = vits.__dict__["vit_base"](patch_size=patch_size, num_classes=0)
model.eval()
model.to(device)

# url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"  # 16
# url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # 8
url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"  # 16
# url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"  # 8
state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
model.load_state_dict(state_dict, strict=True)

transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
train_dataset = CIFAR10(train=False, download=True, root='../input/', transform=transform)
# test_dataset = CIFAR100(train=False, download=True, root='../input/', transform=transform)
test_dataset1 = CIFAR100(train=False, download=True, root='../input/', transform=transform)
test_dataset2 = CIFAR100(train=True, download=True, root='../input/', transform=transform)
test_dataset = ConcatDataset([test_dataset1, test_dataset2])
# dataset010 = Subset(train_dataset, [i for i, (image, label) in enumerate(train_dataset) if label in list(range(10))])
# dataset1020 = Subset(test_dataset, [i for i, (image, label) in enumerate(test_dataset) if label in list(range(10, 100))])
data_loader_train = DataLoader(train_dataset, batch_size=20, num_workers=4)
data_loader_test = DataLoader(test_dataset, batch_size=20, num_workers=4)
# data_loader_010 = DataLoader(dataset010, batch_size=20, num_workers=4)
# data_loader_1020 = DataLoader(dataset1020, batch_size=20, num_workers=4)
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

print("start to cluster...")
parameters = {'n_neighbors': range(1, 20)} # 搜索空间
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, parameters, cv=5) # 5-fold cross validation
clf.fit(X_train, y_train_true)
knn = KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'])
knn.fit(X_train, y_train_true)
y_train_pred = knn.predict(X_train)

# calculate cluster centers
cluster_centers = np.array([X_train[y_train_true == i].mean(axis=0) for i in range(n_clusters)])

# calculate radius of cluster
for k in np.arange(0, 1, 0.01):
    for i, center in enumerate(cluster_centers):
        point_in_cluster = [x for x, y in zip(X_train, y_train_pred) if y == i]
        dists = [np.linalg.norm(point - center) for point in point_in_cluster]
        dists.sort()
        index_X_percent = int(len(dists) * k)
        radius = dists[index_X_percent]
        radiuse[i].append(radius)

print("start to evaluate...")
for i, (img, label) in enumerate(data_loader_test):
    # make the image divisible by the patch size
    w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - img.shape[3] % patch_size
    img = img[:, :w, :h]
    img = img.to(device)
    feature = model(img)
    feature = feature.cpu().detach().numpy()

    for k in range(100):
        for single_feature in feature:
            distance = {}
            dist_list = [np.linalg.norm(single_feature - center) for center in cluster_centers]
            if all(d > r for d, r in zip(dist_list, radiuse[k])):
                unknown[k] += 1
print(unknown)
for i in np.arange(0, 1, 0.01):
    print(f'When k = {i}, unknown:{unknown[int(i * 100)]}')
print('start to draw...')
unknown = [i / 10000 for i in unknown]
x = np.arange(0, 1, 0.01)
plt.plot(x, unknown)
plt.title('unknown')
plt.xlabel('percent')
plt.ylabel('accuracy')
plt.show()

