import random
import os
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

n_clusters = 10
patch_size = 8
UNKNOWN_LABEL = 100
unknown = []
y_test_true = []
y_test_pred = []
X_train = []
y_train_true = []
radiuse = []
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# build model
model = vits.__dict__["vit_base"](patch_size=patch_size, num_classes=0)
model.eval()
model.to(device)

# url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"  # 16
# url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # 8
# url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"  # 16
url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"  # 8
state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
model.load_state_dict(state_dict, strict=True)

transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
r = ['../input/unknown1000/apple/image/', '../input/unknown1000/bear/image/', '../input/unknown1000/bed/image/',
        '../input/unknown1000/cow/image/', '../input/unknown1000/giraffe/image/', '../input/unknown1000/person/image/']
# train_dataset = STL10(root='./datasets/', split='train', download=True, transform=transform)
# test_dataset = STL10(root='./datasets/', split='test', download=True, transform=transform)
train_dataset = CIFAR10(train=True, download=True, root='./datasets/', transform=transform)
# test_dataset = CIFAR10(train=False, download=True, root='./datasets/', transform=transform)
# train_dataset = CIFAR100(train=True, download=True, root='./datasets/', transform=transform)
test_dataset = CIFAR100(train=False, download=True, root='./datasets/', transform=transform)
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
parameters = {'n_neighbors': range(1, 20)}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, parameters, cv=5) # 5-fold cross validation
clf.fit(X_train, y_train_true)
# print(clf.best_params_['n_neighbors'])
knn = KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'])
knn.fit(X_train, y_train_true)
y_train_pred = knn.predict(X_train)

# calculate cluster centers
cluster_centers = np.array([X_train[y_train_true == i].mean(axis=0) for i in range(n_clusters)])

# calculate radius of cluster
for i, center in enumerate(cluster_centers):
    point_in_cluster = [x for x, y in zip(X_train, y_train_pred) if y == i]
    dists = [np.linalg.norm(point - center) for point in point_in_cluster]
    dists.sort()
    index_X_percent = int(len(dists) * 0.75)
    radius = dists[index_X_percent]
    radiuse.append(radius)


# def extract_subset(dataset, classes):
#     """提取包含指定类别的子集"""
#     class_indices = [i for i, label in enumerate(dataset.targets) if label in classes]
#     subset = torch.utils.data.Subset(dataset, class_indices)
#     return subset
#
# ######################################### NEW ###########################################
# # 分割 CIFAR100 数据集为 10 个子集
# num_classes = 100
# subset_size = num_classes // 10
# subsets = {}
# all_predicted_labels = []
# all_true_labels = []
# cluster_centers = [[] for _ in range(10)]
# max_feature_of_cluster = [[] for _ in range(10)]
# min_feature_of_cluster = [[] for _ in range(10)]
# # radiuse = [[[] for _ in range(10)] for _ in range(10)]
# radiuse = [[] for _ in range(10)]
#
# for i in range(0, num_classes, subset_size):
#     classes = list(range(i, i + subset_size))
#     subsets[i] = extract_subset(train_dataset, classes)
#
# counter = 0
# for i, subset in subsets.items():
#     # 创建 DataLoader
#     data_loader = torch.utils.data.DataLoader(subset, batch_size=20, shuffle=False)
#
#     X_train = []
#     y_train_true = []
#
#     print(f'Start clustering for classes {i} to {i + subset_size - 1}...')
#
#     # 提取特征和标签
#     for j, (img, label) in enumerate(data_loader):
#         # make the image divisible by the patch size
#         w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - img.shape[3] % patch_size
#         img = img[:, :w, :h]
#         img = img.to(device)
#         feature = model(img)
#         feature = feature.cpu().detach().numpy()
#         X_train.append(feature)
#         y_train_true.append(label)
#     y_train_true = torch.cat(y_train_true, 0).numpy()
#     X_train = np.concatenate(X_train, axis=0)
#
#     # 进行KNN聚类
#     parameters = {'n_neighbors': range(1, 20)}
#     knn = KNeighborsClassifier()
#     clf = GridSearchCV(knn, parameters, cv=5)
#     clf.fit(X_train, y_train_true)
#     knn = KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'])
#     knn.fit(X_train, y_train_true)
#     y_train_pred = knn.predict(X_train)
#
#     # kmeans = KMeans(n_clusters=10, random_state=0).fit(X_train)
#     # y_train_pred = kmeans.predict(X_train)
#
#     all_predicted_labels.extend(y_train_pred)
#     all_true_labels.extend(y_train_true)
#
#     # 计算簇中心和半径
#     # calculate cluster centers
#     cluster_centers[counter].extend([
#         X_train[y_train_true == k].mean(axis=0)
#         for k in range(subset_size * counter, subset_size * (counter + 1))
#     ])
#     max_feature_of_cluster[counter].extend([
#         np.max(X_train[y_train_true == k], axis=0)
#         for k in range(subset_size * counter, subset_size * (counter + 1))
#     ])
#     min_feature_of_cluster[counter].extend([
#         np.min(X_train[y_train_true == k], axis=0)
#         for k in range(subset_size * counter, subset_size * (counter + 1))
#     ])
#
#     # calculate radius of cluster
#     for j, center in enumerate(cluster_centers[counter]):
#         point_in_cluster = [x for x, y in zip(X_train, y_train_pred) if y == j + counter * subset_size]
#         dists = [np.linalg.norm(point - center) for point in point_in_cluster]
#         dists.sort()
#         index_X_percent = int(len(dists) * 0.9)
#         radius = dists[index_X_percent]
#         radiuse[counter].append(radius)
#     counter += 1
# # 汇总所有预测和真实标签
# all_predicted_labels = np.array(all_predicted_labels)
# all_true_labels = np.array(all_true_labels)
# cluster_centers = [item for item in cluster_centers]
# cluster_centers = np.vstack(cluster_centers)
# radiuse = [item for sublist in radiuse for item in sublist]
# # max_feature_of_cluster = np.array(max_feature_of_cluster)
# # min_feature_of_cluster = np.array(min_feature_of_cluster)
# X_train = []
# y_train_true = []
# print("start to evaluate...")
# for i, (img, label) in enumerate(data_loader_test):
#     # make the image divisible by the patch size
#     w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - img.shape[3] % patch_size
#     img = img[:, :w, :h]
#     img = img.to(device)
#     feature = model(img)
#     feature = feature.cpu().detach().numpy()
#
#     for single_feature in feature:
#         assigned = False  # Flag to check if we have found a cluster within radius
#         distance = {}
#         # Calculate distances to all centers in this cluster
#         dist_list = [np.linalg.norm(single_feature - center) for center in cluster_centers]
#         # dist_list = [np.linalg.norm(single_feature[i] - cluster_centers[i*10: (i+1)*10]) for i in range(10)]
#         # Check if any distance is within radius
#         if all(d > r for d, r in zip(dist_list, radiuse)):
#             unknown.append(single_feature)
#             y_test_pred.append(UNKNOWN_LABEL)
#             continue
#         for cluster_id, center in enumerate(cluster_centers):
#             distance[cluster_id] = np.linalg.norm(single_feature - center)
#         y_test_pred.append(min(distance, key=distance.get))
#
#     y_test_true.append(label)
#
# y_test_true = torch.cat(y_test_true, 0).numpy()

# normalize
# min_value, max_value = np.min(X_train, axis=0), np.max(X_train, axis=0)

print("start to evaluate...")
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
            unknown.append(single_feature)
            y_test_pred.append(UNKNOWN_LABEL)
            continue
        for cluster_id, center in enumerate(cluster_centers):
            distance[cluster_id] = np.linalg.norm(single_feature - center)
        y_test_pred.append(min(distance, key=distance.get))
    y_test_true.append(label)
y_test_true = torch.cat(y_test_true, 0).numpy()

# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm',
]
report = classification_report(y_test_true, y_test_pred, labels=list(range(n_clusters)), target_names=class_names, zero_division=0, digits=3)
print(report)

