import os
import argparse
import torch
import torchvision
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import GridSearchCV
from CC import resnet, network, transform
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageNet, ImageFolder
from torchvision import transforms as pth_transforms
from sklearn.metrics import top_k_accuracy_score, classification_report
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, AffinityPropagation, MeanShift
from torch.utils import data
import copy

radiuse= []
UNKNOWN_LABEL = 100
unknown = []
y_test_true = []
y_test_pred = []

transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
train_dataset = torchvision.datasets.CIFAR10(
            root='../input',
            train=True,
            download=True,
            transform=transform
        )
test_dataset = torchvision.datasets.CIFAR10(
            root='../input',
            train=False,
            download=True,
            transform=transform
        )
data_loader_train = DataLoader(
        train_dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )
data_loader_test = DataLoader(
        test_dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res = resnet.get_resnet('ResNet34')
model = network.Network(res, 128, 10)
model_fp = os.path.join("save/CIFAR-10", "checkpoint_{}.tar".format(1000))
model.load_state_dict(torch.load(model_fp, map_location='cuda')['net'])
model.to(device)


def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    feature_from_resnet = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c, h = model.forward_cluster(x)
        c = c.detach()
        h = h.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        feature_from_resnet.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    feature_from_resnet = np.array(feature_from_resnet)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector, feature_from_resnet


print("### Creating features from model ###")
X, Y, F = inference(data_loader_train, model, device)

# kmeans = KMeans(n_clusters=10, random_state=0).fit(F)
# y_pred = kmeans.predict(F)

# parameters = {'n_neighbors': range(1, 20)} # 搜索空间
# knn = KNeighborsClassifier()
# clf = GridSearchCV(knn, parameters, cv=5) # 5-fold cross validation
# clf.fit(F, Y)
# # print(clf.best_params_['n_neighbors'])
# knn = KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'])
# knn.fit(F, Y)
# y_pred = knn.predict(F)

cluster_centers = np.array([F[y_pred == i].mean(axis=0) for i in range(10)])
# calculate radius of cluster
for i, center in enumerate(cluster_centers):
    point_in_cluster = [x for x, y in zip(F, y_pred) if y == i]
    dists = [np.linalg.norm(point - center) for point in point_in_cluster]
    dists.sort()
    index_X_percent = int(len(dists) * 0.75)
    radius = dists[index_X_percent]
    radiuse.append(radius)

feature_from_resnet = []
for i, (img, label) in enumerate(data_loader_test):
    img = img.to(device)
    with torch.no_grad():
        c, h = model.forward_cluster(img)
    h = h.detach()
    feature_from_resnet = h.cpu().detach().numpy()
    for single_feature in feature_from_resnet:
        distance = {}
        # single_feature = normalize_feature(single_feature, min_value, max_value)
        # recognise unknown class
        dist_list = [np.linalg.norm(single_feature - center) for center in cluster_centers]
        if all(d > r for d, r in zip(dist_list, radiuse)):
            unknown.append(single_feature)
            y_test_pred.append(UNKNOWN_LABEL)
        else:
            for cluster_id, center in enumerate(cluster_centers):
                distance[cluster_id] = np.linalg.norm(single_feature - center)
            y_test_pred.append(min(distance, key=distance.get))
    y_test_true.append(label)
y_test_true = torch.cat(y_test_true, 0).numpy()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
report = classification_report(y_test_true, y_test_pred, labels=list(range(10)), target_names=class_names, zero_division=0, digits=3)
print(report)