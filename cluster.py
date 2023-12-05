import os.path
import re
import time
from PIL import Image
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, AffinityPropagation, MeanShift
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
# from cuml import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageNet, ImageFolder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from torchvision import transforms as pth_transforms
import dino.vision_transformer as vits
from torch.utils.data import ConcatDataset, DataLoader, Subset
from diffusiondet.evaluation import evaluate


def cluster_acc(y_true, y_pred):
    """

    """
    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size


patch_size = 16
X = []
y_true = []
X_train = []
y_train_true = []
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

# image_dir = ('../input/unknown/', '../input/unknown1/')
# image_files = os.listdir(image_dir[0])

transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# train_dataset = STL10(root='../input/', split='train', download=True, transform=transform)
# test_dataset = STL10(root='../input/', split='test', download=True, transform=transform)
# train_dataset = CIFAR10(train=True, download=True, root='../input/', transform=transform)
# test_dataset = CIFAR10(train=False, download=True, root='../input/', transform=transform)
train_dataset = CIFAR100(train=True, download=True, root='./datasets/', transform=transform)
test_dataset = CIFAR100(train=False, download=True, root='./datasets/', transform=transform)
# dataset = ImageFolder(root='../input/imagenet-10/', transform=transform)
# dataset = ImageFolder(root='../input/unknown/', transform=transform)
dataset = ConcatDataset([train_dataset, test_dataset])
data_loader = DataLoader(dataset, batch_size=25, num_workers=4)
data_loader_train = DataLoader(train_dataset, batch_size=25, num_workers=4)
data_loader_test = DataLoader(test_dataset, batch_size=25, num_workers=4)


for n in range(93, 100):
    dataset_cluster_performance = Subset(test_dataset, [i for i, (image, label) in enumerate(test_dataset) if label in list(range(0, n+1))])
    data_loader_cluster_performance = DataLoader(dataset_cluster_performance, batch_size=20, num_workers=4)
    print(f"\nLabel ID from 0 to {n}: \nstart to extract feature...")
    for i, (img, label) in enumerate(data_loader_cluster_performance):
        # make the image divisible by the patch size
        w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - img.shape[3] % patch_size
        img = img[:, :w, :h]
        img = img.to(device)
        feature = model(img)
        feature = feature.cpu().detach().numpy()
        X.append(feature)
        y_true.append(label)
    y_true = torch.cat(y_true, 0).numpy()
    X = np.concatenate(X, axis=0)

    # 对attention进行KMeans聚类
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    # y_pred = kmeans.predict(X)

    parameters = {'n_neighbors': range(1, 20)}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters, cv=5)  # 5-fold cross validation
    clf.fit(X, y_true)
    print(clf.best_params_['n_neighbors'])
    knn = KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'])
    knn.fit(X, y_true)
    y_pred = knn.predict(X)

    # agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    # y_pred = agg_clustering.fit_predict(X)

    # ap = AffinityPropagation(random_state=0)
    # y_pred = ap.fit_predict(X)

    nmi, ari, f, acc = evaluate(y_true, y_pred)
    print('NMI = {:.3f} ARI = {:.3f} F = {:.3f} ACC = {:.3f}'.format(nmi, ari, f, acc))
    print("start to draw...")
    data = '\nLabel ID from 0 to {:d}\nNMI = {:.3f} ARI = {:.3f} F = {:.3f} ACC = {:.3f}'.format(n, nmi, ari, f, acc)
    with open("../output/cluster_log.txt", "a") as f:
        f.write(data)

    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)

    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, s=10, cmap='inferno')

    colorbar = plt.colorbar(scatter)
    plt.tick_params(axis='both', which='major', labelsize=20)
    # plt.get_current_fig_manager().full_screen_toggle()
    plt.savefig(f"../output/figure/cluster_{n}.png", dpi=500)
    plt.clf()

    X = []
    y_true = []