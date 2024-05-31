import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import dino.vision_transformer as vits
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as pth_transforms

SPLIT = 100
# 定义聚类的数量
n_clusters = SPLIT  # 有SPLIT个已知类别
patch_size = 16
UNKNOWN_LABEL = 200
alpha = 0.75
unknown = []
y_test_true_seen = []
y_test_pred_seen = []
y_test_true_unseen = []
y_test_pred_unseen = []
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


# 定义路径和变换
data_dir = '../input/CUB_200_2011/CUB_200_2011/'  # 指向CUB数据集的文件夹路径
transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# 获取全部类别信息
all_classes = os.listdir(os.path.join(data_dir, 'images'))
all_classes.sort()
# 假设我们已经知道哪些是Seen类别，哪些是Unseen类别
seen_classes = all_classes[:SPLIT]    # 取前180个作为seen类别
unseen_classes = all_classes[SPLIT:]  # 剩下的作为unseen类别

# 创建映射，方便后续操作
class_to_idx = {cls_name.split('.')[0]: int(cls_name.split('.')[0]) for cls_name in all_classes}

# 创建数据集实例
full_dataset = ImageFolder(root=os.path.join(data_dir, 'images'), transform=transform)
full_dataset.class_to_idx = class_to_idx
# 根据类名筛选图像索引并分配给seen和unseen
seen_indices = []
unseen_indices = []

for idx, (path, _) in enumerate(full_dataset.samples):
    class_id = int(path.split('/')[-2].split('.')[0])
    if class_id <= SPLIT:
        seen_indices.append(idx)
    else:
        unseen_indices.append(idx)

# 对seen类别进行训练和测试划分
train_idx, test_idx = train_test_split(seen_indices, test_size=0.5, stratify=[full_dataset.targets[i] for i in seen_indices])

# 使用Subset创建训练和测试子集
train_dataset = Subset(full_dataset, train_idx)
test_seen_dataset = Subset(full_dataset, test_idx)  # 测试集seen
test_unseen_dataset = Subset(full_dataset, unseen_indices)  # 测试集unseen

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4)
test_seen_loader = DataLoader(test_seen_dataset, batch_size=20, shuffle=False, num_workers=4)
test_unseen_loader = DataLoader(test_unseen_dataset, batch_size=20, shuffle=False, num_workers=4)

print('start to extract train datasets feature...')
for i, (img, label) in enumerate(train_loader):
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
cluster_centers = np.array([X_train[y_train_pred == i].mean(axis=0) for i in range(n_clusters)])

# calculate radius of cluster
for i, center in enumerate(cluster_centers):
    point_in_cluster = [x for x, y in zip(X_train, y_train_pred) if y == i]
    dists = [np.linalg.norm(point - center) for point in point_in_cluster]
    dists.sort()
    index_X_percent = int(len(dists) * alpha)
    if index_X_percent:
        radius = dists[index_X_percent]
    else: radius = 0
    radiuse.append(radius)


print("start to evaluate seen classes...")
for i, (img, label) in enumerate(test_seen_loader):
    # make the image divisible by the patch size
    w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - img.shape[3] % patch_size
    img = img[:, :w, :h]
    img = img.to(device)
    feature = model(img)
    feature = feature.cpu().detach().numpy()
    for single_feature in feature:
        distance = {}
        dist_list = [np.linalg.norm(single_feature - center) for center in cluster_centers]
        if all(d > r for d, r in zip(dist_list, radiuse)):
            y_test_pred_seen.append(UNKNOWN_LABEL)
        else:
            for cluster_id, center in enumerate(cluster_centers):
                distance[cluster_id] = np.linalg.norm(single_feature - center)
            y_test_pred_seen.append(min(distance, key=distance.get))
    y_test_true_seen.append(label)

print("start to evaluate unseen classes...")
for i, (img, label) in enumerate(test_unseen_loader):
    # make the image divisible by the patch size
    w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - img.shape[3] % patch_size
    img = img[:, :w, :h]
    img = img.to(device)
    feature = model(img)
    feature = feature.cpu().detach().numpy()

    for single_feature in feature:
        distance = {}
        dist_list = [np.linalg.norm(single_feature - center) for center in cluster_centers]
        if all(d > r for d, r in zip(dist_list, radiuse)):
            y_test_pred_unseen.append(UNKNOWN_LABEL)
        else:
            for cluster_id, center in enumerate(cluster_centers):
                distance[cluster_id] = np.linalg.norm(single_feature - center)
            y_test_pred_unseen.append(min(distance, key=distance.get))
    y_test_true_unseen.append(torch.full_like(label, UNKNOWN_LABEL))

# 转换为 NumPy 数组
y_test_true_seen = torch.cat(y_test_true_seen, 0).numpy()
y_test_pred_seen = np.array(y_test_pred_seen)

y_test_true_unseen = torch.cat(y_test_true_unseen, 0).numpy()
y_test_pred_unseen = np.array(y_test_pred_unseen)

# 计算已知类别的精度
seen_accuracy = accuracy_score(y_test_true_seen, y_test_pred_seen)

# 计算未知类别的精度
unseen_accuracy = accuracy_score(y_test_true_unseen, y_test_pred_unseen)

# 计算调和平均
if seen_accuracy + unseen_accuracy > 0:
    hmean = 2 * (seen_accuracy * unseen_accuracy) / (seen_accuracy + unseen_accuracy)
else:
    hmean = 0

print(f"Seen class accuracy (Sacc): {seen_accuracy:.3f}")
print(f"Unseen class accuracy (Uacc): {unseen_accuracy:.3f}")
print(f"Harmonic mean (Hmean): {hmean:.3f}")
