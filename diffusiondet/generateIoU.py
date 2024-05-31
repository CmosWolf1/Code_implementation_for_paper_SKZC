import cv2
import torch
import random
import colorsys
import numpy as np
import torch.nn as nn

from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import dino.vision_transformer as vits
from skimage.measure import find_contours
from torchvision import transforms as pth_transforms
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

patch_size = 8
threshold = None
torch.manual_seed(1)


def generateHeatmap(img, imgSize):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__["vit_small"](patch_size=patch_size, num_classes=0)
    model.eval()
    model.to(device)
    # url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    model.load_state_dict(state_dict, strict=True)
    image_size = imgSize
    if image_size[0] * image_size[1] < 100 ** 2:
        image_size = tuple(x * 4 for x in image_size)
    elif 100 ** 2 <= image_size[0] * image_size[1] < 200 ** 2:
        image_size = tuple(x * 2 for x in image_size)
    elif 1000 ** 2 <= image_size[0] * image_size[1] < 2000 ** 2:
        image_size = tuple(x // 2 for x in image_size)
    elif 2000 ** 2 <= image_size[0] * image_size[1] < 4000 ** 2:
        image_size = tuple(x // 4 for x in image_size)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Img = Image.fromarray(img)
    Img = transform(img)
    # make the image divisible by the patch size
    w, h = Img.shape[1] - Img.shape[1] % patch_size, Img.shape[2] - Img.shape[2] % patch_size
    Img = Img[:, :w, :h].unsqueeze(0)

    w_featmap = Img.shape[-2] // patch_size
    h_featmap = Img.shape[-1] // patch_size

    attentions = model.get_last_selfattention(Img.to(device))

    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
            0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
        0].cpu().numpy()
    return attentions[5]


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def smallAndOverlyingAnchorsFilter(boxes):
    # 创建一个空列表来存储结果
    filtered_boxes = []

    # 遍历所有框
    for i in range(len(boxes)):
        current_box = boxes[i]
        is_inside = False

        # 检查当前框是否完全包含在其他框中
        for j in range(len(boxes)):
            # 跳过自身比较 且 大current_box不参与比较 且 删除被包含的 小anchors
            if i != j and \
                    abs(current_box[0] - current_box[2]) * abs(current_box[1] - current_box[3]) < \
                    abs(boxes[j][0] - boxes[j][2]) * abs(boxes[j][1] - boxes[j][3]):
                # 计算覆盖率的函数放在里面提升运行速度
                if computeIoU(current_box, boxes[j]) >= 0.80:
                    is_inside = True
                    break
            if abs(current_box[0] - current_box[2]) <= 50 or abs(current_box[1] - current_box[3]) <= 50:
                is_inside = True
                break

        # 如果当前框没有完全包含在其他框中，将其添加到结果列表
        if not is_inside:
            filtered_boxes.append(current_box)

    return filtered_boxes


def computeIoU(boxA, boxB):
    x1_A, y1_A, x2_A, y2_A = boxA
    x1_B, y1_B, x2_B, y2_B = boxB

    xA = max(x1_A, x1_B)
    yA = max(y1_A, y1_B)
    xB = min(x2_A, x2_B)
    yB = min(y2_A, y2_B)

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    area_A = (x2_A - x1_A + 1) * (y2_A - y1_A + 1)

    iou = inter_area / float(area_A)
    return iou


def adjustAnchors(boxes, size):
    for i in range(len(boxes)):
        boxes[i][0] = round(boxes[i][0] - abs(boxes[i][0] - boxes[i][2]) * 0.3)
        boxes[i][1] = round(boxes[i][1] - abs(boxes[i][1] - boxes[i][3]) * 0.3)
        boxes[i][2] = round(boxes[i][2] + abs(boxes[i][0] - boxes[i][2]) * 0.3)
        boxes[i][3] = round(boxes[i][3] + abs(boxes[i][1] - boxes[i][3]) * 0.3)
    return boxes


def find_connected_components_in_heatmap(heatmap):
    # 使用OTSU阈值创建二值图像
    ret, thresh = cv2.threshold(cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if ret > 20:
        otsu = ret - 20
    else:
        otsu = ret
    _, binary_heatmap = cv2.threshold(cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY), otsu, 255, cv2.THRESH_BINARY)
    # 标记连通组件
    labeled_heatmap, num_features = measure.label(binary_heatmap, connectivity=2, return_num=True)

    # 创建一个空列表来存储框的坐标
    object_boxes = []

    # 获取每个连通组件的坐标
    for feature in range(1, num_features + 1):
        component = (labeled_heatmap == feature)
        indices = np.where(component)

        # 计算框的左上和右下坐标
        min_y, min_x = np.min(indices, axis=1)
        max_y, max_x = np.max(indices, axis=1)
        # 过滤太小的框
        if abs(min_x - max_x) * abs(min_y - max_y) > 300:
            box = [min_x, min_y, max_x, max_y]
            object_boxes.append(box)
    resBoxes = smallAndOverlyingAnchorsFilter(object_boxes)
    resBoxes = adjustAnchors(resBoxes, heatmap.shape[0] * heatmap.shape[1])
    return resBoxes


def generateROIAnchors(boxes, size, K):
    """
    接收参数：感兴趣的区域矩形框坐标
    返回参数：感兴趣区域内的高斯噪声框坐标
    """
    # 防止dino没有找到感兴趣区域
    if (len(boxes) == 0):
        boxes = [[0, 0, size[0] - 1, size[1] - 1]]
    S = 0
    num_proposals = 0
    box = []
    for i in range(len(boxes)):
        S = S + abs(boxes[i][0]-boxes[i][2]) * abs(boxes[i][1]-boxes[i][3])
    # anchorsNumber = 500 // len(boxes)
    for i in range(len(boxes)):
        minVal = [min(boxes[i][0], boxes[i][2]), min(boxes[i][1], boxes[i][3]),
                  min(boxes[i][0], boxes[i][2]), min(boxes[i][1], boxes[i][3])]
        maxVal = [max(boxes[i][0], boxes[i][2]), max(boxes[i][1], boxes[i][3]),
                  max(boxes[i][0], boxes[i][2]), max(boxes[i][1], boxes[i][3])]
        anchorsNumber = int(((abs(boxes[i][0]-boxes[i][2])*abs(boxes[i][1]-boxes[i][3])) / S) * 500)
        num_proposals = num_proposals + anchorsNumber
        # BOX = abs(torch.randn((anchorsNumber, 4)))
        # BOX = torch.clamp(BOX, min=0, max=1)
        # for i in range(0, 4):
        #     BOX[:, i] = (maxVal[i] - minVal[i]) * BOX[:, i] + minVal[i]
        BOX = torch.randn((anchorsNumber, 4))
        BOX = torch.clamp(BOX, min=-2, max=2)
        BOX = ((BOX / 2) + 1) / 2
        BOX = box_cxcywh_to_xyxy(BOX)
        k = torch.tensor([i - j for i, j in zip(maxVal, minVal)]).unsqueeze(0)
        BOX = BOX * k
        BOX = BOX.to(dtype=torch.int)
        BOX = BOX + torch.tensor([boxes[i][0], boxes[i][1], boxes[i][0], boxes[i][1]])
        box.append(BOX)
    # img = torch.randn((500, 4))
    # img = torch.clamp(img, min=-1 * 2, max=2)
    # img = ((img / 2) + 1) / 2
    # img = box_cxcywh_to_xyxy(img)
    # img = img * K.cpu()
    box = torch.cat(box, 0).unsqueeze(0).to(torch.device('cuda'))
    # box = torch.cat((box, img.cuda()), 1).to(torch.device('cuda'))
    return box, num_proposals


# img = torch.randn(shape, device=self.device)
# img = torch.clamp(img, min=-1 * self.scale, max=self.scale)
# img = ((img / self.scale) + 1) / 2
# img = box_cxcywh_to_xyxy(img)
# img = img * images_whwh[:, None, :]


def drawAnchors(im, bo, color):
    for i in range(len(bo)):
        cv2.rectangle(im, (bo[i][0], bo[i][1]), (bo[i][2], bo[i][3]), color, 2)
