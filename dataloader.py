import cv2
from torch.utils.data.dataset import Dataset
from fc.use import *

class SSDDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, overlap_threshold=0.5):
        super(SSDDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.overlap_threshold = overlap_threshold

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        image, box = self.augmentation(self.annotation_lines[index], self.input_shape)
        image_data = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        if len(box) != 0:
            boxes = np.array(box[:, :4], dtype=np.float32)
            # scale the box
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_shape[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_shape[0]
            # one hot
            one_hot_label = np.eye(self.num_classes - 1)[np.array(box[:, 4], np.int32)]
            box = np.concatenate([boxes, one_hot_label], axis=-1)
        # box 4 coordinates + 判断是否位背景1位+ label for 2 class + 是否有物体
        box = self.assign_boxes(box)
        # 图片信息,图片所对应的先验框预测数据
        return np.array(image_data), np.array(box)

    def iou(self, box):
        inter_botleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_upright = np.minimum(self.anchors[:, 2:4], box[2:])
        inter_wh = inter_upright - inter_botleft
        inter_wh = np.maximum(inter_wh, 0)
        # interact area
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]
        # real box area
        real_box_area = (box[2] - box[0]) * (box[3] - box[1])
        # default box area
        default_box_area = (self.anchors[:, 2] - self.anchors[:, 0])*(self.anchors[:, 3] - self.anchors[:, 1])
        # iou
        union_area = real_box_area + default_box_area - inter_area
        iou = inter_area / union_area
        return iou

    def encode_box(self, box, return_iou=True, variances = [0.1, 0.1, 0.2, 0.2]):
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))
        # set threshold to find default box has larger interact
        assign_mask = iou > self.overlap_threshold
        # if no interact area larger than threshold, take the max interact
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask] # 8732 ,5
        # corresponding default box
        assigned_anchors = self.anchors[assign_mask]
        # real box center, wh
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # good default box center ,wh
        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = (assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2])
        # ssd location prediction format + iou   中心点距离和wh比例
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        return encoded_box

    def assign_boxes(self, boxes):
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1)) # 8732, 8
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        # 对每一个真实框都进行iou计算 [num_true_box, num_anchors, 4 + 1] 4 ssd predict result + iou
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4]) #  4 , 8732 ,5
        # encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        # get prior box to real box and which real box, each prior box predict one real box
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        # get prior box num
        assign_num = len(best_iou_idx)
        # eg. get 4, 50, 5 每个真实框对应的先验框
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        # all prior box coordinate
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        #   4代表为背景的概率，设定为0，因为这些先验框有对应的物体
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
        # 最后一位表示先验框中有物体
        assignment[:, -1][best_iou_mask] = 1
        return assignment

    def augmentation(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvtColor(image)

        image_w, image_h = image.size
        input_h, input_w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]]) # np.array in list can delete
        # image to input size

        scale = min(input_w / image_w, input_h / image_h)
        new_w = int(image_w * scale)
        new_h = int(image_h * scale)
        dx = (input_w - new_w) // 2
        dy = (input_h - new_h) // 2

        # add gray picture
        image = image.resize((new_w, new_h), Image.BICUBIC)
        new_image = Image.new('RGB', (input_w, input_h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)
        # modify the real box
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] / image_w * new_w + dx
            box[:, [1, 3]] = box[:, [1, 3]] / image_h * new_h + dy
            # draw = ImageDraw.Draw(new_image)
            # draw.rectangle((171,141,210,194), fill=None)
            # new_image.show()
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > input_w] = input_w
            box[:, 3][box[:, 3] > input_h] = input_h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        return image_data, box


def ssd_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    bboxes = np.array(bboxes)
    return images, bboxes
