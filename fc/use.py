import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms


class BBoxUtility():
    def __init__(self, class_num):
        self.class_num = class_num

    def decode_box(self, predictions, default_box, confidence, nms_iou, image_shape, letterbox_image, input_shape=[300,300]):
        mbox_loc = predictions[0]
        # get confidence of all the classes
        mbox_conf = nn.Softmax(-1)(predictions[1])
        results = []
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i], default_box, variances=[0.1, 0.2]) # real box coordinates
            for j in range(1, self.class_num):
                correct_confs = mbox_conf[i, :, j]
                correct_confs_m = correct_confs > confidence
                if len(correct_confs[correct_confs_m]) > 0:
                    boxes_to_process = decode_bbox[correct_confs_m] # get correct box
                    confs_to_process = correct_confs[correct_confs_m]
                    # use nms to get good box
                    keep = nms(boxes_to_process, confs_to_process, nms_iou)
                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (j - 1) * torch.ones((len(keep), 1)).cuda()
                    pre = torch.cat((good_boxes, labels, confs),dim=1).cpu().numpy()
                    results[-1].extend(pre)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1]) # 4 coordinates + label index + confidence
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4]) / 2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.correct_boxes(box_xy, box_wh, image_shape, letterbox_image, input_shape)
        return results

    def decode_boxes(self, mbox_loc, default_box, variances=[0.1, 0.2]):
        # get default box center coordinates and w,h
        default_w = default_box[:, 2] - default_box[:, 0]
        default_h = default_box[:, 3] - default_box[:, 1]
        default_center_x = 0.5 * (default_box[:, 2] + default_box[:, 0])
        default_center_y = 0.5 * (default_box[:, 3] + default_box[:, 1])

        # real box bias to default box
        decode_bbox_centerx = mbox_loc[:, 0] * default_w * variances[0]
        decode_bbox_centerx += default_center_x
        decode_bbox_centery = mbox_loc[:, 1] * default_h * variances[0]
        decode_bbox_centery += default_center_y

        # real box w,h
        decode_bbox_width = torch.exp(mbox_loc[:, 2] * variances[1])
        decode_bbox_width *= default_w
        decode_bbox_height = torch.exp(mbox_loc[:, 3] *variances[1])
        decode_bbox_height *= default_h

        # two corner coordinates
        decode_bbox_xmin = decode_bbox_centerx - 0.5 * decode_bbox_width
        decode_bbox_xmax = decode_bbox_centerx + 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_centery - 0.5 * decode_bbox_height
        decode_bbox_ymax = decode_bbox_centery + 0.5 * decode_bbox_height
        decode_bbox = torch.cat((decode_bbox_xmin[:, None],
                                 decode_bbox_ymin[:, None],
                                 decode_bbox_xmax[:, None],
                                 decode_bbox_ymax[:, None]), dim=1)

        decode_bbox = torch.min(torch.max(decode_bbox, torch.zeros_like(decode_bbox)), torch.ones_like(decode_bbox))
        # real box coordinates
        return decode_bbox

    def correct_boxes(self, box_xy, box_wh, image_shape, letterbox_image, input_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)
        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape
            box_yx = (box_yx - offset) * scale
            box_hw *= scale
        box_mins = box_yx - 0.5 * box_hw
        box_maxes = box_yx + 0.5 * box_hw
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        # box in image
        return boxes



def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        return image.convert('RGB')

def resize_image(image, letterbox_image=False):
    image_width, image_height = image.size
    w, h = (300, 300)
    if letterbox_image:
        scale = min(w / image_width, h / image_height)
        new_w = int(image_width * scale)
        new_h = int(image_height * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)
        new_image = Image.new('RGB', (300,300), (128, 128, 128))
        new_image.paste(image, ((w - new_w) // 2, (h - new_h) // 2))#等比例缩放
    else:
        new_image = image.resize((w, h), Image.BICUBIC) #正常缩放
    return new_image

def preprocess_input(inputs):
    MEANS = (104, 117, 123)
    return inputs - MEANS


def weights_init(net):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    net.apply(init_func)
