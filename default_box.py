import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class DefaultBox():
    def __init__(self, input_shape, min_size, max_size, aspect_ratios):
        # 300 x 300
        self.input_shape = input_shape

        # Prior box
        self.min_size = min_size
        self.max_size = max_size

        # aspect ratios
        self.aspect_ratios = []
        for ar in aspect_ratios:
            self.aspect_ratios.append(ar)
            self.aspect_ratios.append(1.0/ar)

    def get_boxes(self, layer_shape, draw_box=False):
        # feature map size
        layer_height = layer_shape[0]
        layer_width = layer_shape[1]

        img_height = self.input_shape[0]
        img_width = self.input_shape[1]

        # get default boxes size
        box_heights = []
        box_widths = []
        for r in self.aspect_ratios:
            if r == 1 and len(box_heights) == 0: # small square box
                box_heights.append(self.min_size)
                box_widths.append(self.min_size)
            elif r == 1 and len(box_heights) > 0: # big square box
                box_heights.append(np.sqrt(self.min_size * self.max_size))
                box_widths.append(np.sqrt(self.min_size * self.max_size))
            else:
                box_heights.append(1/np.sqrt(r)*self.min_size)
                box_widths.append(np.sqrt(r) * self.min_size)


        # step spilt the figure to feature map size block
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        # print(step_x)

        # generate meshgrid 每个feature map上的中心点取框
        lin_x = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        lin_y = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)
        centers_x, centers_y = np.meshgrid(lin_x, lin_y)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        # get default box
        num_box = len(self.aspect_ratios)
        default_box = np.concatenate((centers_x, centers_y), axis=1)
        default_box = np.tile(default_box, (1, 2*num_box))  # get ready for 2 corner coordinates

        # 获取所有先验框1/2,方便后面画框
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # print(box_widths.shape)
        default_box[:, ::4] -= box_widths
        default_box[:, 1::4] -= box_heights
        default_box[:, 2::4] += box_widths
        default_box[:, 3::4] += box_heights

        if draw_box:
            plt.figure()
            plt.scatter(centers_x, centers_y, s=0.5)
            Rec = []
            # for i in range(layer_height*layer_width):
            #     for j in range(num_box):
            #         rect = plt.Rectangle([default_box[i, 4*j], default_box[i,4*j+1]],2*box_widths[j],2*box_heights[j],color='r',fill=False)
            #         Rec.append(rect)
            for j in range(num_box):
                rect = plt.Rectangle([default_box[4, 4*j], default_box[4,4*j+1]],2*box_widths[j],2*box_heights[j],color='r',fill=False)
                Rec.append(rect)
            for i in Rec:
                plt.gca().add_patch(i)
            plt.show()
        # normalization, img = width*height
        default_box[:,::2] /= img_width
        default_box[:,1::2] /= img_height
        default_box = default_box.reshape(-1,4)
        # 防止框超出图片
        default_box = np.minimum(np.maximum(default_box, 0.0), 1.0)
        return default_box # 返回default_box 左下角右上角坐标


def get_default_boxes():
    # collect all the default boxes of one figure
    input_shape = [300, 300]
    anchors_size = [30, 60, 111, 162, 213, 264, 315]
    feature_map_height = [38, 19, 10, 5, 3, 1]
    feature_map_width = [38, 19, 10, 5, 3, 1]
    aspect_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
    anchors = []
    for i in range(len(feature_map_height)):
        anchor_boxes = DefaultBox(input_shape, anchors_size[i], anchors_size[i + 1],
                                  aspect_ratios[i]).get_boxes([feature_map_height[i], feature_map_width[i]])
        anchors.append(anchor_boxes)
    anchors = np.concatenate(anchors, axis=0)
    return anchors


if __name__ == '__main__':
    input_shape = [300, 300]
    anchors_size = [30, 60, 111, 162, 213, 264, 315]
    feature_heights = [38, 19, 10, 5, 3, 1]
    feature_weights = [38, 19, 10, 5, 3, 1]
    # 4, 6 ,6,6,4,4
    aspect_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
    prior_boxes = DefaultBox(input_shape, 213, 264,[1,2])
    a = prior_boxes.get_boxes((3,3),draw_box=True)
    print(a.shape)

