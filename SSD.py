import torch
from default_box import *
import numpy as np
from fc.use import *
import colorsys
import ssd_model
import torch.backends.cudnn as cudnn
from PIL import Image,ImageDraw, ImageFont
import cv2

class SSD():
    defaults = {
        "model_path": './weight/trained_weight.pth',
        "class_name": ['face', 'face_mask'],
        "class_num" : 2,
        "input_shape": [300, 300],
        "confidence": 0.3,
        "nms_iou": 0.45,
        'anchors_size': [30, 60, 111, 162, 213, 264, 315],
        "cuda": True,
        "letterbox_image" : True
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self.defaults)
        self.default_box= torch.from_numpy(get_default_boxes()).type(torch.FloatTensor).cuda() # all default box coordinates
        self.class_num = self.class_num + 1
        # 图框设置
        hsv_tuples = [(x / self.class_num, 1., 1.) for x in range(self.class_num)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(self.class_num)
        self.generate()

    def generate(self):
        self.model = ssd_model.SSD300(self.class_num)
        device = torch.device('cuda')
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model = self.model.eval()
        if self.cuda:
            cudnn.benchmark = True
            self.model = torch.nn.DataParallel(self.model).cuda()

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image) # gray to color
        image_data = resize_image(image,self.letterbox_image)
        # preprocess
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # print(image_data.shape)
        with torch.no_grad():
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            outputs = self.model(images)
            # get box on image
            results = self.bbox_util.decode_box(outputs,self.default_box, self.confidence, self.nms_iou, image_shape, self.letterbox_image)
            if len(results[0]) <= 0:
                return image

            top_label = np.array(results[0][:, 4], dtype='int32')
            top_conf = results[0][:, 5]
            top_boxes = results[0][:, :4]
        font = ImageFont.truetype(font='simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        # plot image
        for i, c in list(enumerate(top_label)):
            class_name = ['no mask', 'mask']
            predicted_class = class_name[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image



if __name__ == '__main__':
    a = SSD()
    img = 'img/test_00001153.jpg'
    img = Image.open(img)
    b = a.detect_image(img)
    # b.show()



