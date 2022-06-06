from SSD import SSD
import ssd_model
import torch
from PIL import Image
from fc.use import *
import cv2
import time

def pic_predict():
    model = SSD()
    image = Image.open('./mask_or_not/JPEGImages/test_00000408.jpg')
    image = model.detect_image(image)
    image.show()
    # image.save('./img/test.jpg')


def video_predict(path=None):
    model = SSD()
    capture = cv2.VideoCapture(0)
    # path = './img/video.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # out = cv2.VideoWriter(path, fourcc, 10, size)
    while (True):
        ref, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(model.detect_image(frame))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("video", frame)
        # out.write(frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            capture.release()
            break




if __name__ == '__main__':
    mode = 'pic'
    if mode == 'pic':
        pic_predict()

    if mode == 'video':
        video_predict()


