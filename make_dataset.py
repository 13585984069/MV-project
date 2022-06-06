import os
import random
import xml.etree.ElementTree as ET
from PIL import Image



def convert_annotation(image_id, list_file):
    file = open('mask_or_not/Annotations/%s.xml'%image_id, encoding='utf-8')
    tree = ET.parse(file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)
    train_val_percent = 0.9
    train_percent = 0.9
    classes = ['face', 'face_mask']
    xmlfilepath = 'mask_or_not/Annotations'
    saveBasePath = 'mask_or_not/ImageSets/Test'
    total_xml = os.listdir(xmlfilepath)

    # print(len)
    train_val_num = int(len(total_xml) * train_val_percent)
    train_num = int(train_val_num * train_percent)
    train_val_index = random.sample(range(len(total_xml)), train_val_num)
    train_index = random.sample(train_val_index, train_num)
    #
    print("train and val size:", train_val_num)
    print("train size:", train_num)

    write = False
    if write:
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
        #
        train_val_data = []
        train_data = []
        val_data = []
        test_data = []
        for i in range(len(total_xml)):
            name = total_xml[i][:-4] + '\n'
            if i in train_val_index:
                train_val_data.append(name)
                if i in train_index:
                    train_data.append(name)
                else:
                    val_data.append(name)
            else:
                test_data.append(name)
        random.shuffle(train_val_data)
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        for i in train_val_data:
            ftrainval.write(i)
        for i in train_data:
            ftrain.write(i)
        for i in val_data:
            fval.write(i)
        for i in test_data:
            ftest.write(i)


        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

        create_data = ['train.txt', 'val.txt','test.txt']

        for i in create_data:
            image_ids = open(os.path.join(saveBasePath, i ),encoding='utf-8').read().strip().split()
            print(image_ids)
            list_file = open('mask_%s'%i, 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('mask_or_not/JPEGImages/%s.jpg' % image_id)
                convert_annotation(image_id, list_file)
                list_file.write('\n')
            list_file.close()
    with open('mask_train.txt','r') as f:
        info = f.readlines()[0]
        print(info)
        img = info.split()[0]
    image = Image.open(img)
    image.show()
