import torch.nn
from torch.utils.data import DataLoader
import torch.optim as optim
from fc.use import weights_init
from ssd_model import SSD300
from default_box import *
from train_and_val import *
from fc.multiboxloss import MultiboxLoss
from torch.backends import cudnn
from dataloader import SSDDataset, ssd_dataset_collate
import time


if __name__ == '__main__':
    time0 = time.time()
    Cuda = True
    cudnn.benchmark = True
    # last time weight
    weight_path = ''
    input_shape = [300, 300]
    pre_trained = False
    anchors_size = [30, 60, 111, 162, 213, 264, 315]

    # data_set
    train_annotation_path = 'mask_train.txt'
    val_annotation_path = 'mask_val.txt'
    # class
    class_name = ['face', 'face_mask']
    class_num = 3

    # get bot left and top right coordinates
    default_boxes = get_default_boxes()
    model = SSD300(class_num)

    # weight initialization
    if not pre_trained:
        weights_init(model)

    # load last weights
    if weight_path != '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weight_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    criterion = MultiboxLoss(class_num, neg_pos_ratio=3.0)

    # read data
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    batch_size = 8
    lr = 1e-4
    start_epoch = 0
    end_epoch = 50

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    train_dataset = SSDDataset(train_lines, input_shape, default_boxes, batch_size, class_num)
    val_dataset = SSDDataset(val_lines, input_shape, default_boxes, batch_size, class_num)
    train_data = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                     drop_last=True, collate_fn=ssd_dataset_collate)
    val_data = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=ssd_dataset_collate)


    for epoch in range(start_epoch, end_epoch):
        train(model,train_data,criterion,optimizer,epoch_step, epoch, end_epoch)
        validate(model, val_data, criterion,optimizer,epoch_step_val,epoch, end_epoch)
        lr_scheduler.step()
