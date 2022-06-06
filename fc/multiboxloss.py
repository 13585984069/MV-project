import torch
import torch.nn as nn


class MultiboxLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0, background_label_id=0, negatives_for_hard=100.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.background_label_id = background_label_id
        self.negatives_for_hard = torch.FloatTensor([negatives_for_hard])[0]

    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = torch.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = torch.where(abs_loss < 1.0, sq_loss, abs_loss - 0.5)
        return torch.sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, min=1e-7)
        softmax_loss = -torch.sum(y_true * torch.log(y_pred),axis=-1)
        return softmax_loss

    def forward(self, y_true, y_pred):
        num_boxes = y_true.size()[1]
        y_pred = torch.cat([y_pred[0], nn.Softmax(-1)(y_pred[1])], dim=-1)
        # classification loss
        conf_loss = self._softmax_loss(y_true[:, :, 4:-1], y_pred[:, :, 4:])
        # location loss
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4], y_pred[:, :, :4])
        # positive label loss
        pos_loc_loss = torch.sum(loc_loss * y_true[:, :, -1],axis=1)
        pos_conf_loss = torch.sum(conf_loss * y_true[:, :, -1],axis=1)

        # num of positive label
        num_pos = torch.sum(y_true[:, :, -1], axis=-1)
        # num of negative label
        num_neg = torch.min(self.neg_pos_ratio * num_pos, num_boxes - num_pos)
        pos_num_neg_mask = num_neg > 0
        has_min = torch.sum(pos_num_neg_mask)

        # if no postive label, choose 100 default box as neg label
        num_neg_batch = torch.sum(num_neg) if has_min > 0 else self.negatives_for_hard

        confs_start = 4 + self.background_label_id + 1
        confs_end = confs_start + self.num_classes - 1

        max_confs = torch.sum(y_pred[:, :, confs_start:confs_end], dim=2)
        max_confs = (max_confs * (1 - y_true[:, :, -1])).view([-1])
        _, indices = torch.topk(max_confs, k=int(num_neg_batch.cpu().numpy().tolist()))
        neg_conf_loss = torch.gather(conf_loss.view([-1]), 0, indices)

        # normalization
        num_pos = torch.where(num_pos != 0, num_pos, torch.ones_like(num_pos))
        total_loss = torch.sum(pos_conf_loss) + torch.sum(neg_conf_loss) + torch.sum(self.alpha * pos_loc_loss)
        total_loss = total_loss / torch.sum(num_pos)
        return total_loss
