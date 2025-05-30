import os
import torch
import torch.nn as nn
import numpy as np
from medpy import metric
from scipy.ndimage import zoom
import seaborn as sns
from PIL import Image 
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('agg')

from segmentation_mask_overlay import overlay_masks
import matplotlib.colors as mcolors

import SimpleITK as sitk
import pandas as pd


from thop import profile
from thop import clever_format

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
    
def one_hot_encoder(input_tensor,dataset,n_classes = None):
    tensor_list = []
    if dataset == 'MMWHS':  
        dict = [0,205,420,500,550,600,820,850]
        for i in dict:
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    else:
        for i in range(n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()    

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        #print(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.assd(pred, gt)
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0


def calculate_metric_percase_dice(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        #hd95 = metric.binary.hd95(pred, gt)
        #jaccard = metric.binary.jc(pred, gt)
        #asd = metric.binary.assd(pred, gt)
        return dice, 0, 0,0
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0




def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0



def test_single_volume_dice(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, class_names=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if class_names==None:
        mask_labels = np.arange(1,classes)
    else:
        mask_labels = class_names
    cmaps = mcolors.CSS4_COLORS
    my_colors=['red','darkorange','yellow','forestgreen','blue','purple','magenta','cyan','deeppink', 'chocolate', 'olive','deepskyblue','darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                P = net(input)
                # outputs = 0.0
                # for idx in range(len(P)):
                #     outputs += P[idx]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                
                lbl = label[ind, :, :]
                masks = []
                for i in range(1, classes):
                    masks.append(lbl==i)
                preds_o = []
                for i in range(1, classes):
                    preds_o.append(pred==i)
                if test_save_path is not None:
                    fig_gt = overlay_masks(image[ind, :, :], masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                    fig_pred = overlay_masks(image[ind, :, :], preds_o, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                    # Do with that image whatever you want to do.
                    fig_gt.savefig(test_save_path + '/'+case + '_' +str(ind)+'_gt.png', bbox_inches="tight", dpi=300)
                    fig_pred.savefig(test_save_path + '/'+case + '_' +str(ind)+'_pred.png', bbox_inches="tight", dpi=300)
                    plt.close('all')

    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = 0.0
            for idx in range(len(P)):
                outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_dice(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list






def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, class_names=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if class_names==None:
        mask_labels = np.arange(1,classes)
    else:
        mask_labels = class_names
    cmaps = mcolors.CSS4_COLORS
    my_colors=['red','darkorange','yellow','forestgreen','blue','purple','magenta','cyan','deeppink', 'chocolate', 'olive','deepskyblue','darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                # outputs = 0.0
                # for idx in range(len(P)):
                #     outputs += P[idx]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                
                lbl = label[ind, :, :]
                masks = []
                for i in range(1, classes):
                    masks.append(lbl==i)
                preds_o = []
                for i in range(1, classes):
                    preds_o.append(pred==i)
                if test_save_path is not None:
                    fig_gt = overlay_masks(image[ind, :, :], masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                    fig_pred = overlay_masks(image[ind, :, :], preds_o, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                    # Do with that image whatever you want to do.
                    fig_gt.savefig(test_save_path + '/'+case + '_' +str(ind)+'_gt.png', bbox_inches="tight", dpi=300)
                    fig_pred.savefig(test_save_path + '/'+case + '_' +str(ind)+'_pred.png', bbox_inches="tight", dpi=300)
                    plt.close('all')

    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = 0.0
            for idx in range(len(P)):
                outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_dice(prediction == i, label == i))

    if test_save_path is not None:
        print('here')
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
def test_single_volume2(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, class_names=None, device='cpu'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # 将 image 与 label 从 tensor 转为 numpy，并 squeeze 掉 batch 维度
    image = image.squeeze(0).cpu().detach().numpy()   # 原期望形状: (C, H, W) 或 (H, W, C)
    label = label.squeeze(0).cpu().detach().numpy()     # (H, W)

    # 检查 image 的维度，确保最终是 (3, H, W)
    if image.ndim == 3:
        # 如果 image 是 HWC 格式 (H, W, 3)，转为 CHW
        if image.shape[-1] == 3 and image.shape[0] != 3:
            image = np.transpose(image, (2, 0, 1))
        # 如果是单通道 (1, H, W)，复制通道至 3 通道
        elif image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
    else:
        raise ValueError("Unexpected image shape, expected 3 dimensions but got: {}".format(image.shape))

    # 设置 mask 标签
    if class_names is None:
        mask_labels = np.arange(1, classes)
    else:
        mask_labels = class_names

    # 设置颜色映射
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    cmaps = mcolors.CSS4_COLORS
    my_colors = ['red', 'darkorange', 'yellow', 'forestgreen', 'blue', 'purple', 'magenta', 'cyan', 'deeppink']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}

    # 预测阶段：转换为 tensor，注意 input shape 应为 (1, 3, H, W)
    input_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)  # (1, 3, H, W)
    net.eval()
    with torch.no_grad():
        outputs = net(input_tensor)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()  # (H, W)

    # 计算 Dice 分数
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_dice(prediction == i, label == i))

    # 保存图像及预测结果
    if test_save_path is not None:
        # 构造 masks 和 preds 数组：将列表堆叠成 numpy 数组
        masks_list = [(label == i) for i in range(1, classes)]
        preds_list = [(prediction == i) for i in range(1, classes)]
        masks_arr = np.stack(masks_list, axis=-1)  # shape: (H, W, num_masks)
        preds_arr = np.stack(preds_list, axis=-1)    # shape: (H, W, num_masks)

        # 用第一通道用于可视化（如果 image 是 RGB，则取第一个通道显示灰度效果）
        vis_image = image[0] if image.shape[0] >= 1 else image

        fig_gt = overlay_masks(vis_image, masks_arr, labels=mask_labels, colors=cmap)
        fig_pred = overlay_masks(vis_image, preds_arr, labels=mask_labels, colors=cmap)

        import matplotlib.pyplot as plt

        # 替换掉原来的 fig_gt.savefig 和 fig_pred.savefig 调用
        gt_save_path = os.path.join(test_save_path, f'{case}_gt.png')
        pred_save_path = os.path.join(test_save_path, f'{case}_pred.png')

        plt.imsave(gt_save_path, fig_gt)  # fig_gt 是 numpy 数组
        plt.imsave(pred_save_path, fig_pred)

        plt.close('all')

        # 保存为 .nii.gz 格式（可选，仅保存第一通道用于显示）
        import SimpleITK as sitk
        img_itk = sitk.GetImageFromArray(image[0].astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, os.path.join(test_save_path, f"{case}_pred.nii.gz"))
        sitk.WriteImage(img_itk, os.path.join(test_save_path, f"{case}_img.nii.gz"))
        sitk.WriteImage(lab_itk, os.path.join(test_save_path, f"{case}_gt.nii.gz"))

    return metric_list


def val_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():                
                P = net(input)
                outputs = 0.0
                for idx in range(len(P)):
                   outputs += P[idx]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = 0.0
            for idx in range(len(P)):
               outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list

