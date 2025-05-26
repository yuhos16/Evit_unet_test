import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_ham10000 import ham_dataset
from utils.utils import test_single_volume_dice, test_single_volume2
from trainer import trainer_synapse

# from unet import Eff_Unet
from unet.eff_unet_mod import Eff_Unet

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (1, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='data/Synapse', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, default='exp2',help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=120, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--epoch', type=int, default=1234, help='test epoch')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--device', default='cpu',  help='device for test')

args = parser.parse_args()
if args.dataset == "ham":
    args.volume_path = os.path.join(args.volume_path, "npz_data")


# def inference(args, model, test_save_path=None):
#     db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()
#     metric_list = 0.0
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         h, w = sampled_batch["image"].size()[2:]
#         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
#         metric_i = test_single_volume2(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
#                                       test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
#         metric_list += np.array(metric_i)
#         # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
#         # print(f'idx:{i_batch}, case:{case_name}, mean dice:{np.mean(metric_i, axis=0)[0]}')
#         logging.info('idx %d case %s mean_dice %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0]))

#     metric_list = metric_list / len(db_test)
#     for i in range(1, args.num_classes):
#         # logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
#         logging.info('Mean class %d mean_dice %f' % (i, metric_list[i-1][0]))
#         print(f'Mean class:{i}, mean dice:{metric_list[i-1][0]}')

#     performance = np.mean(metric_list, axis=0)[0]
#     # mean_hd95 = np.mean(metric_list, axis=0)[1]
#     # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
#     logging.info('Testing performance in best val model: mean_dice : %f.' % (performance))
#     # print(f'Testing performance (mean_dice): {performance}')
#     return "Testing Finished!"
def inference(args, model, test_save_path=None):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    acc_list = 0.0  # [修改] 用于累计每个样本的像素级准确率

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        # 如果 image 的 shape 为 [B, H, W, C]，则需要转换为 [B, C, H, W]
        image = image.permute(0, 3, 1, 2).contiguous()  # [修改] 重排列维度

        metric_i = test_single_volume2(image, label, model, classes=args.num_classes, 
                                       patch_size=[args.img_size, args.img_size],
                                       test_save_path=test_save_path, case=case_name, 
                                       z_spacing=args.z_spacing, device=args.device)
        metric_list += np.array(metric_i)
        
        
        # with torch.no_grad():
        #     # === 图像预处理 ===
        #     # 将 image 从 tensor 转为 numpy（squeeze掉 batch 维度）
        #     image_np = image.squeeze(0).cpu().detach().numpy()  # (H, W, C) 或 (C, H, W)
        #     if image_np.ndim == 3 and image_np.shape[-1] == 3 and image_np.shape[0] != 3:
        #         image_np = np.transpose(image_np, (2, 0, 1))  # → [C, H, W]
        #     # image_np = image_np.astype(np.float32) / 255.0
        #     input_tensor = torch.from_numpy(image_np).unsqueeze(0).float().cuda()  # (1, C, H, W)
            
        #     # === 模型预测 ===
        #     net.eval()
        #     outputs = net(input_tensor)  # 假设输出形状为 [1, num_classes, H, W]
        #     # print(outputs.shape, "---")
        #     # 对空间维度进行 softmax 后取 argmax 得到分割预测
        #     pred_map = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  # (H, W)
        #     prediction = pred_map.cpu().detach().numpy()
            
        #     # === 根据预测分割 map 统计每个类别像素数量，选择最多的一类作为图像级预测 ===
        #     # 注意：类别从 1 到 classes-1，背景为 0
        #     classes = args.num_classes
        #     class_counts = [np.sum(prediction == i) for i in range(1, classes)]
        #     cls_pred = np.argmax(class_counts) + 1  # +1 是因为列表索引从 0 开始

        #     # === 从 label 中提取真实类别 ===
        #     # label 原本为 segmentation mask，取非0的唯一值作为分类标签
        #     label_np = label.squeeze(0).cpu().detach().numpy()  # (H, W)
        #     unique_vals = np.unique(label_np)
        #     unique_vals = unique_vals[unique_vals != 0]  # 去除背景（0）
        #     if len(unique_vals) == 0:
        #         print(f"⚠️ Warning: No foreground label found for case {case}")
        #         continue
        #     cls_gt = int(unique_vals[0])  # 假设只有一个前景类别

        #     # === 计算图像级分类准确率 ===
        #     # 如果预测类别与真实类别一致，则 acc 为 1，否则为 0
        #     print(cls_pred ,cls_gt)
        #     acc = 1 if cls_pred == cls_gt else 0
        #     acc_list += acc
        with torch.no_grad():
            # === 图像预处理 ===
            # 将 image 从 tensor 转为 numpy（squeeze掉 batch 维度）
            image_np = image.squeeze(0).cpu().detach().numpy()  # (H, W, C) 或 (C, H, W)
            if image_np.ndim == 3 and image_np.shape[-1] == 3 and image_np.shape[0] != 3:
                image_np = np.transpose(image_np, (2, 0, 1))  # → [C, H, W]
            # image_np = image_np.astype(np.float32) / 255.0
            input_tensor = torch.from_numpy(image_np).unsqueeze(0).float().to(device)  # (1, C, H, W)
            
            # === 模型预测 ===
            net.eval()
            outputs = net(input_tensor)  # 输出形状为 [1, num_classes, H, W]
            
            # 通过 softmax 得到概率图，然后对空间维度进行平均
            prob_map = torch.softmax(outputs, dim=1)       # (1, num_classes, H, W)
            avg_probs = prob_map.mean(dim=(2, 3))            # (1, num_classes)
            
            # 排除背景类别（假设背景为 0），只考虑前景类别（1 ~ num_classes-1）
            avg_probs_fg = avg_probs[0, 1:]                  
            cls_pred = torch.argmax(avg_probs_fg).item() + 1   # +1 是因为去掉了背景类别

            # === 从 label 中提取真实类别 ===
            # label 原本为 segmentation mask，取非0的唯一值作为分类标签
            label_np = label.squeeze(0).cpu().detach().numpy()  # (H, W)
            unique_vals = np.unique(label_np)
            unique_vals = unique_vals[unique_vals != 0]  # 去除背景（0）
            if len(unique_vals) == 0:
                print(f"⚠️ Warning: No foreground label found for case {label_np}")
                continue
            cls_gt = int(unique_vals[0])  # 假设只有一个前景类别

            # === 计算图像级分类准确率 ===
            # 如果预测类别与真实类别一致，则 acc 为 1，否则为 0
            # print(cls_pred, cls_gt)
            acc = 1 if cls_pred == cls_gt else 0
            acc_list += acc


        logging.info('idx %d case %s mean_dice %f acc %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], acc))

    metric_list = metric_list / len(db_test)
    avg_acc = acc_list / len(db_test)  # [修改] 计算所有样本的平均准确率

    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f' % (i, metric_list[i-1][0]))
        print(f'Mean class:{i}, mean dice:{metric_list[i-1][0]}')

    performance = np.mean(metric_list, axis=0)[0]
    logging.info('Testing performance in best val model: mean_dice : %f, mean_acc: %f.' % (performance, avg_acc))
    print(f'Testing performance (mean_dice): {performance}, (mean_acc): {avg_acc}')
    
    return "Testing Finished!"


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (1, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'ham': {
            'Dataset': ham_dataset,
            'volume_path': args.volume_path,
            'list_dir': r'/root/autodl-tmp/dataset/HAM10000/pd2/text',
            'num_classes': 8,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from unet import Eff_Unet
    net = Eff_Unet(
        layers=[5, 5, 15, 10],
        embed_dims=[40, 80, 192, 384],
        downsamples=[True, True, True, True],
        vit_num=6,
        drop_path_rate=0.1,
        num_classes=8,
        fork_feat=True)
    
    net = net.to(device)
        
    # for epoch in reversed(range(81,150)):
    for epoch in range(300, 3400, 100):
        # if (epoch+1)%2!=0:
        snapshot = os.path.join(args.output_dir, f'epoch_{epoch}.pth')
        msg = net.load_state_dict(torch.load(snapshot), strict=False)
        snapshot_name = snapshot.split('/')[-1]
        # log_folder = f'./test_log/test_log_{args.output_dir}'
        case_name = args.output_dir.split('/')[-1]
        log_folder = f'test_result/best_epoch_{case_name}'
        os.makedirs(log_folder, exist_ok=True)
        logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(snapshot_name)

        # if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
        # else:
        # test_save_path = None
        inference(args, net, test_save_path)

