import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import DiceLoss
from torchvision import transforms
import torchvision
# from utils import test_single_volume
import torch.nn.functional as F
from modules import ClassificationHead

class KDloss(nn.Module):

    def __init__(self,lambda_x):
        super(KDloss,self).__init__()
        self.lambda_x = lambda_x

    def inter_fd(self,f_s, f_t):
        s_C, t_C, s_H, t_H = f_s.shape[1], f_t.shape[1], f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        
        idx_s = random.sample(range(s_C),min(s_C,t_C))
        idx_t = random.sample(range(t_C),min(s_C,t_C))

        #inter_fd_loss = F.mse_loss(f_s[:, 0:min(s_C,t_C), :, :], f_t[:, 0:min(s_C,t_C), :, :].detach())

        inter_fd_loss = F.mse_loss(f_s[:, idx_s, :, :], f_t[:, idx_t, :, :].detach())
        return inter_fd_loss 
    
    def intra_fd(self,f_s):
        sorted_s, indices_s = torch.sort(F.normalize(f_s, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
        f_s = torch.index_select(f_s, 1, indices_s)
        intra_fd_loss = F.mse_loss(f_s[:, 0:f_s.shape[1]//2, :, :], f_s[:, f_s.shape[1]//2: f_s.shape[1], :, :])
        return intra_fd_loss
    
    def forward(self,feature,feature_decoder,final_up):
        # f1 = feature[0][-1] # 
        # f2 = feature[1][-1]
        # f3 = feature[2][-1]
        # f4 = feature[3][-1] # lower feature 

        f1_0 = feature[0] # 
        f2_0 = feature[1]
        f3_0 = feature[2]
        f4_0 = feature[3] # lower feature 

        # f1_d = feature_decoder[0][-1] # 14 x 14
        # f2_d = feature_decoder[1][-1] # 28 x 28
        # f3_d = feature_decoder[2][-1] # 56 x 56

        f1_d_0 = feature_decoder[0] # 14 x 14
        f2_d_0 = feature_decoder[1] # 28 x 28
        f3_d_0 = feature_decoder[2] # 56 x 56

        #print(f3_d.shape)

        final_layer = final_up
        #print(final_layer.shape)


        # loss =  (self.intra_fd(f1)+self.intra_fd(f2)+self.intra_fd(f3)+self.intra_fd(f4))/4
        loss = (self.intra_fd(f1_0)+self.intra_fd(f2_0)+self.intra_fd(f3_0)+self.intra_fd(f4_0))/4
        loss += (self.intra_fd(f1_d_0)+self.intra_fd(f2_d_0)+self.intra_fd(f3_d_0))/3
        # loss += (self.intra_fd(f1_d)+self.intra_fd(f2_d)+self.intra_fd(f3_d))/3


        
        loss += (self.inter_fd(f1_d_0,final_layer)+self.inter_fd(f2_d_0,final_layer)+self.inter_fd(f3_d_0,final_layer)
                   +self.inter_fd(f1_0,final_layer)+self.inter_fd(f2_0,final_layer)+self.inter_fd(f3_0,final_layer)+self.inter_fd(f4_0,final_layer))/7

        
        
        loss = loss * self.lambda_x
        return loss 
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule



def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator,RandomGenerator_DINO,RandomGenerator_DINO_Deform
    from torchvision.transforms import functional as VF

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]),
                                transform_dino=transforms.Compose(
                                   [RandomGenerator_DINO(output_size=[args.img_size, args.img_size])])) #,alpha = args.alpha,sigma=args.sigma
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    #teacher_model.eval()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # kd_loss = KDloss(lambda_x=args.lambda_x)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    momentum_schedule = cosine_scheduler(0.996, 1,
                                               max_iterations, len(trainloader))

    

    for epoch_num in iterator:
        # for i_batch, (sampled_batch,dino_batch) in enumerate(trainloader):
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

          
      

            # outputs, kd_encorder,kd_decorder, final_up = model(image_batch)
            outputs = model(image_batch)
            
          

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss_kd = kd_loss(kd_encorder,kd_decorder,final_up)
            
            loss = 0.4 * loss_ce + 0.6 * loss_dice # + args.dino_weight*loss_dino
            # loss = 0.4 * loss_ce + 0.6 * loss_dice + loss_kd # + args.dino_weight*loss_dino
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/dice_loss', loss_dice, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_dino', loss_dino,iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            # if iter_num % 20 == 0:
            #     image = image_batch[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

            # if iter_num % 20 == 0:
            #     # 获取图像数据的第一个样本
            #     image = image_batch[1, 0:1, :, :]
                
            #     # 将图像数据归一化到0-1之间
            #     image = (image - image.min()) / (image.max() - image.min())
                
            #     # 保存原始图像
            #     torchvision.utils.save_image(image, os.path.join(args.output_dir, f'train_Image_iter_{iter_num}.png'))
                
            #     # 计算预测结果
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                
            #     # 保存预测结果
            #     torchvision.utils.save_image(outputs[1, ...].float() * 50, os.path.join(args.output_dir, f'train_Prediction_iter_{iter_num}.png'))
                
            #     # 保存标签 (Ground Truth)
            #     labs = label_batch[1, ...].unsqueeze(0).float() * 50
            #     torchvision.utils.save_image(labs, os.path.join(args.output_dir, f'train_GroundTruth_iter_{iter_num}.png'))




        save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if epoch_num > 60:   
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_ham(args, model, snapshot_path):
    from datasets.dataset_ham10000 import ham_dataset, RandomGenerator
    from torchvision.transforms import functional as VF

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = ham_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]),
                                ) #,alpha = args.alpha,sigma=args.sigma
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    #teacher_model.eval()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # kd_loss = KDloss(lambda_x=args.lambda_x)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    momentum_schedule = cosine_scheduler(0.996, 1,
                                               max_iterations, len(trainloader))

    

    for epoch_num in iterator:
        # for i_batch, (sampled_batch,dino_batch) in enumerate(trainloader):
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

          
      

            # outputs, kd_encorder,kd_decorder, final_up = model(image_batch)
            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss_kd = kd_loss(kd_encorder,kd_decorder,final_up)
            
            loss = 0.4 * loss_ce + 0.6 * loss_dice # + args.dino_weight*loss_dino
            # loss = 0.4 * loss_ce + 0.6 * loss_dice + loss_kd # + args.dino_weight*loss_dino
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            # writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/dice_loss', loss_dice, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalars('info/loss', {
                'loss': loss,
                'loss_dice': loss_dice,
                'loss_ce': loss_ce
            }, iter_num)

            # writer.add_scalar('info/loss_dino', loss_dino,iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))


        save_interval = 100  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if epoch_num > 200:   
            if epoch_num % save_interval == 0:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

# def trainer_ham(args, model, snapshot_path):
#     # —— 1. 日志 & 数据集 setup —— #
#     os.makedirs(snapshot_path, exist_ok=True)
#     logging.basicConfig(
#         filename=os.path.join(snapshot_path, "log.txt"),
#         level=logging.INFO,
#         format='[%(asctime)s] %(message)s',
#         datefmt='%H:%M:%S'
#     )
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info("Arguments: %s", args)

#     # —— 2. 构造分类头 & 损失 —— #
#     # Eff_Unet 最后一层特征图通道数，与你初始化 Eff_Unet 时的 embed_dims[0] 一致
#     in_ch = model.module.patch_embed[-1].num_features if hasattr(model, 'module') else model.patch_embed[-1].num_features
#     cls_head = ClassificationHead(in_channels=in_ch, num_classes=args.num_classes).cuda()
#     cls_criterion = CrossEntropyLoss()

#     # —— 3. 准备训练集 —— #
#     base_lr   = args.base_lr
#     batch_size = args.batch_size * args.n_gpu
#     db_train = ham_dataset(
#         base_dir=args.root_path,
#         list_dir=args.list_dir,
#         split="train",
#         transform=transforms.Compose([
#             RandomGenerator(output_size=[args.img_size, args.img_size])
#         ])
#     )
#     logging.info("Train set size: %d", len(db_train))

#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)

#     trainloader = DataLoader(
#         db_train,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True,
#         worker_init_fn=worker_init_fn
#     )

#     # —— 4. 多卡 & 训练模式 —— #
#     if args.n_gpu > 1:
#         model    = nn.DataParallel(model)
#         cls_head = nn.DataParallel(cls_head)
#     model.train()
#     cls_head.train()

#     # —— 5. 损失 & 优化器 —— #
#     seg_ce   = CrossEntropyLoss()
#     seg_dice = DiceLoss(args.num_classes)
#     optimizer = optim.AdamW(
#         list(model.parameters()) + list(cls_head.parameters()),
#         lr=base_lr, weight_decay=1e-3
#     )

#     writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
#     iter_num      = 0
#     max_epoch     = args.max_epochs
#     max_iterations = max_epoch * len(trainloader)
#     logging.info("%d iters/epoch, %d total iters", len(trainloader), max_iterations)

#     # —— 6. 训练循环 —— #
#     for epoch in range(max_epoch):
#         for sampled_batch in trainloader:
#             images = sampled_batch['image'].cuda()    # [B, C, H, W]
#             labels = sampled_batch['label'].cuda()    # [B, H, W]

#             # —— 6.1 forward —— #
#             seg_logits, feat_map = model(images)      # seg_logits: [B, K, H, W], feat_map: [B, C, H, W]
#             cls_logits = cls_head(feat_map)           # [B, num_classes]

#             # —— 6.2 分割损失 —— #
#             loss_ce   = seg_ce(seg_logits, labels.long())
#             loss_dice = seg_dice(seg_logits, labels, softmax=True)

#             # —— 6.3 分类标签 & 损失 —— #
#             flat      = labels.view(labels.size(0), -1)   # [B, H*W]
#             cls_label = torch.mode(flat, dim=1)[0]        # [B]
#             loss_cls  = cls_criterion(cls_logits, cls_label)

#             # —— 6.4 总损失 & backward —— #
#             loss = 0.4 * loss_ce + 0.6 * loss_dice + 0.2 * loss_cls
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # —— 6.5 lr & 日志 —— #
#             lr = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for pg in optimizer.param_groups:
#                 pg['lr'] = lr

#             iter_num += 1
#             writer.add_scalars('loss', {
#                 'total':   loss.item(),
#                 'seg_ce':  loss_ce.item(),
#                 'seg_dice':loss_dice.item(),
#                 'cls':     loss_cls.item()
#             }, iter_num)
#             writer.add_scalar('lr', lr, iter_num)
#             logging.info(
#                 "Iter %d: total %.4f, seg_ce %.4f, seg_dice %.4f, cls %.4f",
#                 iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_cls.item()
#             )

#         # —— 6.6 保存模型 —— #
#         if (epoch + 1) % 100 == 0 or epoch == max_epoch - 1:
#             ckpt = {
#                 'seg_model': model.state_dict(),
#                 'cls_head':  cls_head.state_dict()
#             }
#             path = os.path.join(snapshot_path, f'epoch_{epoch}.pth')
#             torch.save(ckpt, path)
#             logging.info("Saved checkpoint: %s", path)

#     writer.close()
#     return "Training Finished!"