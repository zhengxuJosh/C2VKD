import warnings
warnings.filterwarnings('ignore')
import argparse
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from torch.utils.tensorboard import SummaryWriter

from models.deeplabv3plus.deeplabv3plus import DeepLabv3Plus
from utils import losses, ramps
from utils.kd_losses import get_dkd_loss
from dataset.voc.VOC_dataset import VOCDataset
import time,datetime

from utils.mIOU_metrics import mIOUMetrics
import torchvision.models as models

# from models.pvt.FPN import FPN
from models.pvt_v2.FPN import FPN
from models.deeplabv3plus.att import att

def Prorotype(feat,target):
    size_f = (feat.shape[2], feat.shape[3])
    tar_feat = nn.Upsample(size_f,mode='nearest')(target.unsqueeze(1).float()).expand(feat.size())
    center_feat_2 = feat.clone()
    for i in range(19):
        mask = (tar_feat == i).float()
        center_feat = (1 - mask) * center_feat_2 + mask * ((mask * feat).sum(-1).sum(-1) / (mask.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
    return(center_feat)

def channel(feat):
    n,c,w,h = feat.shape
    feat = feat.reshape((n,c,-1))
    feat = feat.softmax(dim=-1)
    return feat

def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_current_consistency_weight(epoch,max_epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.05 * ramps.sigmoid_rampup(epoch, max_epoch)
    #return ramps.sigmoid_rampup(epoch, max_epoch)
def get_linear_rampup(epoch, max_epoch):
    return 0.1 * ramps.linear_rampup(epoch, max_epoch)

def main():
    
    parser = argparse.ArgumentParser(description='pytorch implemention')
    parser.add_argument('--batch-size', type=int, default=6, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--iterations', type=int, default=3670, metavar='N',#3670
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=6e-5, metavar='LR',
                        help='learning rate (default: 0.01)')
    # parser.add_argument('--lr-pvt', type=float, default=0.0005, metavar='LRPVT',
                        # help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--save_root', default = '',
                        help='Please add your model save directory') 
    parser.add_argument('--exp_name', default = '',
                        help='')
    
    parser.add_argument(
        "--sup_set", type=str, default="train", help="supervised training set")

    parser.add_argument('--cutmix', default =False, help='cutmix')

    parser.add_argument('--VOCdevkit_path', type=str, default="/hpc/users/CONNECT/xuzheng/ECCV-TCC/data/VOCdevkit/", help='')

    parser.add_argument('--froze_cnn', type=bool, default=True, help='if froze_resnet')

    # hyper parameters
    parser.add_argument('--dkd_alpha', type=float, default =0.5, help='dkd_alpha')
    parser.add_argument('--dkd_beta', type=float,default =0.5, help='dkd_beta')
    parser.add_argument('--dkd_temperature', type=float, default =1.0, help='dkd_temperature')
    parser.add_argument('--gb_fweight', type=float,default =0.1, help='gb_fweight')
    parser.add_argument('--lc_fweight', type=float,default =0.1, help='lc_fweight')
    parser.add_argument('--graph_loss_weight', type=float,default =0.1, help='graph_loss_weight')
    
    parser.add_argument('--model1_load_path', type=str,default=None, help='')
    parser.add_argument('--model2_load_path', type=str,default=None, help='')

    args = parser.parse_args()
    
    save_path = "{}{}".format(args.save_root,args.exp_name)
    cur_time=str(datetime.datetime.now().strftime("%y%m%d-%H:%M:%S"))
    save_img_path = "{}{}/images/{}/".format(args.save_root,args.exp_name,cur_time)
    writer = SummaryWriter(log_dir=save_path)
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
        os.makedirs(save_img_path)

    torch.cuda.set_device(args.local_rank)
    with torch.cuda.device(args.local_rank):
        dist.init_process_group(backend='nccl',init_method='env://')
        if dist.get_rank() == 0:
            print(args)
            print('init cnn lr: {}, batch size: {}, gpus:{}'.format(args.lr, args.batch_size, dist.get_world_size()))

        cutmix = False

        img_mean=[0.485, 0.456, 0.406]
        img_std=[0.229, 0.224, 0.225]

        dataset_path = args.VOCdevkit_path
        label_dataset = VOCDataset(f'{dataset_path}/VOC2012/ImageSets/Segmentation/',split=args.sup_set, base_size=520, crop_size=512,norm_mean=img_mean, norm_std=img_std)        
        val_dataset = VOCDataset(f'{dataset_path}/VOC2012/ImageSets/Segmentation/',split='val', is_train=False, base_size=520, crop_size=512,norm_mean=img_mean, norm_std=img_std)
        num_classes = len(VOCDataset.CLASSES_NAME)

        label_sampler = DistributedSampler(label_dataset, num_replicas=dist.get_world_size())
        
        label_loader = torch.utils.data.DataLoader(
            label_dataset,
            batch_size=args.batch_size,
            sampler=label_sampler,
            num_workers=4,
            worker_init_fn=lambda x: random.seed(time.time() + x),
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )
        # ResNet 152
        backbone = models.resnet101()
        # ckpt = torch.load('/hpc/users/CONNECT/yunhaoluo/C2VKD/models/ptmodel/resnet101-63fe2227.pth', map_location='cpu')
        # backbone.load_state_dict(ckpt, strict=False)
        model1 = DeepLabv3Plus(backbone)
        model1 = model1.to(args.local_rank)
        model1.load_state_dict(torch.load('/hpc/users/CONNECT/yunhaoluo/C2VKD/CVPR23/1010_DLv3_res101_CE/best.pth', map_location=torch.device("cpu")),strict=False)
        model1.eval()
        # PVT tiny
        model2 = FPN(num_classes=21)
        
        # if args.local_rank == 0:
        #     print(model2)
        model2 = nn.SyncBatchNorm.convert_sync_batchnorm(model2)

        model2 = model2.to(args.local_rank)
        model2 = DistributedDataParallel(model2,device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False, find_unused_parameters=True)

        model0 = att(num_classes=num_classes, encoder_type="att", model_path=None, pretrained_imgnet=False,feature=False)
        model0.load_state_dict(torch.load('/pretrain_model0.pth', map_location=torch.device("cpu")),strict=False)
        model0.eval()

        model0 = model0.to(args.local_rank)

        optimizer_vit = optim.AdamW(model2.parameters(), lr=args.lr, weight_decay=0.01)#1e-4

        epoch, num_classes = 0, 21

        criterion_sup = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

        kl_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=True)
        mse_loss = nn.MSELoss()
        best_performance1 = 0.0
        best_performance2 = 0.0

        sup_loader = iter(label_loader)
        s_length = len(sup_loader)

        avg_loss_vit, avg_loss_feat_vit, avg_loss_dkd_vit, avg_loss_graph = 0.,0.,0.,0.
        avg_loss_mhsa = 0.
        print(f'Dataset s_length:{len(label_dataset)};')
        max_epoch = args.iterations / s_length
        for it in range(1, args.iterations + 1):
            if it % s_length == 0:
                label_loader.sampler.set_epoch(epoch)
                sup_loader = iter(label_loader)

            s_img, s_gt = sup_loader.__next__()
            s_img, s_gt = s_img.to(args.local_rank), s_gt.to(args.local_rank)

            pred_vit, lf_vit = model2(s_img)
            pred_cnn, lf_cnn = model1(s_img)

            # "language compatible feature loss"
            l_vit = model0(lf_vit)
            l_cnn = model0(lf_cnn)
            l_cnn = nn.functional.log_softmax(l_cnn)
            l_vit = nn.functional.log_softmax(l_vit)
            language = 0.5 * kl_loss(l_vit,l_cnn)

            # "Global feature loss"
            cnn_logsfmax = nn.functional.log_softmax(lf_cnn, dim=1).permute(0,2,3,1)
            vit_logsfmax = nn.functional.log_softmax(lf_vit, dim=1).permute(0,2,3,1)

            lc_feat_cnn_logsfmax = nn.functional.log_softmax(lf_cnn, dim=1).permute(0,2,3,1)
            lc_feat_vit_logsfmax = nn.functional.log_softmax(lf_vit, dim=1).permute(0,2,3,1)

            gb_loss = args.gb_fweight * kl_loss(vit_logsfmax, cnn_logsfmax.detach())

            # "Patch-wise feature loss"
            # =========
            feat_c = lf_cnn.shape[1]
            # (B C 32 32) -> (B 32 32 C) -> (B*32*32, C)
            lc_feat_cnn_flat = lf_cnn.permute(0,2,3,1).reshape(-1, feat_c)
            lc_feat_cnn_flat = lc_feat_cnn_flat / lc_feat_cnn_flat.norm(dim=1, keepdim=True)
            lc_feat_cnn_flat =  lc_feat_cnn_flat - lc_feat_cnn_flat.mean(dim=-1, keepdim=True)
            
            lc_feat_vit_flat = lf_vit.permute(0,2,3,1).reshape(-1, feat_c)
            lc_feat_vit_flat = lc_feat_vit_flat / lc_feat_vit_flat.norm(dim=1, keepdim=True)
            lc_feat_vit_flat = lc_feat_vit_flat - lc_feat_vit_flat.mean(dim=-1, keepdim=True)

            # (B*32*32, C) @ (C, B*32*32) = (B*32*32, B*32*32)
            cnn_selfcovar = lc_feat_cnn_flat @ lc_feat_cnn_flat.t() / (feat_c - 1)
            vit_selfcovar = lc_feat_vit_flat @ lc_feat_vit_flat.t() / (feat_c - 1)

            loss_graph = args.graph_loss_weight * mse_loss(cnn_selfcovar, vit_selfcovar)
            
            # "CE Loss"
            ce = criterion_sup(pred_vit, s_gt)

            # " KL Loss"
            #kd = kl_loss(pred_vit, pred_cnn)

            "dkd loss"
            # param: "logits_student, logits_teacher, target, alpha, beta, temperature"
            dkd = get_dkd_loss(pred_vit, pred_cnn, s_gt, alpha=args.dkd_alpha, beta=args.dkd_beta, temperature=args.dkd_temperature)
            
            #loss_vit = dkd + loss_graph
            loss_vit  = dkd + gb_loss + loss_graph + language
            writer.add_scalar('CE_Loss',ce,epoch)
            writer.add_scalar('DKD_Loss',dkd,epoch)
            writer.add_scalar('GB_Loss',gb_loss,epoch)
            writer.add_scalar('Graph_Loss',loss_graph,epoch)
            writer.add_scalar('Total_Loss',loss_vit,epoch)
            writer.add_scalar('language',language,epoch)

            if it % s_length == 0:
                if dist.get_rank() == 0:
                    print(f'it:{it};loss_CE:{ce:.4f}')
                    print(f'it:{it};DKD_Loss:{dkd:.4f}')
                    print(f'it:{it};GB_Loss:{gb_loss:.4f}')
                    print(f'it:{it};Graph_Loss:{loss_graph:.4f}')

            optimizer_vit.zero_grad()
            loss_vit.backward()
            optimizer_vit.step()

            # lr decay here!
            #base_lr = args.lr
            #lr_ = base_lr * (1.0 - it/args.iterations) ** 0.9
            #for param_group in optimizer_vit.param_groups:
            #    param_group['lr'] = lr_ 
            
            # swin lr
            base_lr = args.lr
            if it <= 1500:
                lr_ = base_lr * (it / 1500)
                for param_group in optimizer_vit.param_groups:
                    param_group['lr'] = lr_ 
            else:
                lr_ = adjust_learning_rate_poly(optimizer_vit,it - 1500,args.iterations,args.lr,1)
            #print('lr:',lr_)   
                
            if it == 1 and dist.get_rank() == 0:
                print('s_img: ', s_img.shape)
                print('s_gt: ', s_gt.shape)
                print('s_len', s_length)

            if it % s_length == 0 or it == 1:
                #print(f'[Validation it: {it}] lr: {lr_}')
                miou_metrics_1 = mIOUMetrics(num_classes,255,args.local_rank)
                miou_metrics_2 = mIOUMetrics(num_classes,255,args.local_rank)
                if it != 1:
                    epoch += 1
                if dist.get_rank() == 0:
                    model2.eval()
                    with torch.no_grad():
                        val_mIOU = 0.0
                        val_mIOU_total = 0.0
                        for i, (image,label) in enumerate(val_loader):
                                image, label = image.to(args.local_rank), label.to(args.local_rank)
                                pred, _ = model2(image)
                                miou_metrics_2.update(pred,label)
                        #val_mIOU_final = val_mIOU_total / len(val_dataset)
                        val_mIOU_final = miou_metrics_2.get_mIOU()
                        miou_metrics_2.reset()
                        writer.add_scalar('model2 val',val_mIOU_final,epoch)
                        if val_mIOU_final > best_performance2:
                            best_performance2 = val_mIOU_final
                            torch.save(model2.module.state_dict(),save_path+"/best.pth")
                        print('epoch:',epoch,'model val_mIOU:',val_mIOU_final.item(), 'best:', best_performance2.item())
                    model2.train()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    print('file name: ', __file__)
    my_seed = random.randint(1,1000000)
    print('seed:', my_seed)
    setup_seed(my_seed)
    # setup_seed(133)
    main()
