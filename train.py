r""" training (validation) code """
import torch.optim as optim
import torch.nn as nn
import torch
import os

from model.SCDNet import SCDNet
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset
import torch.nn.functional as F
import os.path as osp

from util import dataset, transform, transform_tri, config
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from math import pi, cos, sqrt

class SurrogateLoss(nn.Module):
    
    def __init__(self, num_classes=15, feat_dim=2048, use_gpu=True, surrogate_momentum=0.99999, init='random', num_sections=8, max_degree=90, weight_a = 10, weight_r = 0.5, fold=0):
        super(SurrogateLoss, self).__init__()
        self.surrogate_momentum = surrogate_momentum
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.num_sections = num_sections
        self.max_degree = max_degree
        self.weight_a = weight_a
        self.weight_r = weight_r
        self.fold = fold

        self.init_surrogate(init)

        self.label2surr = {}
        # PASCAL
        if self.num_classes == 15:
            idx = 0
            for i in range(1, 21):
                if (i - 1) // 5  != self.fold:
                    self.label2surr[i] = idx
                    idx = idx + 1
        # COCO
        elif self.num_classes == 60:
            idx = 0
            for i in range(1, 81):
                if i % 4  != self.fold + 1:
                    self.label2surr[i] = idx
                    idx = idx + 1

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        # print(x.shape, )
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
        # The fold causes the lables to be discontinuous
        labels = labels + 1
        old_labels = deepcopy(labels)
        labels = [self.label2surr[label.item()] for label in labels]
            
        surrogate = self.surrogates[labels]
        kl = F.kl_div(F.log_softmax(x,dim=-1),F.softmax(surrogate,dim=-1),reduction='batchmean')
        self.update_surrogate(labels, x)
        loss = torch.clamp(kl, min=1e-5, max=1e+5).mean(dim=-1)
        return loss
    
    def init_surrogate(self, init='random'):
        if init == 'random':
            self.surrogates = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

        elif init == 'orthogonal':
            self.surrogates = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            nn.init.orthogonal_(self.surrogates)

        elif init == 'angle':
            surrogates_r = torch.randn([self.num_classes, self.feat_dim])

            max_radians = self.max_degree * pi / 180
            unit_radians = max_radians / self.num_sections
            section_dim = self.feat_dim // self.num_sections

            sections = []
            for i in range(1, self.num_sections + 1):
                theta = i * unit_radians
                sections.append(self.create_section(self.num_classes, section_dim, theta))
            surrogates_a = torch.cat(sections, dim=1)

            surrogates = self.weight_a * surrogates_a + self.weight_r * surrogates_r
            self.surrogates = nn.Parameter(surrogates.cuda())


    def create_section(self, section_num, section_dim, theta):
        section = torch.randn([section_num, section_dim])
        nn.init.orthogonal_(section)

        for i in range(1, section_num):
            z = section[i, :]

            a = cos(theta) / (1 + (i - 1) * cos(theta))
            b = sqrt(1 - (i + i * (i - 1) * cos(theta)) * a**2)

            section[i, :] = a * torch.sum(section[:i, :], dim=0) + b * z
        return section

    @torch.no_grad()
    def update_surrogate(self, labels, x):
        """
        Update surrogates.
        """

        # ema update
        self.surrogates[labels] = self.surrogates[labels] * self.surrogate_momentum + x * (1 - self.surrogate_momentum)

def train(epoch, model, dataloader, optimizer, surrogate_loss, training, nshot=1):
    r""" Train """

    torch.autograd.set_detect_anomaly(False)

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        batch = utils.to_cuda(batch)

        logit_mask, prototype_out, confusion_map, prototype_target, query_feat = model(batch['query_img'], 
                                                                                    batch['support_imgs'].squeeze(1), 
                                                                                    batch['support_masks'].squeeze(1), 
                                                                                    batch['class_id'], 
                                                                                    surrogate_loss.surrogates.detach(), 
                                                                                    training)
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute Seg loss & update model parameters
        
        loss = model.module.compute_objective(logit_mask, batch['query_mask'], confusion_map)
        
        if training:
            recon_loss = model.module.compute_reconstruction_loss(prototype_target, query_feat, batch['query_mask'])
            loss = loss + 0.1 * recon_loss + surrogate_loss(prototype_out, batch['class_id'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou

if __name__ == '__main__':
    
    args = parse_opts()
    # ddp backend initialization
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # Model initialization
    model = SCDNet(args.backbone, args.feature_extractor_path, False, batch_size=args.bsz, dataset=args.benchmark, nshot=args.nshot, fold=args.fold)
    
    num_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters (model): {num_params}")
    
    if args.reload_path:
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.reload_path, map_location='cpu').items()})
        print('reload success')
    device = torch.device("cuda", args.local_rank)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)
    num_classes = {'coco':60, 'pascal':15, 'fss':520}
    if args.backbone == 'resnet50':
        surrogate_loss = SurrogateLoss(feat_dim=2048, num_classes=num_classes[args.benchmark], init='angle', fold=args.fold).cuda()
    else:
        surrogate_loss = SurrogateLoss(feat_dim=4096, num_classes=num_classes[args.benchmark], init='angle', fold=args.fold).cuda()
        
    num_params_surrogate = sum(p.numel() for p in surrogate_loss.parameters())
    # print(f"Number of parameters (surrogate): {num_params_surrogate}")
    
    # Helper classes (for training) initialization
    optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr,
                            "momentum": 0.9, "weight_decay": args.lr/10, "nesterov": True}])
    
    
    Evaluator.initialize()
    if args.local_rank == 0:
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # Train
    train_transform = transform.Compose([
        # transform.RandScale([args.scale_min, args.scale_max]),
        # transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        # transform.RandomGaussianBlur(),
        # transform.RandomHorizontalFlip(),
        transform.Resize([args.train_h, args.train_w]),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std), 
        ])
    train_transform_tri = transform_tri.Compose([
        # transform_tri.RandScale([args.scale_min, args.scale_max]),
        # transform_tri.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        # transform_tri.RandomGaussianBlur(),
        # transform_tri.RandomHorizontalFlip(),
        transform_tri.Resize([args.train_h, args.train_w]),
        transform_tri.ToTensor(),
        transform_tri.Normalize(mean=mean, std=std), 
        ])
    
    base_data_root = osp.join(args.base_data_root, args.benchmark)

    if args.benchmark == 'pascal':
        data_root = osp.join(args.data_root, 'VOCdevkit2012/VOC2012')
    elif args.benchmark == 'coco':
        data_root = osp.join(args.data_root, 'MSCOCO2014')
        
    train_data = dataset.SemData(split=args.fold, shot=args.nshot, data_root=data_root, base_data_root=base_data_root, data_list=args.data_list, \
                                transform=train_transform, transform_tri=train_transform_tri, mode='train', \
                                data_set=args.benchmark, use_split_coco=args.use_split_coco, test_num=args.test_num)
    train_sampler = DistributedSampler(train_data) if args.distributed else None
    dataloader_trn = torch.utils.data.DataLoader(train_data, batch_size=args.bsz, num_workers=args.nworker, \
                                                pin_memory=True, sampler=train_sampler, drop_last=True, \
                                                shuffle=False if args.distributed else True)
    # Val
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
            val_transform_tri = transform_tri.Compose([
                transform_tri.Resize(size=args.val_size),
                transform_tri.ToTensor(),
                transform_tri.Normalize(mean=mean, std=std)])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
            val_transform_tri = transform_tri.Compose([
                transform_tri.test_Resize(size=args.val_size),
                transform_tri.ToTensor(),
                transform_tri.Normalize(mean=mean, std=std)])
        val_data = dataset.SemData(split=args.fold, shot=args.nshot, data_root=data_root, base_data_root=base_data_root, data_list=args.data_list, \
                                transform=val_transform, transform_tri=val_transform_tri, mode='val', \
                                data_set=args.benchmark, use_split_coco=args.use_split_coco, test_num=args.test_num)                                   
        dataloader_val = torch.utils.data.DataLoader(val_data, batch_size=args.bsz, shuffle=False, num_workers=args.nworker, pin_memory=False, sampler=None)

    # Train
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.nepoch):
        epoch = epoch
        dataloader_trn.sampler.set_epoch(epoch)
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, surrogate_loss, training=True)
        
        # evaluation
        if args.local_rank == 0:
            if epoch > -1:
                with torch.no_grad():
                    val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, surrogate_loss, training=False)
                Logger.save_last_model(model, epoch, val_miou)
                # Save the best model
                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    Logger.save_model_miou(model, epoch, val_miou)
                Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
                Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
                Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
                Logger.tbd_writer.flush()
            else:
                Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss}, epoch)
                Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou}, epoch)
                Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou}, epoch)
                Logger.tbd_writer.flush()

    if args.local_rank == 0:
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')
