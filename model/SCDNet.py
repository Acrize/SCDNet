from functools import reduce
from operator import add
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from .base.swin_transformer import SwinTransformer
from model.base.transformer import MultiHeadedAttention, PositionalEncoding

def similarity_func(feature_q, fg_proto, dim=1):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=dim)*10
        return similarity_fg

def masked_average_pooling(feature, mask):
    mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
    masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)
    return masked_feature

# load word embeddings
def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        embed = torch.from_numpy(pickle.load(f, encoding="latin-1"))
    embed.requires_grad = False
    return embed   # [C, 300]


class SCDNet(nn.Module):

    def __init__(self, backbone, pretrained_path, use_original_imgsize, batch_size, dataset, nshot, fold):
        super(SCDNet, self).__init__()

        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize
        self.dataset = dataset
        self.nshot = nshot
        self.fold = fold

        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        self.model = SCDNet_model(in_channels=self.feat_channels, stack_ids=self.stack_ids, backbone=self.backbone, dataset=self.dataset, fold=self.fold)

        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.mse_loss = nn.MSELoss()

    def forward(self, query_img, support_img, support_mask, class_id, surrogates, training):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img)
            support_feats = self.extract_feats(support_img)

        outputs = self.model(query_feats, support_feats, support_mask.clone(), class_id, surrogates, self.nshot, training)

        return outputs

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []

        if self.backbone == 'swin':
            _ = self.feature_extractor.forward_features(img)
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                h = int(hw ** 0.5)
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)
        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch['query_img']
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']

        if nshot == 1:
            logit_mask = self(query_img, support_imgs[:, 0], support_masks[:, 0])
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                for k in range(nshot):
                    support_feats = self.extract_feats(support_imgs[:, k])
                    n_support_feats.append(support_feats)
            logit_mask = self.model(query_feats, n_support_feats, support_masks.clone(), nshot)

        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(logit_mask, support_imgs[0].size()[2:], mode='bilinear', align_corners=True)

        return logit_mask.argmax(dim=1)

    def compute_objective(self, logit_mask, gt_mask, confusion_map):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask_flat = gt_mask.view(bsz, -1).long()
        
        mask_loss = self.cross_entropy_loss(logit_mask, gt_mask_flat)
        
        if confusion_map is not None:            
            confusion_map = confusion_map * (1 - gt_mask) + gt_mask
            confusion_map = confusion_map.view(bsz, -1).long()
            assert confusion_map.shape == mask_loss.shape, "loss and confusion map do not have the same size"    
            
        return mask_loss.mean()
    
    def compute_reconstruction_loss(self, recon_prototype, query_feat, query_mask):
        prototype = masked_average_pooling(query_feat, query_mask)
        return self.mse_loss(recon_prototype, prototype)

    def train_mode(self):
        self.train()
        self.feature_extractor.eval()

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
  
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class SCDNet_model(nn.Module):
    def __init__(self, in_channels, stack_ids, backbone, dataset, fold):
        super(SCDNet_model, self).__init__()

        self.stack_ids = stack_ids
        self.backbone = backbone
        self.dataset = dataset
        self.fold = fold
        
        # text embedding
        self.embeddings = load_obj('embeddings/word2vec_' + self.dataset).cuda()
        self.embed_dim = self.embeddings.shape[1]
        
        self.label2surr = {}
        self.label_fold = []
        # PASCAL
        if self.dataset == 'pascal':
            self.train_classes = 15
            self.all_classes = 20
            idx = 0
            for i in range(1, 21):
                if (i - 1) // 5  != self.fold:
                    self.label_fold.append(i)
                    self.label2surr[i] = idx
                    idx = idx + 1
        # COCO
        elif self.dataset == 'coco':
            self.train_classes = 60
            self.all_classes = 80
            idx = 0
            for i in range(1, 81):
                if i % 4  != self.fold + 1:
                    self.label_fold.append(i)
                    self.label2surr[i] = idx
                    idx = idx + 1

        self.u = 0.995
        self.v = 1.005


        # DCAMA blocks
        self.DCAMA_blocks = nn.ModuleList()
        self.pe = nn.ModuleList()
        for inch in in_channels[1:]:
            self.DCAMA_blocks.append(MultiHeadedAttention(h=8, d_model=inch, dropout=0.5))
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))

        outch1, outch2, outch3 = 16, 64, 128

        # MSCA blocks
        self.pre_conv32 = self.build_conv_block(in_channels[-1]+self.embed_dim, [in_channels[-1]]*2, [1]*2, [1]*2) # 1/32
        self.pre_conv16 = self.build_conv_block(in_channels[-2]+self.embed_dim, [in_channels[-2]]*2, [1]*2, [1]*2) # 1/16
        self.pre_conv8 = self.build_conv_block(in_channels[-3]+self.embed_dim, [in_channels[-3]]*2, [3,1], [1]*2) # 1/8

        self.change_channel = nn.ModuleList()
        self.change_channel.append(nn.Sequential(
            nn.Conv2d(in_channels[-3], in_channels[-2], kernel_size=1, padding=0),
            nn.ReLU(),
        ))
        self.change_channel.append(nn.Sequential(
            nn.Conv2d(in_channels[-1], in_channels[-2], kernel_size=1, padding=0),
            nn.ReLU(),
        ))

        if self.backbone == 'resnet50':
            self.mlp = MLP(in_channels[-2]+self.embed_dim, 2048)
            self.mlp_reverse = MLP(2048+self.embed_dim, in_channels[1])
        else:
            self.mlp = MLP(in_channels[-2]+self.embed_dim, 4096)
            self.mlp_reverse = MLP(4096+self.embed_dim, in_channels[1])

        # conv blocks
        self.conv1 = self.build_conv_block(stack_ids[3]-stack_ids[2], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1]) # 1/32
        self.conv2 = self.build_conv_block(stack_ids[2]-stack_ids[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1]) # 1/16
        self.conv3 = self.build_conv_block(stack_ids[1]-stack_ids[0], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1]) # 1/8

        self.conv4 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
        self.conv5 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8

        # mixer blocks
        self.mixer1 = nn.Sequential(nn.Conv2d(outch3+2*in_channels[1]+2*in_channels[0], outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer3 = nn.Sequential(nn.Conv2d(outch1, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))

    def forward(self, query_feats, support_feats, support_mask, class_id, surrogates, nshot=1, training=True):
        coarse_masks = []
        support_prototypes = []
        
        class_id = class_id + 1

        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx < self.stack_ids[0]: continue

            bsz, ch, ha, wa = query_feat.size()
            
            # Prior Masks
            # 1-shot
            if nshot == 1:
                support_feat = support_feats[idx]
                
                support_prototype_fg = masked_average_pooling(support_feat, (support_mask == 1).float())[None, :]  # [1, B, C]
                support_prototype_bg = masked_average_pooling(support_feat, (support_mask == 0).float())[None, :]  # [1, B, C]
                
            # n-shot
            else:
                support_prototypes_fg = []
                support_prototypes_bg = []
                
                for k in range(nshot):
                    support_feat = support_feats[k][idx]
                    
                    support_prototype_fg_k = masked_average_pooling(support_feat, (support_mask == 1).float())[None, :]  # [1, B, C]
                    support_prototype_bg_k = masked_average_pooling(support_feat, (support_mask == 0).float())[None, :]  # [1, B, C]
                    
                    support_prototypes_fg.append(support_prototype_fg_k)
                    support_prototypes_bg.append(support_prototype_bg_k)
                    
                support_prototype_fg = torch.stack(support_prototypes_fg).mean(dim=0)  # [N, 1, B, C] -> [1, B, C]
                support_prototype_bg = torch.stack(support_prototypes_bg).mean(dim=0)  # [N, 1, B, C] -> [1, B, C]
                
            prior_mask_fg = similarity_func(query_feat, support_prototype_fg.squeeze(0).unsqueeze(-1).unsqueeze(-1)).unsqueeze(1)  # [B, 1, H, W]
            prior_mask_bg = similarity_func(query_feat, support_prototype_bg.squeeze(0).unsqueeze(-1).unsqueeze(-1)).unsqueeze(1)  # [B, 1, H, W]
            
            # [B, 2, H, W]
            prior_mask_fb = torch.cat([prior_mask_fg, prior_mask_bg], dim=1).softmax(dim=1)
            support_mask_fb = F.interpolate(torch.stack([support_mask == 1, support_mask == 0], dim=1).float(), size=support_feat.shape[-2:], mode='bilinear', align_corners=True)
            
            for b in range(bsz):
                fg_embeddings = self.embeddings[class_id].to(support_feat)  # [B, 300]
                bg_embeddings = self.embeddings[0].unsqueeze(0).expand(bsz, -1).to(support_feat)  # [300] -> [B, 300]
                
                fb_embedding = torch.stack([fg_embeddings, bg_embeddings], dim=1)  # [B, 2, 300]
                
                support_text_semantic_map = torch.einsum('bchw,bcd->bdhw', support_mask_fb, fb_embedding)  # [B, 300, H, W]
                query_text_semantic_map = torch.einsum('bchw,bcd->bdhw', prior_mask_fb, fb_embedding)  # [B, 300, H, W]

            # MSCA
            query_feat_text = torch.cat([query_feat, query_text_semantic_map], dim=1)
            
            # 1-shot
            if nshot == 1:
                support_feat = support_feats[idx]
                support_feat_text = torch.cat([support_feat, support_text_semantic_map], dim=1)
                
                if idx < self.stack_ids[1]:
                    query_feat = self.pre_conv8(query_feat_text) + query_feat
                    support_feats[idx] = self.pre_conv8(support_feat_text) + support_feats[idx]
                    
                    support_feat_aligned = self.change_channel[0](support_feats[idx])
                    
                elif idx < self.stack_ids[2]:
                    query_feat = self.pre_conv16(query_feat_text) + query_feat
                    support_feats[idx] = self.pre_conv16(support_feat_text) + support_feats[idx]
                    
                    support_feat_aligned = support_feats[idx]
                    
                else:
                    query_feat = self.pre_conv32(query_feat_text) + query_feat
                    support_feats[idx] = self.pre_conv32(support_feat_text) + support_feats[idx]
                    
                    support_feat_aligned = self.change_channel[1](support_feats[idx])
                    
                support_prototype = masked_average_pooling(support_feat_aligned, (support_mask == 1).float())[None, :]  # [1, B, C]
                support_prototype_text = torch.cat([support_prototype, fg_embeddings.unsqueeze(0)], dim=2)
                
                support_prototypes.append(support_prototype_text)
                
            # n-shot
            else:
                if idx < self.stack_ids[1]:
                    query_feat = self.pre_conv8(query_feat_text) + query_feat
                    for k in range(nshot):
                        support_feat_text = torch.cat([support_feats[k][idx], support_text_semantic_map], dim=1)
                        support_feats[k][idx] = self.pre_conv8(support_feat_text) + support_feats[k][idx]
                        
                        support_feat_aligned = self.change_channel[0](support_feats[k][idx])
                        support_prototype = masked_average_pooling(support_feat_aligned, (support_mask == 1).float())[None, :]  # [1, B, C]
                        support_prototype_text = torch.cat([support_prototype, fg_embeddings.unsqueeze(0)], dim=2)
                        support_prototypes.append(support_prototype_text)
                        
                elif idx < self.stack_ids[2]:
                    query_feat = self.pre_conv16(query_feat_text) + query_feat
                    for k in range(nshot):
                        support_feat_text = torch.cat([support_feats[k][idx], support_text_semantic_map], dim=1)
                        support_feats[k][idx] = self.pre_conv16(support_feat_text) + support_feats[k][idx]
                        
                        support_feat_aligned = support_feats[k][idx]
                        support_prototype = masked_average_pooling(support_feat_aligned, (support_mask == 1).float())[None, :]  # [1, B, C]
                        support_prototype_text = torch.cat([support_prototype, fg_embeddings.unsqueeze(0)], dim=2)
                        support_prototypes.append(support_prototype_text)
                        
                else:
                    query_feat = self.pre_conv32(query_feat_text) + query_feat
                    for k in range(nshot):
                        support_feat_text = torch.cat([support_feats[k][idx], support_text_semantic_map], dim=1)
                        support_feats[k][idx] = self.pre_conv32(support_feat_text) + support_feats[k][idx]
                        
                        support_feat_aligned = self.change_channel[1](support_feats[k][idx])
                        support_prototype = masked_average_pooling(support_feat_aligned, (support_mask == 1).float())[None, :]  # [1, B, C]
                        support_prototype_text = torch.cat([support_prototype, fg_embeddings.unsqueeze(0)], dim=2)
                        support_prototypes.append(support_prototype_text)            

            # reshape the input feature and mask
            query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
            if nshot == 1:
                support_feat = support_feats[idx]
                mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                     align_corners=True).view(support_feat.size()[0], -1)
                support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()
            else:
                support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
                support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
                mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                    for k in support_mask])
                mask = mask.view(bsz, -1)

            mask = mask.unsqueeze(1)

            # DCAMA blocks forward
            if idx < self.stack_ids[1]:
                coarse_mask, _ = self.DCAMA_blocks[0](self.pe[0](query), self.pe[0](support_feat), mask)
            elif idx < self.stack_ids[2]:
                coarse_mask, _ = self.DCAMA_blocks[1](self.pe[1](query), self.pe[1](support_feat), mask)
            else:
                coarse_mask, _ = self.DCAMA_blocks[2](self.pe[2](query), self.pe[2](support_feat), mask)

            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, 1, ha, wa)) # [B, HW, 1] -> [B, 1, HW] -> [B, 1, H, W]

        # multi-scale conv blocks forward
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[3]-1-self.stack_ids[0]].size()
        coarse_masks1 = torch.stack(coarse_masks[self.stack_ids[2]-self.stack_ids[0]:self.stack_ids[3]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[2]-1-self.stack_ids[0]].size()
        coarse_masks2 = torch.stack(coarse_masks[self.stack_ids[1]-self.stack_ids[0]:self.stack_ids[2]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[1]-1-self.stack_ids[0]].size()
        coarse_masks3 = torch.stack(coarse_masks[0:self.stack_ids[1]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)

        coarse_masks1 = self.conv1(coarse_masks1)
        coarse_masks2 = self.conv2(coarse_masks2)
        coarse_masks3 = self.conv3(coarse_masks3)

        # multi-scale cascade (pixel-wise addition)
        coarse_masks1 = F.interpolate(coarse_masks1, coarse_masks2.size()[-2:], mode='bilinear', align_corners=True)
        mix = coarse_masks1 + coarse_masks2
        mix = self.conv4(mix)

        mix = F.interpolate(mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True)
        mix = mix + coarse_masks3
        mix = self.conv5(mix)

        # skip connect 1/8 and 1/4 features (concatenation)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[1] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]).max(dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[1] - 1], support_feat), 1)

        upsample_size = (mix.size(-1) * 2,) * 2
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[0] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]).max(dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1], support_feat), 1)

        # mixer blocks forward
        out = self.mixer1(mix)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        out = self.mixer2(out)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.mixer3(out)

        projecter_in = torch.cat(support_prototypes, dim=0).mean(dim=0)
        pseudo_surrogate = self.mlp(projecter_in)

        # SDLR
        if training:
            query_feat = query_feats[self.stack_ids[1] - 1].detach()  # [B, C, H, W]
            
            surrogate_text = torch.cat([surrogates, self.embeddings[self.label_fold]], dim=1)  # [Cls, dim_Proto] + [Cls, D]
            prototype_recon = self.mlp_reverse(surrogate_text)  # [Cls, C]
            
            prototype_nontarget = []
            for b in range(bsz):
                s = self.label2surr[class_id[b].item()]
                prototype_nontarget.append(torch.cat([prototype_recon[:s], prototype_recon[s+1:]], dim=0))    
            prototype_batched =  torch.stack(prototype_nontarget, dim=1)  # [Cls-1, B, C]
            
            prototype_target = prototype_recon[[self.label2surr[label.item()] for label in class_id]]  # [B, C]
            
            confusion_maps = similarity_func(query_feat.unsqueeze(0).expand(self.train_classes-1, *query_feat.shape), 
                                            prototype_batched.unsqueeze(-1).unsqueeze(-1), dim=2) # [Cls-1, B, H, W]
            adjacent_scores = similarity_func(prototype_target.unsqueeze(0).expand(self.train_classes-1, *prototype_target.shape),
                                        prototype_batched, dim=2)  # [Cls-1, B]

            # Synthesize an unscaled confusion map
            confusion_map = torch.einsum('cb,cbhw->bhw', adjacent_scores, confusion_maps).unsqueeze(1)  # [B, 1, H, W]
            
            # Scale the consufion map
            confusion_map = self.u + (self.v - self.u) * (confusion_map - confusion_map.min()) / (confusion_map.max() - confusion_map.min())
            bsz, ch, ha, wa = query_feat.shape
            confusion_map = F.interpolate(confusion_map, [ha*8, wa*8]).squeeze(1)
        else:
            prototype_target = confusion_map = None

        return logit_mask, pseudo_surrogate, confusion_map, prototype_target, query_feat

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)
