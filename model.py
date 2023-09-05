import random
import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import numpy as np
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class CMAlign_mask(nn.Module):
    def __init__(self, batch_size=8, num_pos=4, temperature=50):
        super(CMAlign_mask, self).__init__()
        self.batch_size = batch_size
        self.num_pos = num_pos
        self.criterion = nn.TripletMarginLoss(margin=0.3, p=2.0, reduce=False)
        self.temperature = temperature
        self.maskFc = nn.Linear(512, 18*9, bias=False)
        self.maskFc.apply(weights_init_classifier)

    def _random_pairs(self):
        batch_size = self.batch_size
        num_pos = self.num_pos

        pos = []
        for batch_index in range(batch_size):
            pos_idx = random.sample(list(range(num_pos)), num_pos)
            pos_idx = np.array(pos_idx) + num_pos*batch_index
            pos = np.concatenate((pos, pos_idx))
        pos = pos.astype(int)

        neg = []
        for batch_index in range(batch_size):
            batch_list = list(range(batch_size))
            batch_list.remove(batch_index)
            
            batch_idx = random.sample(batch_list, num_pos)
            neg_idx = random.sample(list(range(num_pos)), num_pos)

            batch_idx, neg_idx = np.array(batch_idx), np.array(neg_idx)
            neg_idx = batch_idx*num_pos + neg_idx
            neg = np.concatenate((neg, neg_idx))
        neg = neg.astype(int)

        return {'pos': pos, 'neg': neg}

    def _define_pairs(self):
        pairs_v = self._random_pairs()
        pos_v, neg_v = pairs_v['pos'], pairs_v['neg']

        pairs_t = self._random_pairs()
        pos_t, neg_t = pairs_t['pos'], pairs_t['neg']
        
        pos_v += self.batch_size*self.num_pos
        neg_v += self.batch_size*self.num_pos

        return {'pos': np.concatenate((pos_v, pos_t)), 'neg': np.concatenate((neg_v, neg_t))}

    def feature_similarity(self, feat_q, feat_k):
        batch_size, fdim, h, w = feat_q.shape
        feat_q = feat_q.view(batch_size, fdim, -1)
        feat_k = feat_k.view(batch_size, fdim, -1)

        feature_sim = torch.bmm(F.normalize(feat_q, dim=1).permute(0,2,1), F.normalize(feat_k, dim=1))
        return feature_sim

    def matching_probability(self, feature_sim):
        M, _ = feature_sim.max(dim=-1, keepdim=True)
        feature_sim = feature_sim - M # for numerical stability
        exp = torch.exp(self.temperature*feature_sim)
        exp_sum = exp.sum(dim=-1, keepdim=True)
        return exp / exp_sum

    def soft_warping(self, matching_pr, feat_k):
        batch_size, fdim, h, w = feat_k.shape
        feat_k = feat_k.view(batch_size, fdim, -1)
        feat_warp = torch.bmm(matching_pr, feat_k.permute(0,2,1))
        feat_warp = feat_warp.permute(0,2,1).view(batch_size, fdim, h, w)
        
        return feat_warp

    def reconstruct(self, mask, feat_warp, feat_q):
        return mask*feat_warp + (1.0-mask)*feat_q

    def compute_mask(self, feat, text):
        batch_size, fdim, h, w = feat.shape
        norms = self.maskFc(text) #* torch.norm(feat, p=2, dim=1).view(batch_size, h*w)
        norms -= norms.min(dim=-1, keepdim=True)[0]
        norms /= norms.max(dim=-1, keepdim=True)[0] + 1e-12
        mask = norms.view(batch_size, 1, h, w)

        return mask

    def compute_comask(self, matching_pr, mask_q, mask_k):
        batch_size, mdim, h, w = mask_q.shape
        mask_q = mask_q.view(batch_size, -1, 1)
        mask_k = mask_k.view(batch_size, -1, 1)
        comask = mask_q * torch.bmm(matching_pr, mask_k)
        
        comask = comask.view(batch_size, -1)
        comask -= comask.min(dim=-1, keepdim=True)[0]
        comask /= comask.max(dim=-1, keepdim=True)[0] + 1e-12
        comask = comask.view(batch_size, mdim, h, w)
        
        return comask.detach()

    def forward(self, feat_v, feat_t, text):
        feat = torch.cat([feat_v, feat_t], dim=0)
        mask = self.compute_mask(feat, text)
        batch_size, fdim, h, w = feat.shape

        pairs = self._define_pairs()
        pos_idx, neg_idx = pairs['pos'], pairs['neg']

        # positive
        feat_target_pos = feat[pos_idx]
        feature_sim = self.feature_similarity(feat, feat_target_pos)
        matching_pr = self.matching_probability(feature_sim)
        
        comask_pos = self.compute_comask(matching_pr, mask, mask[pos_idx])
        feat_warp_pos = self.soft_warping(matching_pr, feat_target_pos)
        feat_recon_pos = self.reconstruct(mask, feat_warp_pos, feat)

        # negative
        feat_target_neg = feat[neg_idx]
        feature_sim = self.feature_similarity(feat, feat_target_neg)
        matching_pr = self.matching_probability(feature_sim)
        
        feat_warp = self.soft_warping(matching_pr, feat_target_neg)
        feat_recon_neg = self.reconstruct(mask, feat_warp, feat)

        # feat = feat.permute(0,2,3,1)
        # feat_recon_neg = feat_recon_neg.permute(0,2,3,1)
        # feat_recon_pos_ = feat_recon_pos.permute(0,2,3,1)

        loss = torch.mean(comask_pos.squeeze(1) * self.criterion(feat, feat_recon_pos, feat_recon_neg))

        return {'feat': feat_recon_pos, 'loss': loss}

class GeMP(nn.Module):
  def __init__(self, p=3.0, eps=1e-12):
    super(GeMP, self).__init__()
    self.p = p
    self.eps = eps

  def forward(self, x):
    p, eps = self.p, self.eps
    if x.ndim != 2:
      batch_size, fdim = x.shape[:2]
      x = x.view(batch_size, fdim, -1)
    return (torch.mean(x**p, dim=-1)+eps)**(1/p)

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x

class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x

class base_resnet_align(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet_align, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        feat4 = self.base.layer4(x)
        return {'feat3':x,'feat4':feat4}

class model(nn.Module):
    def __init__(self,  class_num=498, model_clip=None, batch_size=8, num_pos=4, arch='resnet50', train_multi_query=True, test_multi_query=True):
        super(model, self).__init__()

        self.clip = model_clip
        self.train_multi_query = train_multi_query
        self.test_multi_query = test_multi_query

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet_align(arch=arch)

        pool_dim = 2048
        self.nonLocal = Non_local(64)

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        
        self.pool = GeMP()
        self.cmalign = CMAlign_mask(batch_size, num_pos)

    def forward(self, x1, x2, text, style, modal=0):
        # x1, x2: [bs, 3, 288, 144]
        if modal == 0:
            x1 = self.visible_module(x1) # [bs, 64, 72, 36]
            if self.train_multi_query:
                _x2 = []
                for id,i in enumerate(x2):
                    _x2.append(torch.mean(self.thermal_module(i[:style[id]]), dim=0))
                x2 = self.nonLocal(torch.stack(_x2,0))
            else:
                x2 = self.thermal_module(x2)
                
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            if self.test_multi_query:
                _x2 = []
                for id,i in enumerate(x2):
                    _x2.append(torch.mean(self.thermal_module(i[:style[id]]), dim=0))
                x = self.nonLocal(torch.stack(_x2,0))
            else:
                x = self.thermal_module(x2)
        # x.shape = [bs, 64,72,36]
        
        # shared block
        feat = self.base_resnet(x)

        if self.training:
            ### layer3
            text = self.clip.encode_text(text).float()

            feat3 = feat['feat3']
            batch_size, fdim, h, w = feat3.shape
            out3 = self.cmalign(feat3[:batch_size//2], feat3[batch_size//2:], text)

            feat3_recon = self.base_resnet.base.layer4(out3['feat'])
            feat3_recon_p = self.pool(feat3_recon)
            cls_ic_layer3 = self.classifier(self.bottleneck(feat3_recon_p))

            ### layer4
            feat4 = feat['feat4']
            feat4_p = self.pool(feat4)
            batch_size, fdim, h, w = feat4.shape
            out4 = self.cmalign(feat4[:batch_size//2], feat4[batch_size//2:], text)

            feat4_recon = out4['feat']
            feat4_recon_p = self.pool(feat4_recon)
            cls_ic_layer4 = self.classifier(self.bottleneck(feat4_recon_p))

            ### compute losses
            cls_id = self.classifier(self.bottleneck(feat4_p))
            loss_dt = out3['loss'] + out4['loss']

            return {
                'feat4_p': feat4_p,
                'cls_id': cls_id, 
                'cls_ic_layer3': cls_ic_layer3, 
                'cls_ic_layer4': cls_ic_layer4, 
                'loss_dt': loss_dt
                }

        else:
            feat4 = feat['feat4']
            batch_size, fdim, h, w = feat4.shape
            feat4_flatten = feat['feat4'].view(batch_size, fdim, -1)
            feat4_p = self.pool(feat4_flatten)
            cls_id = self.classifier(self.bottleneck(feat4_p))
            return {
                'feat4_p': feat4_p,
                'cls_id': cls_id,
                'feat4_p_norm': F.normalize(feat4_p, p=2.0, dim=1)
            }