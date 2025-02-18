# -*- coding:utf-8 -*-
# author: Xinge
# @file: segmentator_3d_asymm_spconv.py

import numpy as np
import spconv.pytorch as spconv
from spconv.pytorch import ops
from spconv.pytorch import functional as Fsp
import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef2")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef4")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        
        # shortcut = shortcut.replace_feature(self.conv1_2(shortcut))
        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))
        
        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))
        
        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))        
        resA = resA.replace_feature(resA.features + shortcut.features)
        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef2")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef4")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key+'up1')
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key+'up2')
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key+'up3')
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))

        ## upsample
        upA = self.up_subm(upA)

        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))

        return upE
    
# https://github.com/traveller59/spconv/issues/519
class SparseGlobalMaxPool_OLD(spconv.SparseModule):
    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        ndim = len(spatial_shape)
        ksize = spatial_shape

        out_spatial_shape = ops.get_conv_output_size(
            spatial_shape, ksize, [1] * ndim, [0] * ndim,[1] * ndim)
        outids, indice_pairs, indice_pairs_num = ops.get_indice_pairs(indices, batch_size, spatial_shape, spconv.ConvAlgo.Native, ksize, [1] * ndim ,[0] * ndim, [1]* ndim, [0]* ndim, False)
        out_features = Fsp.indice_maxpool(features, indice_pairs.to(device),
                                          indice_pairs_num.to(device),
                                          outids.shape[0])
        out_tensor = spconv.SparseConvTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor
    
class SparseGlobalAvgPool(spconv.SparseModule):
    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        ndim = len(spatial_shape)
        ksize = spatial_shape
        _, out_feat = features.shape
        
        return_feat = torch.zeros((0, out_feat)).to(device)
        
        _, cnt = torch.unique(indices[:, 0], return_counts=True)
        feature_len = 0
        prev_feature_len = 0
        
        for batch in range(batch_size):
            feature_len += cnt[batch]
            single_batch_feature = features[prev_feature_len:feature_len,:]
            prev_feature_len += cnt[batch]
            mean_feature = torch.mean(single_batch_feature, dim=0)
            mean_feature = mean_feature.unsqueeze(0)
            return_feat = torch.cat((return_feat, mean_feature), dim=0)
        
        return return_feat

class SparseGlobalMaxPool(spconv.SparseModule):
    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        ndim = len(spatial_shape)
        ksize = spatial_shape
        _, out_feat = features.shape
        
        return_feat = torch.zeros((0, out_feat)).to(device)
        
        _, cnt = torch.unique(indices[:, 0], return_counts=True)
        feature_len = 0
        prev_feature_len = 0
        
        for batch in range(batch_size):
            feature_len += cnt[batch]
            single_batch_feature = features[prev_feature_len:feature_len,:]
            prev_feature_len += cnt[batch]
            max_feature, max_idx = torch.max(single_batch_feature, dim=0)
            max_feature = max_feature.unsqueeze(0)
            return_feat = torch.cat((return_feat, max_feature), dim=0)
        
        return return_feat
            
class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef2")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut3.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)

        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        # 512 -> 128
        self.weather_logits = spconv.SubMConv3d(16 * init_size, 128, indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.weather_max_pool = SparseGlobalMaxPool()
        self.weather_fc1 = nn.Linear(128, 3)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down4c)
        weather_feat2 = self.weather_max_pool(weather_feat1)
        weather_result = self.weather_fc1(weather_feat2.features)
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result
class Asymm_3d_spconv_original(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_original, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y    

class Asymm_3d_spconv_clf_v1(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v1, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        # 512 -> 128
        self.weather_logits = spconv.SubMConv3d(16 * init_size, 128, indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.weather_max_pool = SparseGlobalMaxPool()
        self.weather_fc1 = nn.Linear(128, 3)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down4c)
        weather_feat2 = self.weather_max_pool(weather_feat1)
        weather_result = self.weather_fc1(weather_feat2.features)
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result
    
class Asymm_3d_spconv_clf_v2(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v2, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        # 512 -> 256
        self.weather_logits = spconv.SubMConv3d(16 * init_size, 256, indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.weather_max_pool = SparseGlobalMaxPool()
        # 256 -> 64
        self.weather_fc1 = nn.Linear(256, 64)
        # 64 -> 3
        self.weather_fc2 = nn.Linear(64, 3)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down4c)
        weather_feat2 = self.weather_max_pool(weather_feat1)
        weather_feat3 = self.weather_fc1(weather_feat2.features)
        weather_result = self.weather_fc2(weather_feat3)
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result
    
class Asymm_3d_spconv_clf_v3(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v3, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        # 512 -> 256
        self.weather_logits = spconv.SubMConv3d(16 * init_size, 256, indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.weather_max_pool = SparseGlobalMaxPool()
        # 256 -> 64
        self.weather_fc1 = nn.Linear(256, 64)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(64)
        # 64 -> 3
        self.weather_fc2 = nn.Linear(64, 3)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down4c)
        weather_feat2 = self.weather_max_pool(weather_feat1)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2.features)))
        
        weather_result = self.weather_fc2(weather_feat3)
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result
    
class Asymm_3d_spconv_clf_v4(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v4, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        # 512 -> 256
        # init_size = 16
        self.weather_logits = spconv.SubMConv3d(16 * init_size, 256, indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.weather_max_pool = SparseGlobalMaxPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(256, 128)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(128)
        # 128 -> 64
        self.weather_fc2 = nn.Linear(128,64)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(64)
        # 128 -> 64
        self.weather_fc3 = nn.Linear(64,3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down4c)
        weather_feat2 = self.weather_max_pool(weather_feat1)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_result = self.weather_fc3(weather_feat4)
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result    
    
class Asymm_3d_spconv_clf_v5(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v5, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        # 512 -> 256
        # init_size = 16
        self.weather_logits = spconv.SubMConv3d(8 * init_size, 128, indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.weather_max_pool = SparseGlobalMaxPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(128, 64)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(64)
        # 128 -> 64
        self.weather_fc2 = nn.Linear(64,32)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(32)
        # 128 -> 64
        self.weather_fc3 = nn.Linear(32,3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down3c)
        weather_feat2 = self.weather_max_pool(weather_feat1)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2.features)))
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_result = self.weather_fc3(weather_feat4)
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result    

class Asymm_3d_spconv_clf_v6(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v6, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        # 512 -> 256
        # init_size = 16
        self.weather_logits = spconv.SubMConv3d(4 * init_size, 64, indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.weather_max_pool = SparseGlobalMaxPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(64, 32)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(32)
        # 128 -> 64
        self.weather_fc2 = nn.Linear(32,16)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(16)
        # 128 -> 64
        self.weather_fc3 = nn.Linear(16,3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down2c)
        weather_feat2 = self.weather_max_pool(weather_feat1)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_result = self.weather_fc3(weather_feat4)
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result    

class Asymm_3d_spconv_clf_v7(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v7, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        '''
        Weather Clf Net
        '''
        # 512 -> 256
        # init_size = 32
        in_channel = int(4 * init_size)
        self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.sp_relu1 = nn.LeakyReLU()
        self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        self.weather_avg_pool = SparseGlobalAvgPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(int(in_channel/2), int(in_channel/4))
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(int(in_channel/4))
        # 128 -> 64
        self.weather_fc2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(int(in_channel/8))
        # 128 -> 64
        self.weather_fc3 = nn.Linear(int(in_channel/8), 3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down2c)
        weather_feat1 = weather_feat1.replace_feature(self.sp_relu1(weather_feat1.features))
        weather_feat1 = weather_feat1.replace_feature(self.sp_bn1(weather_feat1.features))        

        weather_feat2 = self.weather_avg_pool(weather_feat1)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_result = self.weather_fc3(weather_feat4)        
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result
    
class Asymm_3d_spconv_clf_v8(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v8, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        '''
        Weather Clf Net
        '''
        # init_size = 32
        in_channel = 8 * init_size
        self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.sp_relu1 = nn.LeakyReLU()
        self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        self.weather_avg_pool = SparseGlobalAvgPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(int(in_channel/2), int(in_channel/4))
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(int(in_channel/4))
        # 128 -> 64
        self.weather_fc2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(int(in_channel/8))
        # 128 -> 64
        self.weather_fc3 = nn.Linear(int(in_channel/8), 3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down3c)
        weather_feat1 = weather_feat1.replace_feature(self.sp_relu1(weather_feat1.features))
        weather_feat1 = weather_feat1.replace_feature(self.sp_bn1(weather_feat1.features))
        
        weather_feat2 = self.weather_avg_pool(weather_feat1)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_result = self.weather_fc3(weather_feat4)        
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result
    
class Asymm_3d_spconv_clf_v9(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v9, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        '''
        Weather Clf Net
        '''
        # init_size = 32
        in_channel = 16 * init_size
        self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.sp_relu1 = nn.LeakyReLU()
        self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        self.weather_avg_pool = SparseGlobalAvgPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(int(in_channel/2), int(in_channel/4))
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(int(in_channel/4))
        # 128 -> 64
        self.weather_fc2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(int(in_channel/8))
        # 128 -> 64
        self.weather_fc3 = nn.Linear(int(in_channel/8), 3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down4c)
        weather_feat1 = weather_feat1.replace_feature(self.sp_relu1(weather_feat1.features))
        weather_feat1 = weather_feat1.replace_feature(self.sp_bn1(weather_feat1.features))
        
        weather_feat2 = self.weather_avg_pool(weather_feat1)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_result = self.weather_fc3(weather_feat4)        
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result
    
class Asymm_3d_spconv_clf_v10(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v10, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        '''
        Weather Clf Net
        '''
        # init_size = 32
        in_channel = 16 * init_size
        # self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        # self.sp_relu1 = nn.LeakyReLU()
        # self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        self.weather_avg_pool = SparseGlobalAvgPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(int(in_channel), int(in_channel/2))
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(int(in_channel/2))
        # 128 -> 64
        self.weather_fc2 = nn.Linear(int(in_channel/2), int(in_channel/4))
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(int(in_channel/4))
        # 128 -> 64
        self.weather_fc3 = nn.Linear(int(in_channel/4), 3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
        
        # weather_feat1 = self.weather_logits(down4c)
        # weather_feat1 = weather_feat1.replace_feature(self.sp_relu1(weather_feat1.features))
        # weather_feat1 = weather_feat1.replace_feature(self.sp_bn1(weather_feat1.features))
        
        weather_feat2 = self.weather_avg_pool(down4c)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_result = self.weather_fc3(weather_feat4)        
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result        
    
class Asymm_3d_spconv_clf_v11(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v11, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        '''
        Weather Clf Net
        '''
        # init_size = 32
        in_channel = 16 * init_size
        self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.sp_relu1 = nn.ReLU()
        self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        self.weather_avg_pool = SparseGlobalAvgPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(int(in_channel/2), int(in_channel/4))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(int(in_channel/4))
        # 128 -> 64
        self.weather_fc2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(int(in_channel/8))
        # 128 -> 64
        self.weather_fc3 = nn.Linear(int(in_channel/8), 3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down4c)
        weather_feat1 = weather_feat1.replace_feature(self.sp_relu1(weather_feat1.features))
        weather_feat1 = weather_feat1.replace_feature(self.sp_bn1(weather_feat1.features))
        
        weather_feat2 = self.weather_avg_pool(weather_feat1)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_result = self.weather_fc3(weather_feat4)        
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result

class Asymm_3d_spconv_clf_v12(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v12, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        '''
        Weather Clf Net
        '''
        # init_size = 32
        in_channel = 16 * init_size
        self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.sp_relu1 = nn.LeakyReLU()
        self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        self.weather_avg_pool = SparseGlobalAvgPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(int(in_channel/2), int(in_channel/4))
        self.weather_shortcut1 = nn.Linear(int(in_channel/2), int(in_channel/4))
        
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(int(in_channel/4))
        # 128 -> 64
        self.weather_fc2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.weather_shortcut2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(int(in_channel/8))
        # 128 -> 64
        self.weather_fc3 = nn.Linear(int(in_channel/8), 3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down4c)
        weather_feat1 = weather_feat1.replace_feature(self.sp_relu1(weather_feat1.features))
        weather_feat1 = weather_feat1.replace_feature(self.sp_bn1(weather_feat1.features))
        
        weather_feat2 = self.weather_avg_pool(weather_feat1)
        weather_feat2_shorcut = self.weather_shortcut1(weather_feat2)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        
        # weather_feat : 2,3 residual
        weather_feat3 = weather_feat3 + weather_feat2_shorcut
        
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_feat3_shorcut = self.weather_shortcut2(weather_feat3)
        # weather_feat : 3,4 residual
        weather_feat4 = weather_feat4 + weather_feat3_shorcut
        weather_result = self.weather_fc3(weather_feat4)        
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result    

class Asymm_3d_spconv_clf_v13(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v13, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        '''
        Weather Clf Net
        '''
        # init_size = 32
        in_channel = 16 * init_size
        self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.sp_relu1 = nn.ReLU()
        self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        self.weather_avg_pool = SparseGlobalAvgPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(int(in_channel/2), int(in_channel/4))
        self.weather_shortcut1 = nn.Linear(int(in_channel/2), int(in_channel/4))
                
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(int(in_channel/4))
        # 128 -> 64
        self.weather_fc2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.weather_shortcut2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(int(in_channel/8))
        # 128 -> 64
        self.weather_fc3 = nn.Linear(int(in_channel/8), 3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
        
        weather_feat1 = self.weather_logits(down4c)
        weather_feat1 = weather_feat1.replace_feature(self.sp_relu1(weather_feat1.features))
        weather_feat1 = weather_feat1.replace_feature(self.sp_bn1(weather_feat1.features))
        
        weather_feat2 = self.weather_avg_pool(weather_feat1)
        weather_feat2_shorcut = self.weather_shortcut1(weather_feat2)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        
        # weather_feat : 2,3 residual
        weather_feat3 = weather_feat3 + weather_feat2_shorcut
        
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_feat3_shorcut = self.weather_shortcut2(weather_feat3)
        # weather_feat : 3,4 residual
        weather_feat4 = weather_feat4 + weather_feat3_shorcut
        weather_result = self.weather_fc3(weather_feat4)            
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result    

class Asymm_3d_spconv_clf_v14(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v14, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        '''
        Weather Clf Net
        '''
        # init_size = 32
        in_channel = 4 * init_size
        self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.sp_relu1 = nn.ReLU()
        self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        self.weather_avg_pool = SparseGlobalAvgPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(int(in_channel/2), int(in_channel/4))
        self.weather_shortcut1 = nn.Linear(int(in_channel/2), int(in_channel/4))
                
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(int(in_channel/4))
        # 128 -> 64
        self.weather_fc2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.weather_shortcut2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(int(in_channel/8))
        # 128 -> 64
        self.weather_fc3 = nn.Linear(int(in_channel/8), 3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
               
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)
        
        weather_feat1 = self.weather_logits(up2e)
        weather_feat1 = weather_feat1.replace_feature(self.sp_relu1(weather_feat1.features))
        weather_feat1 = weather_feat1.replace_feature(self.sp_bn1(weather_feat1.features))
        
        weather_feat2 = self.weather_avg_pool(weather_feat1)
        weather_feat2_shorcut = self.weather_shortcut1(weather_feat2)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        
        # weather_feat : 2,3 residual
        weather_feat3 = weather_feat3 + weather_feat2_shorcut
        
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_feat3_shorcut = self.weather_shortcut2(weather_feat3)
        # weather_feat : 3,4 residual
        weather_feat4 = weather_feat4 + weather_feat3_shorcut
        weather_result = self.weather_fc3(weather_feat4)   

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result    
    
class Asymm_3d_spconv_clf_v15(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v15, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")     

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        '''
        Weather Clf Net
        '''
        # init_size = 32
        in_channel = 2 * init_size
        self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        self.sp_relu1 = nn.ReLU()
        self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        self.weather_avg_pool = SparseGlobalAvgPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(int(in_channel/2), int(in_channel/4))
        self.weather_shortcut1 = nn.Linear(int(in_channel/2), int(in_channel/4))
                
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(int(in_channel/4))
        # 128 -> 64
        self.weather_fc2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.weather_shortcut2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(int(in_channel/8))
        # 128 -> 64
        self.weather_fc3 = nn.Linear(int(in_channel/8), 3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)      
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)
        
        weather_feat1 = self.weather_logits(up1e)
        weather_feat1 = weather_feat1.replace_feature(self.sp_relu1(weather_feat1.features))
        weather_feat1 = weather_feat1.replace_feature(self.sp_bn1(weather_feat1.features))
        
        weather_feat2 = self.weather_avg_pool(weather_feat1)
        weather_feat2_shorcut = self.weather_shortcut1(weather_feat2)
        weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        
        # weather_feat : 2,3 residual
        weather_feat3 = weather_feat3 + weather_feat2_shorcut
        
        weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        weather_feat3_shorcut = self.weather_shortcut2(weather_feat3)
        # weather_feat : 3,4 residual
        weather_feat4 = weather_feat4 + weather_feat3_shorcut
        weather_result = self.weather_fc3(weather_feat4) 

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result    
class Asymm_3d_spconv_clf_v17(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v17, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        # '''
        # Weather Clf Net
        # '''
        # # init_size = 32
        # in_channel = 2 * init_size
        # self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        # self.sp_relu1 = nn.ReLU()
        # self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        # self.weather_avg_pool = SparseGlobalAvgPool()
        # # 256 -> 128
        # self.weather_fc1 = nn.Linear(int(in_channel/2), int(in_channel/4))
        # self.weather_shortcut1 = nn.Linear(int(in_channel/2), int(in_channel/4))
                
        # self.relu1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(int(in_channel/4))
        # # 128 -> 64
        # self.weather_fc2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        # self.weather_shortcut2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        # self.relu2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(int(in_channel/8))
        # # 128 -> 64
        # self.weather_fc3 = nn.Linear(int(in_channel/8), 3)
        
        # self.weight_initialization()

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)      
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)
        
        # weather_feat1 = self.weather_logits(up1e)
        # weather_feat1 = weather_feat1.replace_feature(self.sp_relu1(weather_feat1.features))
        # weather_feat1 = weather_feat1.replace_feature(self.sp_bn1(weather_feat1.features))
        
        # weather_feat2 = self.weather_avg_pool(weather_feat1)
        # weather_feat2_shorcut = self.weather_shortcut1(weather_feat2)
        # weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        
        # # weather_feat : 2,3 residual
        # weather_feat3 = weather_feat3 + weather_feat2_shorcut
        
        # weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        # weather_feat3_shorcut = self.weather_shortcut2(weather_feat3)
        # # weather_feat : 3,4 residual
        # weather_feat4 = weather_feat4 + weather_feat3_shorcut
        # weather_result = self.weather_fc3(weather_feat4) 

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, None    
class Asymm_3d_spconv_clf_v18(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_v18, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        # '''
        # Weather Clf Net
        # '''
        # # init_size = 32
        # in_channel = 2 * init_size
        # self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        # self.sp_relu1 = nn.ReLU()
        # self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        self.weather_avg_pool = SparseGlobalAvgPool()
        # # 256 -> 128
        # self.weather_fc1 = nn.Linear(int(in_channel/2), int(in_channel/4))
        # self.weather_shortcut1 = nn.Linear(int(in_channel/2), int(in_channel/4))
                
        # self.relu1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(int(in_channel/4))
        # # 128 -> 64
        # self.weather_fc2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        # self.weather_shortcut2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        # self.relu2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(int(in_channel/8))
        # # 128 -> 64
        # self.weather_fc3 = nn.Linear(int(in_channel/8), 3)
        
        # self.weight_initialization()

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)      
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)
        
        # weather_feat1 = self.weather_logits(up1e)
        # weather_feat1 = weather_feat1.replace_feature(self.sp_relu1(weather_feat1.features))
        # weather_feat1 = weather_feat1.replace_feature(self.sp_bn1(weather_feat1.features))
        
        # weather_feat2 = self.weather_avg_pool(weather_feat1)
        # weather_feat2_shorcut = self.weather_shortcut1(weather_feat2)
        # weather_feat3 = self.bn1(self.relu1(self.weather_fc1(weather_feat2)))
        
        # # weather_feat : 2,3 residual
        # weather_feat3 = weather_feat3 + weather_feat2_shorcut
        
        # weather_feat4 = self.bn2(self.relu2(self.weather_fc2(weather_feat3)))
        # weather_feat3_shorcut = self.weather_shortcut2(weather_feat3)
        # # weather_feat : 3,4 residual
        # weather_feat4 = weather_feat4 + weather_feat3_shorcut
        # weather_result = self.weather_fc3(weather_feat4) 

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, None, self.weather_avg_pool(down4c)
                    
class Asymm_3d_spconv_clf_test(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv_clf_test, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")
        self.resBlock6 = ResBlock(16 * init_size, 32 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down6")   
        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        '''
        Weather Clf Net
        '''
        # init_size = 32
        in_channel = 64 * init_size
        # self.weather_logits = spconv.SubMConv3d(in_channel, int(in_channel/2), indice_key="weather_logit", kernel_size=3, stride=1, padding=1,bias=True)
        # self.sp_relu1 = nn.ReLU()
        # self.sp_bn1 = nn.BatchNorm1d(int(in_channel/2))
        self.weather_avg_pool = SparseGlobalAvgPool()
        # 256 -> 128
        self.weather_fc1 = nn.Linear(int(in_channel/2), int(in_channel/4))                
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(int(in_channel/4))
        # 128 -> 64
        self.weather_fc2 = nn.Linear(int(in_channel/4), int(in_channel/8))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(int(in_channel/8))
        # 128 -> 64
        self.weather_fc3 = nn.Linear(int(in_channel/8), 3)
        
        self.weight_initialization()
        
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down1c : m, 64
        down1c, down1b = self.resBlock2(ret)
        # down2c : m, 128
        down2c, down2b = self.resBlock3(down1c)
        # down3c : m, 256
        down3c, down3b = self.resBlock4(down2c)
        # down4c : m, 512
        down4c, down4b = self.resBlock5(down3c)
        # down5c : m, 1024
        down5c, _ = self.resBlock6(down4c)
        
        avg_feature = self.weather_avg_pool(down5c)
        weather_feat1 = self.weather_fc1(avg_feature)
        weather_feat2 = self.bn1(self.relu1(weather_feat1))
        
        weather_feat3 = self.bn2(self.relu2(self.weather_fc2(weather_feat2)))
        weather_result = self.weather_fc3(weather_feat3)
     
        
        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y, weather_result    