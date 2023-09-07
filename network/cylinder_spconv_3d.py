# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py

from torch import nn
import torch
import torch.nn.functional as F
REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class cylinder_asym(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        self.weather_clf = test_clf(num_class = 4)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result

@register_model
class cylinder_asym_ours(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        self.weather_clf = test_clf(num_class = 3)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result
@register_model
class cylinder_asym_ours_test(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        # self.weather_clf = test_clf(num_class = 3)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features, weather_result = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        # weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result

@register_model
class cylinder_asym_clf_v1(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        # self.weather_clf = test_clf(num_class = 3)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features, weather_result = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        # weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result

@register_model
class cylinder_asym_clf_v2(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        # self.weather_clf = test_clf(num_class = 3)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features, weather_result = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        # weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result    

@register_model
class cylinder_asym_clf_v3(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        # self.weather_clf = test_clf(num_class = 3)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features, weather_result = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        # weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result  
@register_model
class cylinder_asym_clf_v4(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        # self.weather_clf = test_clf(num_class = 3)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features, weather_result = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        # weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result 
@register_model
class cylinder_asym_clf_v5(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        # self.weather_clf = test_clf(num_class = 3)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features, weather_result = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        # weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result 

@register_model
class cylinder_asym_clf_v6(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        # self.weather_clf = test_clf(num_class = 3)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features, weather_result = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        # weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result 

@register_model
class cylinder_asym_clf_v7(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        # self.weather_clf = test_clf(num_class = 3)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features, weather_result = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        # weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result 

@register_model
class cylinder_asym_clf_v8(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        # self.weather_clf = test_clf(num_class = 3)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features, weather_result = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        # weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result         
    
@register_model
class cylinder_asym_clf_test(nn.Module):
    '''
    network for lisa dataset
    '''
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv
        # self.weather_clf = test_clf(num_class = 3)
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        coords, features_3d, pts_feature = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)
        spatial_features, weather_result = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        # weather_result = self.weather_clf(pts_feature, coords)
        return spatial_features, weather_result 
    
class test_clf(nn.Module):
    def __init__(self, num_class = 4):
        super(test_clf, self).__init__()
        # clear / 30mm / 20mm / 10mm
        self.num_class = num_class
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

    def forward(self, in_x, coords):
        _, all_cnt = torch.unique(coords[:, 0], return_counts = True)
        batch_size = coords[:, 0].max() + 1
        prev_cnt = 0
        ret = torch.zeros(batch_size, self.num_class).to(in_x.device)
        for idx, cnt in enumerate(all_cnt):
            slice_x = in_x[prev_cnt:prev_cnt+cnt]
            # mean_x = torch.mean(slice_x, dim=0)
            # x = F.relu(self.fc1(mean_x))
            max_x, _ = torch.max(slice_x, dim=0)
            x = F.relu(self.fc1(max_x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            # x = F.log_softmax(x, dim=1)
            ret[idx] = x
            prev_cnt += cnt
        return ret