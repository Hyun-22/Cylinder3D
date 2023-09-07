# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from glob import glob

from builder import data_builder, model_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint
import open3d as o3d
import warnings

warnings.filterwarnings("ignore")

def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return

def color_map():
    cmap = {0: [0, 0, 0],
    1: [100, 150, 245],
    2: [100, 230, 245],
    3: [30, 60, 150],
    4: [80, 30, 180],
    5: [0, 0, 255],
    6: [255, 30, 30],
    7: [255, 40, 200],
    8: [150, 30, 90],
    9: [255, 0, 255],
    10: [255, 150, 255],
    11: [75, 0, 75],
    12: [175, 0, 75],
    13: [255, 200, 0],
    14: [255, 120, 50],
    15: [0, 175, 0],
    16: [135, 60, 0],
    17: [150, 240, 80],
    18: [255, 240, 150],
    19: [255, 0, 0],
    20: [0, 0, 255],}
    return cmap

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)

def lidar_preprocessing(lidar_path, fixed_volume_space, max_volume_space, min_volume_space, grid_size):
    pcd_data = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    xyz = pcd_data[:, 0:3]
    sig = pcd_data[:, 3]    
    # roi slicing
    xyz = xyz[np.where((xyz[:, -1] < 2) & (xyz[:, -1] > -4))]
    sig = sig[np.where((xyz[:, -1] < 2) & (xyz[:, -1] > -4))]
    xyz_pol = cart2polar(xyz)

    max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
    min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
    max_bound = np.max(xyz_pol[:, 1:], axis=0)
    min_bound = np.min(xyz_pol[:, 1:], axis=0)
    max_bound = np.concatenate(([max_bound_r], max_bound))
    min_bound = np.concatenate(([min_bound_r], min_bound))
    if fixed_volume_space:
        max_bound = np.asarray(max_volume_space)
        min_bound = np.asarray(min_volume_space)
    # get grid index
    crop_range = max_bound - min_bound
    cur_grid_size = grid_size
    intervals = crop_range / (cur_grid_size - 1)

    if (intervals == 0).any(): print("Zero interval!")
    grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int64)

    voxel_position = np.zeros(grid_size, dtype=np.float32)
    dim_array = np.ones(len(grid_size) + 1, int)
    dim_array[0] = -1
    voxel_position = np.indices(grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
    voxel_position = polar2cat(voxel_position)

    # center data on each voxel for PTnet
    voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
    return_xyz = xyz_pol - voxel_centers
    return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)
    
    return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)
    
    grid_ind = np.expand_dims(grid_ind, axis=0)
    return_fea = np.expand_dims(return_fea, axis=0) 
    return voxel_position, grid_ind, return_fea
    
def load_label(label_path):
    label = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
    label = label & 0xFFFF  # delete high 16 digits binary
    
    return label
def load_pcd(pcd_path):
    pcd_data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    return pcd_data

def main(args):
    cmap = color_map()
    data_paths = glob(os.path.join(args.data_path, "*.bin"))
    data_paths.sort()


    # visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # validation
    for path_idx, data_path in enumerate(data_paths):
        '''
        Load data
        '''
        # load lidar data        
        points = load_pcd(data_path)
        # load label if exist
        if args.label_exist:
            label_path = data_path.replace("velodyne", "labels").replace("bin", "label")
            label = load_label(label_path)

        '''
        Visualization
        '''
        color_pred = [cmap[x] for x in label]
        color_pred = np.array(color_pred)/255
        # bev_points = points[:, :2]
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(color_pred)
        vis.clear_geometries()
        vis.add_geometry(point_cloud)

        # ViewControl Setting
        view_control = vis.get_view_control()
        view_control.set_front((0.49342297000636554, -0.56786074454406688, 0.65883833182045159))  # set the positive direction of the x-axis toward you
        view_control.set_lookat((0.84637521845988761, -3.5969191823325253, -2.7728844874157144))  # set the original point as the center point of the window
        view_control.set_up((-0.38995242420475884, 0.53265155242439555, 0.75114541239144472))
        view_control.scale(-30)

        # Update the visualization
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_path', default="./")
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
