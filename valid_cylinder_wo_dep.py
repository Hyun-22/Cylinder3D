# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


from importlib.resources import path
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

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
    # print(sig.min(), sig.max())
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
    
# def text_3d(text, pos, direction=None, degree=0.0, font='DejaVu Sans Mono for Powerline.ttf', font_size=16):
def text_3d(text, pos, direction=None, degree=0.0, density=100000, font='/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', font_size=5):    
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    # pcd.points = o3d.utility.Vector3dVector(indices / 100.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd

def main(args):
    pytorch_device = torch.device('cuda:2')
    cmap = color_map()
    config_path = args.config_path
    configs = load_config_data(config_path)
    dataset_config = configs['dataset_params']
    val_dataloader_config = configs['val_data_loader']
    val_batch_size = val_dataloader_config['batch_size']
    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    train_dataloader_config = configs['train_data_loader']
    model_load_path = train_hypers['model_load_path']
    # model_load_path = train_hypers['model_load_path'].replace("load.pt", "save_partial_test.pt")
    log_path = train_hypers['log_save_path']
    
    result_dir = os.path.join(log_path, "result_urban")
    os.makedirs(result_dir, exist_ok=True)

    data_paths = glob(os.path.join(args.data_path, "*.bin"))
    data_paths.sort()

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        print("load model from : {}".format(model_load_path))
        my_model = load_checkpoint(model_load_path, my_model)
    else:
        print("model not exist!!!")

    my_model.to(pytorch_device)

    # train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
    #                                                               train_dataloader_config,
    #                                                               val_dataloader_config,
    #                                                               grid_size=grid_size)
    # validation
    my_model.eval()
    weather_pred_list = []
    weather_gt_list = []
    # visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    weather_status = {0:"sunny", 1:"Moderate", 2:"Heavy"}
    total_weather_pred = np.zeros((3,))
    
    # validation
    with torch.no_grad():
        for path_idx, data_path in enumerate(data_paths):
            # if path_idx % 50 == 14 and path_idx > 2700 :
            # if True:
            if path_idx > 700:
                file_name = data_path.split(os.sep)[-1]
                dst_path = os.path.join(result_dir, file_name)
                dst_path = dst_path.replace(".bin", ".pcd")
                # print("data_path : ",dst_path)
                total_s = time.time()
                # load lidar data
                voxel_position, val_grid, val_pt_fea = lidar_preprocessing(data_path, dataset_config["fixed_volume_space"], dataset_config["max_volume_space"], dataset_config["min_volume_space"], np.array(grid_size))
                # load label if exist
                if args.label_exist:
                    label_path = data_path.replace("velodyne", "labels").replace("bin", "label")
                    label = load_label(label_path)
                                
                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]

                # weather_gt = torch.tensor(weather_gt).to(pytorch_device).long()
                s = time.time()
                predict_labels, predict_weathers = my_model(val_pt_fea_ten, val_grid_ten, 1)
                e = time.time()

                # print("inference time : {:.4f}".format(e-s))
                
                softmax_weathers = torch.softmax(predict_weathers, dim=1)
                max_weathers = torch.argmax(softmax_weathers, dim=1)
                # print("probablity : ", softmax_weathers)
                print(weather_status[max_weathers[0].item()])

                predict_labels = torch.argmax(predict_labels, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()

                weather_pred_list.append(max_weathers.item())
                # weather_gt_list.append(weather_gt.item())
                # print("===================")
                
                point_pred = predict_labels[0, val_grid[0][:, 0], val_grid[0][:, 1],val_grid[0][:, 2]]
                color_pred = [cmap[x] for x in point_pred]
                color_pred = np.array(color_pred)/255
                points_xy = val_pt_fea[0][:,-3:-1]
                points_z = val_pt_fea[0][:, -4]
                points_z = np.expand_dims(points_z, -1)
                points = np.concatenate((points_xy, points_z), axis = 1)
                total_e = time.time()
                '''
                remove specific class
                '''
                # for remove_cls in [9,10,11,17,20]:
                #     points = points[(point_pred != remove_cls)]
                #     point_pred = point_pred[(point_pred != remove_cls)]

                color_pred = [cmap[x] for x in point_pred]
                color_pred = np.array(color_pred)/255
                # bev_points = points[:, :2]
                
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points)
                point_cloud.colors = o3d.utility.Vector3dVector(color_pred)

                # o3d.visualization.gui.Label3D(weather_status[max_weathers[0].item()], np.array((10,0,0)))
                # text = text_3d(weather_status[max_weathers[0].item()], np.array((1,0,0)))

                total_weather_pred[max_weathers[0].item()] +=1
                # print(weather_status[max_weathers[0].item()])
                # print(max_weathers[0].item())
                # print(total_weather_pred)
                # o3d.visualization.draw_geometries([point_cloud, text])
                # o3d.visualization.draw_geometries([text])
                # o3d.io.write_point_cloud(dst_path, point_cloud)
                vis.clear_geometries()
                vis.add_geometry(point_cloud)

                # ViewControl Setting
                view_control = vis.get_view_control()
                view_control.set_front((0.49342297000636554, -0.56786074454406688, 0.65883833182045159))  # set the positive direction of the x-axis toward you
                view_control.set_lookat((0.84637521845988761, -3.5969191823325253, -2.7728844874157144))  # set the original point as the center point of the window
                view_control.set_up((-0.38995242420475884, 0.53265155242439555, 0.75114541239144472))
                view_control.scale(-32)

                # Update the visualization
                vis.update_geometry(point_cloud)
                vis.poll_events()
                vis.update_renderer()

                

                # print("total inference time : {:.4f}".format(total_e-total_s))
            else:
                continue
    print(total_weather_pred)
    x = range(1, len(weather_pred_list) + 1)  # x 축은 인덱스로 설정합니다.

    plt.plot(x, weather_pred_list)
    plt.xlabel('time')
    plt.ylabel('pred_value')
    plt.savefig(log_path+"/valid_result.png")
    plt.close('all')
    # plt.hist(total_weather_pred)
    # plt.show()


    del val_grid, val_pt_fea, val_grid_ten


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    parser.add_argument('-d', '--data_path', default="./")
    parser.add_argument('-l', '--label_exist', default=False)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
