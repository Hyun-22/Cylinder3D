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

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
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

def main(args):
    pytorch_device = torch.device('cuda:0')
    cmap = color_map()
    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']


    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']
    train_dataloader_config = configs['train_data_loader']
    model_load_path = train_hypers['model_load_path']
    log_save_path = train_hypers['log_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        print("load model from : {}".format(model_load_path))
        my_model = load_checkpoint(model_load_path, my_model)
    else:
        print("model not exist!!!")

    my_model.to(pytorch_device)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)
    # validation
    my_model.eval()
    hist_list = []
    val_loss_list = []
    weather_pred_list = []
    weather_gt_list = []
    # visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # validation
    with torch.no_grad():
        a = 0
        for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, weather_gt) in enumerate(
                val_dataset_loader):

            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)
            val_batch_size = val_vox_label.shape[0]
            weather_gt = torch.tensor(weather_gt).to(pytorch_device).long()
            s = time.time()
            predict_labels, predict_weathers = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
            e = time.time()

            print("elapsed time : {:.4f}".format(e-s))

            softmax_weathers = torch.softmax(predict_weathers, dim=1)
            max_weathers = torch.argmax(softmax_weathers, dim=1)

            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()

            weather_pred_list.append(max_weathers.item())
            weather_gt_list.append(weather_gt.item())
            # print("===================")
            for count, i_val_grid in enumerate(val_grid):
                point_pred = predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]]
                color_pred = [cmap[x] for x in point_pred]
                color_pred = np.array(color_pred)/255
                points_xy = val_pt_fea[count][:,-3:-1]
                points_z = val_pt_fea[count][:, -4]
                points_z = np.expand_dims(points_z, -1)
                points = np.concatenate((points_xy, points_z), axis = 1)
                # bev_points = points[:, :2]

                text = o3d.geometry.Text3D()
                text.text = "Hello"  # 원하는 텍스트 입력
                text.position = [0, 0, 0]  # 텍스트 위치 (좌상단에 배치하려면 [0, 0, 0] 사용)
                
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points)
                point_cloud.colors = o3d.utility.Vector3dVector(color_pred)
                vis.clear_geometries()
                vis.add_geometry(point_cloud)

                # ViewControl 객체 가져오기
                view_control = vis.get_view_control()

                # 관점 
                # view_control.set_up((0.5, 1, 1))  # set the negative direction of the y-axis as the up direction
                # view_control.set_front((0.5, 0.5, 0.5))  # set the positive direction of the x-axis toward you
                # view_control.set_lookat((a, 0, 0))  # set the original point as the center point of the window
                
                # view_control.set_up((1, 0.3, 0))  # set the positive direction of the x-axis as the up direction
                # view_control.set_up((0, 0.1, 0))  # set the negative direction of the y-axis as the up direction
                view_control.set_front((0.49342297000636554, -0.56786074454406688, 0.65883833182045159))  # set the positive direction of the x-axis toward you
                view_control.set_lookat((0.84637521845988761, -3.5969191823325253, -2.7728844874157144))  # set the original point as the center point of the window
                view_control.set_up((-0.38995242420475884, 0.53265155242439555, 0.75114541239144472))
                view_control.scale(-30)


                # Update the visualization
                vis.update_geometry(point_cloud)
                vis.poll_events()
                vis.update_renderer()

    weather_pred_arr = np.array(weather_pred_list)
    weather_gt_arr = np.array(weather_gt_list)
    correct = np.sum(weather_pred_arr == weather_gt_arr)
    total = len(weather_gt_arr)
    accuracy = correct / total

    msg = "weather accuracy: {}".format(accuracy) 
    save_to_log(log_save_path, "valid_log.txt", msg)
    
    iou = per_class_iu(sum(hist_list))
    msg = 'Validation per class iou: '
    save_to_log(log_save_path, "valid_log.txt", msg)
    for class_name, class_iou in zip(unique_label_str, iou):
        msg = '%s : %.2f%%' % (class_name, class_iou * 100)
        save_to_log(log_save_path, "valid_log.txt", msg)
    val_miou = np.nanmean(iou) * 100
    del val_vox_label, val_grid, val_pt_fea, val_grid_ten

    msg = 'Current val miou is %.3f' % (val_miou)
    save_to_log(log_save_path, "valid_log.txt", msg)
    msg = 'Current val loss is %.3f' %(np.mean(val_loss_list))
    save_to_log(log_save_path, "valid_log.txt", msg)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
