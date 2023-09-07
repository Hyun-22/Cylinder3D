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

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


warnings.filterwarnings("ignore")

def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return

def draw_confusion_matrix(target_list, predicted_list, logdir, suffix="", num_classes = 3, class_names = ["Clear", "Moderate", "Heavy"]):
    font_size = [10, 10, 8] # conf, axis, tick
    conf = confusion_matrix(target_list, predicted_list, labels=range(num_classes), normalize='true')
    cm_percentage = conf * 100  # convert to percentage
    fig, ax = plt.subplots()  # Create figure and axes
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage,
                                    display_labels=class_names)
    disp = disp.plot(cmap=plt.cm.Blues, values_format=".1f", ax=ax, xticks_rotation="vertical")
    disp.figure_.set_dpi(200)
    disp.figure_.set_tight_layout(True)
    # Adjust font size inside confusion matrix
    for text_row in disp.text_:
        for text in text_row:  # Loop over text instances
            text.set_fontsize(font_size[0])  # You can adjust this value as per your requirement
    # Adjust font size for x and y labels
    ax.xaxis.label.set_fontsize(font_size[1])
    ax.yaxis.label.set_fontsize(font_size[1])
    # Adjust font size for x and y ticks
    ax.tick_params(axis='x', labelsize=font_size[2])  # adjust as necessary
    ax.tick_params(axis='y', labelsize=font_size[2])  # adjust as necessary
    plt.savefig(logdir+"/confusion_matrix_{}.png".format(suffix))
    plt.close('all')


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

    # validation
    with torch.no_grad():
        a = 0
        for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, weather_gt) in enumerate(tqdm(
                val_dataset_loader)):

            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)
            val_batch_size = val_vox_label.shape[0]
            weather_gt = torch.tensor(weather_gt).to(pytorch_device).long()
            s = time.time()
            predict_labels, predict_weathers = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
            e = time.time()

            # print("elapsed time : {:.4f}".format(e-s))

            softmax_weathers = torch.softmax(predict_weathers, dim=1)
            max_weathers = torch.argmax(softmax_weathers, dim=1)

            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()

            weather_pred_list.append(max_weathers.item())
            weather_gt_list.append(weather_gt.item())

    weather_pred_arr = np.array(weather_pred_list)
    weather_gt_arr = np.array(weather_gt_list)
    correct = np.sum(weather_pred_arr == weather_gt_arr)
    total = len(weather_gt_arr)
    accuracy = correct / total
    draw_confusion_matrix(weather_gt_arr, weather_pred_arr, log_save_path, suffix="")
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
