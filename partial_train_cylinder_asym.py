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

import warnings

warnings.filterwarnings("ignore")

def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return

def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']
    
    model_save_path = model_save_path.replace(".pt", "_partial_test.pt")
    log_save_path = train_hypers['log_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        print("load data from {}".format(model_load_path))
        my_model = load_checkpoint(model_load_path, my_model)
    
    # add new weather clf module
    for name, param in my_model.named_parameters():
        if "weather" in name:
            print("{} will be trained".format(name))
            param.requires_grad = True
        else:
            param.requires_grad = False
        # print("name : ", name)
        # print("requires_grad : ", param.requires_grad)
        # print()
        
    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)
    clf_loss_func = torch.nn.CrossEntropyLoss()
    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # remove model weather clf parameters
    # del my_model.weather_clf
    
    # freeze pretrained model

        
    # add new weather clf module
    
    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    print("Model will save in : {}".format(model_save_path))
    
    
    
    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(3)
        # lr_scheduler.step(epoch)
        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea, weather_gt) in enumerate(train_dataset_loader):
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)
            train_batch_size = train_vox_label.shape[0]
            weather_gt = torch.tensor(weather_gt).to(pytorch_device).long()
            # forward + backward + optimize
            outputs, weathers = my_model(train_pt_fea_ten, train_vox_ten, train_batch_size)
            
            softmax_weathers = torch.softmax(weathers, dim=1)
            max_weathers = torch.argmax(softmax_weathers, dim=1)
            
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=0) + loss_func(outputs, point_label_tensor) + clf_loss_func(weathers, weather_gt)
            print("===================")
            print(max_weathers, weather_gt, clf_loss_func(weathers, weather_gt))
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')
        # validation
        my_model.eval()
        hist_list = []
        val_loss_list = []
        weather_pred_list = []
        weather_gt_list = []
        # validation
        with torch.no_grad():
            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, weather_gt) in enumerate(
                    val_dataset_loader):

                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                    val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)
                val_batch_size = val_vox_label.shape[0]
                weather_gt = torch.tensor(weather_gt).to(pytorch_device).long()
                
                predict_labels, predict_weathers = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                
                softmax_weathers = torch.softmax(predict_weathers, dim=1)
                max_weathers = torch.argmax(softmax_weathers, dim=1)

                # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                        ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
                predict_labels = torch.argmax(predict_labels, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()

                weather_pred_list.append(max_weathers.item())
                weather_gt_list.append(weather_gt.item())
                print("===================")
                print(max_weathers, weather_gt, clf_loss_func(predict_weathers, weather_gt))
                for count, i_val_grid in enumerate(val_grid):
                    hist_list.append(fast_hist_crop(predict_labels[
                                                        count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                        val_grid[count][:, 2]], val_pt_labs[count],
                                                    unique_label))
                val_loss_list.append(loss.detach().cpu().numpy())
        my_model.train()
        weather_pred_arr = np.array(weather_pred_list)
        weather_gt_arr = np.array(weather_gt_list)
        correct = np.sum(weather_pred_arr == weather_gt_arr)
        total = len(weather_gt_arr)
        accuracy = correct / total

        msg = "weather accuracy: {}".format(accuracy) 
        save_to_log(log_save_path, "log_partial.txt", msg)
        
        iou = per_class_iu(sum(hist_list))
        msg = 'Validation per class iou: '
        save_to_log(log_save_path, "log_partial.txt", msg)
        for class_name, class_iou in zip(unique_label_str, iou):
            msg = '%s : %.2f%%' % (class_name, class_iou * 100)
            save_to_log(log_save_path, "log_partial.txt", msg)
        val_miou = np.nanmean(iou) * 100
        del val_vox_label, val_grid, val_pt_fea, val_grid_ten

        # save model if performance is improved
        if best_val_miou < val_miou:
            best_val_miou = val_miou
            torch.save(my_model.state_dict(), model_save_path)

        msg = 'Current val miou is %.3f while the best val miou is %.3f' % (val_miou, best_val_miou)
        save_to_log(log_save_path, "log_partial.txt", msg)
        msg = 'Current val loss is %.3f' %(np.mean(val_loss_list))
        save_to_log(log_save_path, "log_partial.txt", msg)
        
        pbar.close()
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
