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
import matplotlib.pyplot as plt

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
    log_save_path = train_hypers['log_save_path']
    os.makedirs(log_save_path, exist_ok=False)

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)
    clf_loss_func = torch.nn.CrossEntropyLoss()
    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    print("Model will save in : {}".format(model_save_path))
    
    '''
    Performace measure
    '''
    train_total_loss_list = []
    train_total_pcss_loss_list = []
    train_total_acc_loss_list = []
    train_total_miou_list = []
    train_total_acc_list = []
    
    valid_total_loss_list = []
    valid_total_pcss_loss_list = []
    valid_total_acc_loss_list = []
    valid_total_miou_list = []
    valid_total_acc_list = []
    
    while epoch < train_hypers['max_num_epochs']:
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(1)
        
        train_hist_list = []
        train_loss_list = []
        train_pcss_loss_list = []
        train_acc_loss_list = []
        train_weather_pred_list = []
        train_weather_gt_list = []
        for i_iter, (_, train_vox_label, train_grid, train_pt_labs, train_pt_fea, weather_gt) in enumerate(train_dataset_loader):
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
            
            pcss_loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=0) + loss_func(outputs, point_label_tensor) 
            weather_loss = clf_loss_func(weathers, weather_gt)
            loss = pcss_loss + weather_loss
            
            predict_outputs = torch.argmax(outputs, dim=1)
            predict_outputs = predict_outputs.cpu().detach().numpy()
            
            train_pcss_loss_list.append(pcss_loss.item())
            train_acc_loss_list.append(weather_loss.item())
            train_loss_list.append(loss.item())
            
            # train_weather_pred_list.append(max_weathers.item())
            # train_weather_gt_list.append(weather_gt.item())
            train_weather_pred_list += list(max_weathers.detach().cpu().numpy())
            train_weather_gt_list += list(weather_gt.detach().cpu().numpy())
            print("===================")
            print("pcss loss : ", pcss_loss)
            print("weather prob : ", softmax_weathers)
            print(max_weathers, weather_gt, weather_loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            
            for count, i_train_grid in enumerate(train_grid):
                train_hist_list.append(fast_hist_crop(predict_outputs[
                                                count, train_grid[count][:, 0], train_grid[count][:, 1],
                                                train_grid[count][:, 2]], train_pt_labs[count],
                                            unique_label))
                    
            if global_iter % 1000 == 0:
                if len(train_loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(train_loss_list)))
                else:
                    print('loss error')

            if global_iter % check_iter == 0:
                if len(train_loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(train_loss_list)))
                else:
                    print('loss error')
        
        
        '''
        # validation Phase
        '''
        my_model.eval()
        valid_hist_list = []
        valid_loss_list = []
        valid_pcss_loss_list = []
        valid_acc_loss_list = []
        valid_weather_pred_list = []
        valid_weather_gt_list = []
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
                pcss_loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels), val_label_tensor, ignore=0) + loss_func(predict_labels, val_label_tensor) 
                weather_loss = clf_loss_func(predict_weathers, weather_gt)
                loss = pcss_loss + weather_loss
                
                predict_labels = torch.argmax(predict_labels, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()

                valid_weather_pred_list.append(max_weathers.item())
                valid_weather_gt_list.append(weather_gt.item())
                print("===================")
                print(max_weathers, weather_gt, weather_loss.item())
                
                valid_pcss_loss_list.append(pcss_loss.item())
                valid_acc_loss_list.append(weather_loss.item())
                valid_loss_list.append(loss.item())
                for count, i_val_grid in enumerate(val_grid):
                    valid_hist_list.append(fast_hist_crop(predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],val_grid[count][:, 2]], val_pt_labs[count], unique_label))
                valid_loss_list.append(loss.detach().cpu().numpy())
        '''
        After one epoch
        '''
        my_model.train()
        # save train weather accuracy
        train_weather_pred_arr = np.array(train_weather_pred_list)
        train_weather_gt_arr = np.array(train_weather_gt_list)
        correct = np.sum(train_weather_pred_arr == train_weather_gt_arr)
        total = len(train_weather_gt_arr)
        accuracy = correct / total
        
        train_total_acc_list.append(accuracy)
        msg = "train weather accuracy: {}".format(accuracy)
        save_to_log(log_save_path, "log.txt", msg)
        
        # save valid weather accuracy
        valid_weather_pred_arr = np.array(valid_weather_pred_list)
        valid_weather_gt_arr = np.array(valid_weather_gt_list)
        correct = np.sum(valid_weather_pred_arr == valid_weather_gt_arr)
        total = len(valid_weather_gt_arr)
        accuracy = correct / total
        
        valid_total_acc_list.append(accuracy)
        msg = "valid weather accuracy: {}".format(accuracy) 
        save_to_log(log_save_path, "log.txt", msg)
        
        # save train mIOU
        iou = per_class_iu(sum(train_hist_list))
        msg = 'Training per class iou: '
        save_to_log(log_save_path, "log.txt", msg)
        for class_name, class_iou in zip(unique_label_str, iou):
            msg = '%s : %.2f%%' % (class_name, class_iou * 100)
            save_to_log(log_save_path, "log.txt", msg)
        train_miou = np.nanmean(iou) * 100
        train_total_miou_list.append(train_miou)
        
        # save valid mIOU
        iou = per_class_iu(sum(valid_hist_list))
        msg = 'Validation per class iou: '
        save_to_log(log_save_path, "log.txt", msg)
        for class_name, class_iou in zip(unique_label_str, iou):
            msg = '%s : %.2f%%' % (class_name, class_iou * 100)
            save_to_log(log_save_path, "log.txt", msg)
        val_miou = np.nanmean(iou) * 100
        valid_total_miou_list.append(val_miou)
        
        train_total_loss_list.append(np.mean(train_loss_list))
        train_total_pcss_loss_list.append(np.mean(train_pcss_loss_list))
        train_total_acc_loss_list.append(np.mean(train_acc_loss_list))
        
        valid_total_loss_list.append(np.mean(valid_loss_list))
        valid_total_pcss_loss_list.append(np.mean(valid_pcss_loss_list))
        valid_total_acc_loss_list.append(np.mean(valid_acc_loss_list))
        
        # save model if performance is improved
        if best_val_miou < val_miou:
            best_val_miou = val_miou
            torch.save(my_model.state_dict(), model_save_path)

        msg = 'Current val miou is %.3f while the best val miou is %.3f' % (val_miou, best_val_miou)
        save_to_log(log_save_path, "log.txt", msg)
        msg = 'Current val loss is %.3f' %(np.mean(valid_loss_list))
        save_to_log(log_save_path, "log.txt", msg)
        
        pbar.close()
        epoch += 1
        
        del val_vox_label, val_grid, val_pt_fea, val_grid_ten
    '''
    After end of training
    '''
    
    # draw total loss (train vs valid)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Total loss")
    plt.plot(train_total_loss_list, label='train')
    plt.plot(valid_total_loss_list, label='valid')
    
    plt.subplot(1, 3, 2)
    plt.title("PCSS loss")
    plt.plot(train_total_pcss_loss_list, label='train')
    plt.plot(valid_total_pcss_loss_list, label='valid')
    
    plt.subplot(1, 3, 3)
    plt.title("Weather loss")
    plt.plot(train_total_acc_loss_list, label='train')
    plt.plot(valid_total_acc_loss_list, label='valid')
    plt.ylim(0, 2)    
    
    plt.legend()
    plt.savefig(log_save_path + "/loss.png")
    
    # draw performance (train vs valid)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("mIOU")
    plt.plot(train_total_miou_list, label='train')
    plt.plot(valid_total_miou_list, label='valid')
    
    plt.subplot(1, 2, 2)
    plt.title("Weather accuracy")
    plt.plot(train_total_acc_list, label='train')
    plt.plot(valid_total_acc_list, label='valid')
    
    plt.legend()
    plt.savefig(log_save_path + "/performance.png")
    
    
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
