import os.path as osp
import cv2
import os
import json
from lanedet.utils.tusimple_metric import LaneEval
from .registry import DATASETS
import logging
import random
import numpy as np  
import math 

import torch 
from torch.utils.data import Dataset 
from .process import Process 
from lanedet.utils.visualization import imshow_lanes 
from mmcv.parallel import DataContainer as DC  

from PIL import Image 


SPLIT_FILES = {
    'trainval': ['scenario_1.txt', 'scenario_2.txt', 'scenario_3.txt', 'scenario_4.txt', 'scenario_5.txt', 'scenario_6.txt', 'scenario_7.txt', 'scenario_8.txt'], 
    'train': ['scenario_1.txt', 'scenario_2.txt', 'scenario_3.txt', 'scenario_4.txt', 'scenario_5.txt'],
    'val': ['scenario_6.txt', 'scenario_7.txt', 'scenario_8.txt'],
    'test': ['scenario_9.txt', 'scenario_10.txt'],
}

# SPLIT_FILES = {
#     'trainval': ['scenario_1.txt'], 
#     'train': ['scenario_1.txt'],
#     'val': ['scenario_6.txt'],
#     'test': ['scenario_9.txt'],
# }



@DATASETS.register_module
class DssDataset(Dataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        self.cfg = cfg 
        self.logger = logging.getLogger(__name__) 
        self.data_root = data_root 
        self.training = 'train' in split 
        self.process = Process(processes, cfg) 

        self.scenarios = SPLIT_FILES[split] 
        self.data_infos = self.load_annotations()
        # self.h_samples = list(range(160, 720, 10)) 


    def load_annotations(self): 
        self.logger.info('Loading DSS annotations...') 
        data_infos = [] 

        for scenario in self.scenarios: 
            img_paths_file = osp.join(self.data_root, scenario) 
            with open(img_paths_file, 'r') as f: 
                lines = f.readlines() 

            for line in lines: 
                img_path = line.strip() 

                data_infos.append({
                    'img_path': img_path, 
                    'img_name': img_path.split('/')[-1],
                    'mask_path': img_path.replace('images', 'seg_labels') 
                })

        if self.training: 
            random.shuffle(data_infos) 

        return data_infos 


    def __len__(self): 
        return len(self.data_infos) 
    
    def __getitem__(self, idx): 
        data_info = self.data_infos[idx] 
        if not osp.isfile(data_info['img_path']): 
            raise FileNotFoundError('cannot find file: {}'.format(data_info['img_path'])) 
        
        img = cv2.imread(data_info['img_path']) 
        img = img[self.cfg.cut_height:, :, :] 
        sample = data_info.copy() 
        sample.update({'img': img}) 

        if self.training: 
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED) 
            if len(label.shape) > 2: 
                label = label[:, :, 0] 

            label = label.squeeze() 
            label = label[self.cfg.cut_height:, :] 
            sample.update({'mask': label}) 

        sample = self.process(sample) 
        meta = {'full_img_path': data_info['img_path'], 
                'img_name': data_info['img_name']} 
        meta = DC(meta, cpu_only=True) 
        sample.update({'meta': meta})  

        return sample  
    
    
    def calc_dice(self, im1, im2): 
        if torch.is_tensor(im1): 
            im1 = im1.cpu().numpy() 
        if torch.is_tensor(im2): 
            im2 = im2.cpu().numpy() 

        im1 = np.asarray(im1).astype(np.bool_) 
        im2 = np.asarray(im2).astype(np.bool_) 

        if im1.shape != im2.shape: 
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.") 
        
        intersection = np.logical_and(im1, im2) 
        dice_score = 2. * intersection.sum() / (im1.sum() + im2.sum()) 

        return dice_score if not math.isnan(dice_score) else 0.0 
    

    def get_dice_score(self, preds, output_basedir): 
        dice_score = 0.0 
        count = 0 
        for pred in preds:

            if len(pred['pred_mask']) == 0: 
                count +=1 
                continue 

            pred_mask = torch.round(torch.sigmoid(pred['pred_mask']))  
            pred_img_path = pred['img_path'] 
            gt_mask = cv2.imread(pred_img_path.replace('images', 'seg_labels'), cv2.IMREAD_UNCHANGED)
            if len(gt_mask.shape) > 2: 
                gt_mask = gt_mask[:, :, 0] 
            gt_mask = gt_mask.squeeze()
            gt_mask = gt_mask[self.cfg.cut_height:, :]

            gt_mask = cv2.resize(gt_mask, (self.cfg.img_width, self.cfg.img_height), interpolation=cv2.INTER_NEAREST) 

            if count < 3: 
                pred_mask = pred_mask.cpu().numpy() 

                cv2.imwrite(f'{output_basedir}/img_{count}.png', cv2.imread(pred_img_path))
                cv2.imwrite(f'{output_basedir}/gt_mask_{count}.png', (gt_mask*255).astype(np.uint8))  
                cv2.imwrite(f'{output_basedir}/pred_mask1_{count}.png', (pred_mask[0]*255).astype(np.uint8)) 
                cv2.imwrite(f'{output_basedir}/pred_mask2_{count}.png', (pred_mask[1]*255).astype(np.uint8))  


            temp_dice = [] 
            for mask in pred_mask: 
                temp_dice.append(self.calc_dice(mask, gt_mask))
            dice_score += max(temp_dice)  
            count += 1 

        return dice_score / count 


    def evaluate(self, predictions, output_basedir, runtimes=None):
        dice_score = self.get_dice_score(predictions, output_basedir) 
        self.logger.info(dice_score)
        return dice_score 
