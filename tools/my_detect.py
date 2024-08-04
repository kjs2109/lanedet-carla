import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm
import math 

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(self.net, device_ids = range(1)).cuda() 
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32) # 이미지의 윗부분을 잘라내고, float32로 변환
        data = {'img': img, 'lanes': []} 
        data = self.processes(data) 
        data['img'] = data['img'].unsqueeze(0)  
        data.update({'img_path':img_path, 'ori_img':ori_img})
        return data

    def inference(self, data):
        # print('img shape ', data['img'].shape)
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir 
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data 
    
    def vis_inference(self, ori_img, cfg): 
        # print(ori_img.shape)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32) 
        data = {'img': img, 'lanes': []}
        data = self.processes(data) 
        data['img'] = data['img'].unsqueeze(0)
        # data.update({'ori_img':ori_img})
        data['lanes'] = self.inference(data)[0] 
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        for lane in lanes: 
            for x, y in lane: 
                if x <= 0 or y <= 0: 
                    continue 
                x, y = int(x), int(y) 
                cv2.circle(ori_img, (x, y), 4, (255, 0, 0), 2) 

        result_img = ori_img
        return result_img  
    
        
    def vis_inference_with_line(self, ori_img, cfg): 
        # print(ori_img.shape)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32) 
        data = {'img': img, 'lanes': []}
        data = self.processes(data) 
        data['img'] = data['img'].unsqueeze(0)
        # data.update({'ori_img':ori_img})
        data['lanes'] = self.inference(data)[0] 
        lanes = [np.array(lane.to_array(self.cfg), dtype=np.int32) for lane in data['lanes']] 
        filtered_lanes = []  
        for lane in lanes: 
            filtered_lane = [] 
            prev_point = lane[0] 
            for curr_point in lane[1:]: 
                
                if (curr_point[0] - prev_point[0]) == 0: 
                    continue  
                
                a = (curr_point[1] - prev_point[1]) / (curr_point[0] - prev_point[0])

                if abs(a) < 0.6: 
                    continue 

                filtered_lane.append(curr_point)

                prev_point = curr_point  
            filtered_lanes.append(np.array(filtered_lane, dtype=np.int32))


        cv2.polylines(ori_img, filtered_lanes, isClosed=False, color=(0, 255, 0), thickness=2)
        # for lane in lanes:
        #     cv2.polylines(ori_img, [lane], isClosed=False, color=(0, 255, 0), thickness=2) 

        result_img = ori_img
        return result_img 


def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def process(args):
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        detect.run(p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='The path of config file')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)
