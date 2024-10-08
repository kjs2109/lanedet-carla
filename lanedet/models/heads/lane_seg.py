import torch
from torch import nn
import torch.nn.functional as F
from lanedet.core.lane import Lane, MyLane 
import cv2
import numpy as np

from ..registry import HEADS, build_head

@HEADS.register_module
class LaneSeg(nn.Module):
    def __init__(self, decoder, exist=None, thr=0.6,
            sample_y=None, cfg=None):
        super(LaneSeg, self).__init__()
        self.cfg = cfg
        self.thr = thr
        self.sample_y = sample_y

        self.decoder = build_head(decoder, cfg)
        self.exist = build_head(exist, cfg) if exist else None 

    def get_lanes(self, output, vis_mode): 
        segs = output['seg']
        segs = F.softmax(segs, dim=1)
        segs = segs.detach().cpu().numpy()
        if 'exist' in output:
            exists = output['exist']
            exists = exists.detach().cpu().numpy()
            exists = exists > 0.5
        else:
            exists = [None for _ in segs]

        ret = []
        for seg, exist in zip(segs, exists):
            if vis_mode == 'semantic':
                lanes = self.probmap2lane_seg(seg, exist)
            elif vis_mode == 'instance':
                lanes = self.probmap2lane(seg, exist)
            else: 
                raise ValueError("Invalid vis mode")
            ret.append(lanes)
        return ret

    def probmap2lane(self, probmaps, exists=None):
        lanes = []
        probmaps = probmaps[1:, ...]  # remove background
        if exists is None:
            exists = [True for _ in probmaps]
        for probmap, exist in zip(probmaps, exists):
            if exist == 0:
                continue
            probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)  # probmap.shape = (cfg.img_height, cfg.img_width)
            cut_height = self.cfg.cut_height
            ori_h = self.cfg.ori_img_h - cut_height
            coord = []
            for y in self.sample_y:
                proj_y = round((y - cut_height) * self.cfg.img_height/ori_h) 
                if proj_y < 0 or proj_y >= self.cfg.img_height:
                    continue
                line = probmap[proj_y]
                if np.max(line) < self.thr:
                    continue
                value = np.argmax(line)
                x = value*self.cfg.ori_img_w/self.cfg.img_width#-1.
                if x > 0:
                    coord.append([x, y])
            if len(coord) < 5:
                continue

            coord = np.array(coord)
            coord = np.flip(coord, axis=0)
            coord[:, 0] /= self.cfg.ori_img_w
            coord[:, 1] /= self.cfg.ori_img_h
            lanes.append(Lane(coord))
    
        return lanes 
    
    def check_point(self, check_index, curr_index):
        for index in check_index:
            if abs(index - curr_index) < 20:
                return False
        return True 
    
    def probmap2lane_seg(self, probmaps, exists=None):
        lanes = [] 
        probmaps = probmaps[1:, ...] 
        if exists is None: 
            exists = [True for _ in probmaps] 
        for probmap, exist in zip(probmaps, exists): 
            if exist == 0: 
                continue 
            probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)  # probmap.shape = (cfg.img_height, cfg.img_width)
            cut_height = self.cfg.cut_height
            ori_h = self.cfg.ori_img_h - cut_height
            coord = [] 
            for y in self.sample_y: 
                proj_y = round((y - cut_height) * self.cfg.img_height/ori_h) 
                if proj_y < 0 or proj_y >= self.cfg.img_height: 
                    continue 
                line = probmap[proj_y]  
                if np.max(line) < self.thr: 
                    continue 

                count = 0 
                check_index = [] 
                sorted_point_index = np.argsort(line)[::-1] 
                for curr_index in sorted_point_index: 
                    if (line[curr_index] > self.thr) and self.check_point(check_index, curr_index):  
                        x = curr_index * self.cfg.ori_img_w / self.cfg.img_width
                        if x > 0: 
                            check_index.append(curr_index)
                            coord.append([x, y])  
                            count += 1 
                    if count > 5: 
                        break 

            lanes.append(MyLane(coord))

        return lanes 


    def loss(self, output, batch):
        weights = torch.ones(self.cfg.num_classes)
        weights[0] = self.cfg.bg_weight
        weights = weights.cuda()
        criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label, weight=weights).cuda()
        criterion_exist = torch.nn.BCEWithLogitsLoss().cuda()
        loss = 0.
        loss_stats = {}
        seg_loss = criterion(F.log_softmax(output['seg'], dim=1), batch['mask'].long()) 
        loss += seg_loss
        loss_stats.update({'seg_loss': seg_loss})

        if 'exist' in output:
            exist_loss = 0.1 * \
                criterion_exist(output['exist'], batch['lane_exist'].float())
            loss += exist_loss
            loss_stats.update({'exist_loss': exist_loss})

        ret = {'loss': loss, 'loss_stats': loss_stats}
        return ret


    def forward(self, x, **kwargs):
        output = {}
        x = x[-1]
        output.update(self.decoder(x))
        if self.exist:
            output.update(self.exist(x))

        return output 
