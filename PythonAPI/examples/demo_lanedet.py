import carla 
import math 
import random 
import time  
import numpy as np 
import cv2   
import argparse

import os 
import cv2 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from collections import deque

from tools.my_detect import Detect 
from lanedet.utils.config import Config  


class ImageBuffer:
    def __init__(self, size=10):
        self.buffer = deque(maxlen=size)
    
    def add(self, image):
        self.buffer.append(image)
    
    def get(self):
        if len(self.buffer) > 0:
            return self.buffer[-1]
        else:
            return None


def camera_callback(image, data_dict, image_buffer, carla_detect, cfg, vis_type): 
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))[:, :, :3] 
    img = np.ascontiguousarray(img)

    image_buffer.add(img) 
    buffered_img = image_buffer.get() 

    if buffered_img is not None:
        if vis_type == 'circle': 
            result_img = carla_detect.vis_inference(img, cfg)  
        elif vis_type == 'line':
            result_img = carla_detect.vis_inference_with_line(img, cfg) 
        else: 
            raise ValueError() 
        data_dict['image'] = result_img  
    else:
        data_dict['image'] = img


def simulate(args): 

    random.seed(args.seed) 

    # lanedet model setting 
    config_file = args.config

    cfg = Config.fromfile(config_file)
    cfg.show = False 
    cfg.savedir = '/root/works/lanedet'
    cfg.load_from = args.load_from
    carla_detector = Detect(cfg)

    # carla setting 
    client = carla.Client(args.host, args.port) 
    world = client.get_world()   # client.load_world('Town01')

    bp_lib = world.get_blueprint_library() 
    spawn_points = world.get_map().get_spawn_points() 

    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[args.spawn_point])  # vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points)) 

    spectator = world.get_spectator() 
    transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)), vehicle.get_transform().rotation) 
    spectator.set_transform(transform)  

    camera_bp = bp_lib.find('sensor.camera.rgb') 
    camera_init_trans = carla.Transform(carla.Location(x=0.6, z=1.6))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle) 


    image_buffer = ImageBuffer(size=10) 

    image_w = camera_bp.get_attribute("image_size_x").as_int()  
    image_h = camera_bp.get_attribute("image_size_y").as_int() 

    camera_data = {"image": np.zeros((image_w, image_h, 3))} 

    camera.listen(lambda image: camera_callback(image, camera_data, image_buffer, carla_detector, cfg, args.vis_type)) 

    vehicle.set_autopilot(True) 


    cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE) 
    cv2.imshow('RGB Camera', camera_data['image']) 
    cv2.waitKey(1) 

    while True: 
        cv2.imshow('RGB Camera', camera_data['image']) 

        if cv2.waitKey(1) == ord('q'): 
            break

    cv2.destroyAllWindows() 




if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default="/root/lanedet-carla/demo/configs/carla_scnn_tusimple.py")
    parser.add_argument('--load_from', default='/root/lanedet-carla/demo/checkpoints/scnn_r18_tusimple.pth') 
    parser.add_argument('--vis_type', type=str, default='circle', help='vis_type ["circle", "line"]')
    parser.add_argument('--spawn_point', type=int, default=1, help='spawn point index')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='carla server IP') 
    parser.add_argument('--port', type=int, default=2000, help='port')
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    args = parser.parse_args()

    simulate(args)
