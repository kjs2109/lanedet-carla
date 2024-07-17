# from __future__ import print_function

# import glob
# import os
# import sys
# import csv
# import time

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass 


# import carla

# from carla import ColorConverter as cc

# import argparse
# import collections
# import datetime
# import logging
# import math
# import random
# import re
# import weakref

# import os
# import argparse
# import pygame
# from threading import Thread
# from PIL import Image 

import carla 
import math 
import random 
import time  
import numpy as np 
import cv2   


import os 
import cv2 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from collections import deque

from tools.my_detect import Detect 
from lanedet.utils.config import Config  

random.seed(41) 

# lanedet model setting 
config_file = '/root/works/lanedet/demo/configs/carla_scnn_tusimple.py'

cfg = Config.fromfile(config_file)
cfg.show = False 
cfg.savedir = '/root/works/lanedet'
cfg.load_from = '/root/works/lanedet/demo/checkpoints/scnn_r18_tusimple.pth'
carla_detect = Detect(cfg)

# carla setting 
client = carla.Client('172.30.1.101', 2000) 
# client.load_world('Town01')
world = client.get_world() 

bp_lib = world.get_blueprint_library() 
spawn_points = world.get_map().get_spawn_points() 

vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
# vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points)) 
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[1])

spectator = world.get_spectator() 
transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)), vehicle.get_transform().rotation) 
spectator.set_transform(transform)  

camera_bp = bp_lib.find('sensor.camera.rgb') 
camera_init_trans = carla.Transform(carla.Location(x=0.6, z=1.6))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle) 

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

image_buffer = ImageBuffer(size=10) 

def camera_callback(image, data_dict): 
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))[:, :, :3] 
    img = np.ascontiguousarray(img)

    image_buffer.add(img) 
    buffered_img = image_buffer.get() 

    if buffered_img is not None:
        result_img = carla_detect.vis_inference(img, cfg)  
        # result_img = carla_detect.vis_inference_with_line(img, cfg) 
        data_dict['image'] = result_img  
    else:
        data_dict['image'] = img



image_w = camera_bp.get_attribute("image_size_x").as_int()  
image_h = camera_bp.get_attribute("image_size_y").as_int() 

camera_data = {"image": np.zeros((image_w, image_h, 3))} 

camera.listen(lambda image: camera_callback(image, camera_data)) 

vehicle.set_autopilot(True) 


cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE) 
cv2.imshow('RGB Camera', camera_data['image']) 
cv2.waitKey(1) 


while True: 
    cv2.imshow('RGB Camera', camera_data['image']) 

    if cv2.waitKey(1) == ord('q'): 
        break

cv2.destroyAllWindows() 