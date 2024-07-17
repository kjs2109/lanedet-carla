# # from __future__ import print_function

# # import glob
# # import os
# # import sys
# # import csv
# # import time

# # try:
# #     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
# #         sys.version_info.major,
# #         sys.version_info.minor,
# #         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# # except IndexError:
# #     pass 


# # import carla

from carla import ColorConverter as cc

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

from manual_control import (KeyboardControl, World, HUD, FadingText, HelpText, CollisionSensor, 
                            LaneInvasionSensor, GnssSensor, IMUSensor, RadarSensor) #  CameraManager)


# lanedet model setting 
config_file = '/root/works/lanedet/demo/configs/carla_scnn_tusimple.py'

cfg = Config.fromfile(config_file)
cfg.show = False 
cfg.savedir = '/root/works/lanedet'
cfg.load_from = '/root/works/lanedet/demo/checkpoints/scnn_r18_tusimple.pth'
carla_detect = Detect(cfg)

# # carla setting 
# client = carla.Client('172.30.1.101', 2000) 
# world = client.get_world() 

# bp_lib = world.get_blueprint_library() 
# spawn_points = world.get_map().get_spawn_points() 

# vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
# vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points)) 

# spectator = world.get_spectator() 
# transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)), vehicle.get_transform().rotation) 
# spectator.set_transform(transform)  

# camera_bp = bp_lib.find('sensor.camera.rgb') 
# camera_init_trans = carla.Transform(carla.Location(x=0.6, z=1.6))
# camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle) 

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

# def camera_callback(image, data_dict): 
#     img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))[:, :, :3] 
#     img = np.ascontiguousarray(img)

#     image_buffer.add(img) 
#     buffered_img = image_buffer.get() 

#     if buffered_img is not None:
#         result_img = carla_detect.vis_inference(img, cfg)  
#         data_dict['image'] = result_img  
#     else:
#         data_dict['image'] = img



# image_w = camera_bp.get_attribute("image_size_x").as_int() 
# image_h = camera_bp.get_attribute("image_size_y").as_int() 

# camera_data = {"image": np.zeros((image_w, image_h, 3))} 

 

# def main(args): 
#     hud = HUD(args.width, args.height)
#     sim_world = World(world, hud)
#     controller = KeyboardControl(sim_world, args.autopilot) 


#     while True: 
#         cv2.imshow('RGB Camera', camera_data['image']) 

#         if cv2.waitKey(1) == ord('q'): 
#             break

#     cv2.destroyAllWindows()   



# if __name__ == '__main__': 

#     import argparse

#     argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
#     argparser.add_argument('--verbose', action='store_true', dest='debug', help='print debug information')
#     argparser.add_argument('--host',metavar='H',default= '172.30.1.101', help='IP of the host server (default: 127.0.0.1)') # '127.0.0.1',h
#     argparser.add_argument('--port',metavar='P',default=2000,type=int,help='TCP port to listen to (default: 2000)')
#     argparser.add_argument('--autopilot',action='store_true',help='enable autopilot')
#     argparser.add_argument('--res',metavar='WIDTHxHEIGHT',default='1280x720',help='window resolution (default: 1280x720)')
#     argparser.add_argument('--filter',metavar='PATTERN',default='vehicle.*',help='actor filter (default: "vehicle.*")')
#     argparser.add_argument('--generation',metavar='G',default='2',help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
#     argparser.add_argument('--rolename',metavar='NAME',default='hero',help='actor role name (default: "hero")')
#     argparser.add_argument('--gamma',default=2.2,type=float,help='Gamma correction of the camera (default: 2.2)')
#     argparser.add_argument('--sync',action='store_true',help='Activate synchronous mode execution')
#     args = argparser.parse_args()

#     args.width, args.height = [int(x) for x in args.res.split('x')] 

#     main(args) 




import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return [] 


 

# def main(args): 
#     hud = HUD(args.width, args.height)
#     sim_world = World(world, hud)
#     controller = KeyboardControl(sim_world, args.autopilot) 


#     while True: 
#         cv2.imshow('RGB Camera', camera_data['image']) 

#         if cv2.waitKey(1) == ord('q'): 
#             break

#     cv2.destroyAllWindows()   


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        # self.camera_data = {"image": np.zeros((hud.dim[0], hud.dim[1], 3))} 

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image)) # , self.camera_data))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        # else:
        #     image.convert(self.sensors[self.index][1])
        #     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        #     array = np.reshape(array, (image.height, image.width, 4))
        #     array = array[:, :, :3]
        #     img = np.ascontiguousarray(array)


        #     # array = array[:, :, ::-1]
        #     # self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        #     logging.info('img shape ', img.shape)
        #     result_img = carla_detect.vis_inference(img, cfg)  
        #     # self.camera_data['image'] = result_img   
        #     self.surface = pygame.surfarray.make_surface(result_img.swapaxes(0, 1)) 

        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            # OpenCV를 사용하여 사각형 그리기
            # 사각형의 좌상단 좌표 (x1, y1)와 우하단 좌표 (x2, y2)
            x1, y1 = 50, 50
            x2, y2 = 200, 200
            color = (0, 0, 255)  # 사각형 색상 (BGR 형식)
            thickness = 2  # 사각형 두께
            array = np.ascontiguousarray(array)
            cv2.rectangle(array, (x1, y1), (x2, y2), color, thickness)

            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
import pygame 

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None 

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        sim_world = client.get_world()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host',metavar='H',default= '172.30.1.101', help='IP of the host server (default: 127.0.0.1)') # '127.0.0.1',h
    argparser.add_argument('--port',metavar='P',default=2000,type=int,help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--autopilot',action='store_true',help='enable autopilot')
    argparser.add_argument('--res',metavar='WIDTHxHEIGHT',default='1280x720',help='window resolution (default: 1280x720)')
    argparser.add_argument('--filter',metavar='PATTERN',default='vehicle.*',help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--generation',metavar='G',default='2',help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument('--rolename',metavar='NAME',default='hero',help='actor role name (default: "hero")')
    argparser.add_argument('--gamma',default=2.2,type=float,help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument('--sync',action='store_true',help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
