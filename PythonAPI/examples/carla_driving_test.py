import os 
import carla
import time
import json
import pygame
import numpy as np
import argparse
import math 
import random 

random.seed(42)

# Pygame 초기화
pygame.init()
display = pygame.display.set_mode((1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)

def process_img(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def display_img(display, image):
    image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    display.blit(image_surface, (0, 0))
    pygame.display.flip()

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_actor_display_name(actor, truncate=20):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name

def destroy_actors(world, location, radius=5.0):
    actors = world.get_actors()
    for actor in actors:
        if actor.get_location().distance(location) < radius:
            actor.destroy()

def main(args):
    # 데이터 파일 경로 설정
    data_file = os.path.join('./log', args.log_file)
    drive_data = load_json(data_file)

    # CARLA 클라이언트 설정
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    # 월드 가져오기
    world = client.get_world()

    # 청사진 라이브러리 가져오기
    blueprint_library = world.get_blueprint_library()

    # 차량 청사진 선택
    # vehicle_bp = blueprint_library.find('vehicle.tesla.model3') 
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    # 첫 번째 데이터에서 시작 위치 가져오기
    start_data = drive_data[0]
    start_location_x = start_data['Location']['x']
    start_location_y = start_data['Location']['y']
    start_height = start_data['Height']
    start_location = carla.Location(x=start_location_x, y=start_location_y, z=start_height+2)

    # 기존 객체 제거
    destroy_actors(world, start_location)

    # 차량 생성 시도
    vehicle = None
    while vehicle is None:
        vehicle = world.try_spawn_actor(vehicle_bp, carla.Transform(start_location))
        if vehicle is None:
            print("Spawn failed, retrying in 1 second...")
            time.sleep(1)
            destroy_actors(world, start_location)

    # 카메라 센서 청사진 선택
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '110')

    # 카메라 센서 부착 위치 설정
    # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera_transform = carla.Transform(carla.Location(x=1.0, y=0.0, z=1.2), carla.Rotation(pitch=0.0))

    # 카메라 센서 생성 및 차량에 부착
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 카메라 센서에서 이미지를 받아 처리할 콜백 함수 설정
    camera.listen(lambda image: display_img(display, process_img(image)))

    try:
        for entry in drive_data:
            # 위치 및 센서 데이터 가져오기
            location_x = entry['Location']['x']
            location_y = entry['Location']['y']
            height = entry['Height']
            compass = entry['Compass']
            accel_x, accel_y, accel_z = entry['Accelerometer']
            gyro_x, gyro_y, gyro_z = entry['Gyroscope']

            # 차량의 새로운 위치 및 방향 설정
            new_location = carla.Location(x=location_x, y=location_y, z=height)
            yaw = compass - 90  # Convert compass heading to yaw angle
            new_rotation = carla.Rotation(yaw=yaw)
            new_transform = carla.Transform(new_location, new_rotation)
            vehicle.set_transform(new_transform)

            # 차량의 가속도 및 자이로 데이터 출력
            # print(f"Accel: {accel_x}, {accel_y}, {accel_z}, Gyro: {gyro_x}, {gyro_y}, {gyro_z}") 
            print(f"Time: {entry['Simulation time']}  |  Location_x: {location_x} Location_y: {location_y} Heading: {entry['Heading']}")

            # 타임스탬프 간격에 따라 대기
            time.sleep(args.sleep)  # Adjust as necessary to match the timing of the log entries

    finally:
        # 차량 및 센서 삭제
        vehicle.destroy()
        camera.destroy()
        pygame.quit()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description="CARLA Driving Test") 
    argparser.add_argument('-f', '--log_file', default='driving_log.json', help='Log file to replay') 
    argparser.add_argument('--host', metavar='H', default='172.30.1.101', help='IP of the host server (default:172.30.1.101)') 
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--sleep', default=0.05, type=float, help='Time to sleep between log entries')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    
    args = argparser.parse_args() 
    main(args)
