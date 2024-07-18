import carla
import json
import datetime
import math
import pygame 
import argparse

class CarlaLogger:
    def __init__(self, filename):
        self.filename = filename
        self.data = []

    def log(self, world, player, clock):
        v = player.get_velocity()
        t = player.get_transform()
        compass = world.player.get_compass()
        heading = self.get_heading(compass)

        log_entry = {
            'Server FPS': world.get_settings().fixed_delta_seconds,
            'Client FPS': clock.get_fps(),
            'Vehicle': self.get_actor_display_name(world.player, truncate=20),
            'Map': world.get_map().name.split('/')[-1],
            'Simulation time': str(datetime.timedelta(seconds=int(world.get_snapshot().timestamp.elapsed_seconds))),
            'Speed km/h': 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2),
            'Compass': compass,
            'Heading': heading,
            'Accelerometer': world.imu_sensor.accelerometer,
            'Gyroscope': world.imu_sensor.gyroscope,
            'Location': {'x': t.location.x, 'y': t.location.y},
            'GNSS': {'lat': world.gnss_sensor.lat, 'lon': world.gnss_sensor.lon},
            'Height': t.location.z
        }
        self.data.append(log_entry)

    def save(self):
        with open(self.filename, 'w') as file:
            json.dump(self.data, file, indent=4)

    @staticmethod
    def get_actor_display_name(actor, truncate=20):
        name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
        return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name

    @staticmethod
    def get_heading(compass):
        heading = 'N'
        if 45 <= compass < 135:
            heading = 'E'
        elif 135 <= compass < 225:
            heading = 'S'
        elif 225 <= compass < 315:
            heading = 'W'
        return heading

def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    clock = pygame.time.Clock()

    logger = CarlaLogger(args.log_file)

    try:
        while True:
            clock.tick_busy_loop(60)
            world.tick()
            logger.log(world, clock)

    finally:
        logger.save()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='172.30.1.101',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--log-file',
        metavar='FILE',
        default='drive_log.json',
        help='path to the log file (default: "drive_log.json")')
    args = argparser.parse_args()

    main(args)
