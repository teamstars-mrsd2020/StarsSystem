#!/usr/bin/env python

import glob
import json
import os
import random
import sys
import weakref
from time import sleep

from tqdm import tqdm
from IPython import embed
from numpy import cos, pi, sin, tan
import cv2

import carla
from carla import ColorConverter as cc
from datetime import datetime

from SimulationSubsystem.verify_dp_params import verify_dp_params

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass


try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

try:
    import numpy as np
except ImportError:
    raise RuntimeError("cannot import numpy, make sure numpy package is installed")

import spawn

local = True

VIEW_WIDTH = 1920
VIEW_HEIGHT = 1080
# VIEW_WIDTH = 1024
# VIEW_HEIGHT = 768
VIEW_FOV = 90

# BB_COLOR = (248, 64, 24)
BB_COLOR = (0, 255, 0)

class_color = {"pedestrian": (0, 255, 0), "vehicle": (0, 0, 255), "bike": (255, 0, 0)}
DISTANCE_TOLERANCE = {"pedestrian": 10, "vehicle": 25, "bike": 20}
# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_class_name(actor):
        attr = actor.attributes
        if attr["role_name"] == "pedestrian":
            return "pedestrian"
        if attr["role_name"] == "autopilot":
            if attr["number_of_wheels"] == "2":
                return "bike"
            else:
                return "vehicle"

    @staticmethod
    def get_metadata(actor):
        type_id = actor.type_id

        def getDictFromVect(vect):
            return {"x": vect.x, "y": vect.y, "z": vect.z}

        id = actor.id
        clsname = ClientSideBoundingBoxes.get_class_name(actor)
        loc = actor.get_location()
        location = getDictFromVect(loc)
        bbox3d = actor.bounding_box
        bbox3d_offset = getDictFromVect(bbox3d.location)
        bbox3d_extent = getDictFromVect(bbox3d.extent)
        velocity = getDictFromVect(actor.get_velocity())
        acc = getDictFromVect(actor.get_acceleration())
        angular_vel = getDictFromVect(actor.get_angular_velocity())

        return {
            "type_id": type_id,
            "agentid": id,
            "classname": clsname,
            "location": location,
            "bounding_box3d": {"offset": bbox3d_offset, "extent": bbox3d_extent},
            "velocity": velocity,
            "acceleration": acc,
            "angular_velocity": angular_vel,
        }

    @staticmethod
    def get_agent_metadata(actors, camera, calibration):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [
            ClientSideBoundingBoxes.get_bounding_box(actor, camera, calibration)
            for actor in actors
        ]
        metadata = [ClientSideBoundingBoxes.get_metadata(actor) for actor in actors]
        # embed()
        # filter objects behind camera
        final_bboxes = []
        final_metadata = []
        for i in range(len(bounding_boxes)):
            if all(bounding_boxes[i][:, 2] > 0):
                final_bboxes.append(bounding_boxes[i])
                final_metadata.append(metadata[i])
        return final_bboxes, final_metadata

    @staticmethod
    def draw_2d_bounding_boxes(
        display, bounding_boxes, metadata, clientObj, camera_location
    ):
        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        final_metadata = []
        for idx, bounding_box in enumerate(bounding_boxes):
            metadat = metadata[idx]
            pos = metadat["location"]
            agent_loc = carla.Location(pos["x"], pos["y"], pos["z"])
            agent_distance = camera_location.distance(agent_loc)
            classname = metadat["classname"]
            clr = class_color[classname]
            bounding_box = bounding_box[:, :2]
            max_bb = (np.asarray(np.max(bounding_box, axis=0)).reshape(2)).astype(int)
            min_bb = (np.asarray(np.min(bounding_box, axis=0)).reshape(2)).astype(int)

            max_x, max_y = max_bb[0], max_bb[1]
            min_x, min_y = min_bb[0], min_bb[1]

            # Choose 4 out of 8 points
            w = max_x - min_x
            h = max_y - min_y
            x1 = (min_x, max_y)
            x2 = (max_x, min_y)

            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(VIEW_WIDTH, max_x)
            max_y = min(VIEW_HEIGHT, max_y)

            # area = w * h
            # if classname == "vehicle":
            #     AREA_THRES = 50
            # else:
            #     AREA_THRES = 20
            if (
                min_x >= 0
                and min_y >= 0
                and max_x <= VIEW_WIDTH
                and max_y <= VIEW_HEIGHT
                and min_x < max_x
                and min_y < max_y
                # and area > AREA_THRES
            ):
                ## Filtering occluded boxes here
                box_centerx = (min_x + max_x) // 2
                box_centery = (min_y + max_y) // 2
                depth_img = clientObj.depth_image

                if depth_img is not None:

                    dy_min = max(0, box_centery - 5)
                    dy_max = min(VIEW_HEIGHT, box_centery + 5)
                    dx_min = max(0, box_centerx - 5)
                    dx_max = min(VIEW_WIDTH, box_centerx + 5)

                    # center_depth = depth_img[box_centery, box_centerx]
                    center_depth = depth_img[dy_min:dy_max, dx_min:dx_max].mean()

                    diff = agent_distance - center_depth
                    # print("Depth diff = ", diff, idx)
                    if abs(diff) > DISTANCE_TOLERANCE[classname]:
                        # FOR VISUALIZATION
                        # clr = (255, 255, 255)
                        # print("Agent Beyond Dist Thres")
                        continue

                    # embed()

                point1 = x1
                point2 = x2
                point3 = (x1[0], x2[1])
                point4 = (x2[0], x1[1])

                pygame.draw.line(bb_surface, clr, point1, point3, 3)
                pygame.draw.line(bb_surface, clr, point2, point3, 3)
                pygame.draw.line(bb_surface, clr, point2, point4, 3)
                pygame.draw.line(bb_surface, clr, point4, point1, 3)

                box = [min_x, min_y, max_x, max_y]

                metadat["bounding_box2d"] = [int(r) for r in box]
                metadat["bounding_box2d_format"] = "XYXY_ABS"
                final_metadata.append(metadat)
                # final_boxes.append([min_x, min_y, max_x, max_y, classname])

        display.blit(bb_surface, (0, 0))
        return final_metadata

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        display.blit(bb_surface, (0, 0))

    def get_bounding_box(vehicle, camera, calibration):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(
            bb_cords, vehicle, camera
        )[:3, :]
        cords_y_minus_z_x = np.concatenate(
            [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]]
        )
        bbox = np.transpose(np.dot(calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1
        )

        # Get agent
        # print("\n\ncamera_bbox")

        # print(camera_bbox)

        # ClientSideBoundingBoxes.draw_2dbounding_boxes_harsh(camera_bbox[:,:2])
        # print("\n\ncamera_calibration")
        # print(camera.calibration)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        # print("BB Vehicle\n\n")
        # print(bb_vehicle_matrix)

        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(
            vehicle.get_transform()
        )
        # print("Vehicke world\n\n")
        # print(vehicle_world_matrix)
        # print("\n\n")
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        # print("bb world\n\n")
        # print("\n\n")
        # print(bb_world_matrix) # TODO:is this the homography??
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        # print("cords\n\n")
        # print("\n\n")
        # print(cords)
        # print("\n\n")
        # print(world_cords.T)
        # print("\n\n")
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.tm = None
        self.camera = None
        self.camera_bev = None
        self.bev_sensor_world_matrix = None

        self.display = None
        self.image = None
        self.depth_image = None
        self.capture = True
        self.traffic_light = None
        self.transformation_matrix = None
        self.camera_pose = None

        self.camera2 = None
        self.display = None
        self.image2 = None
        self.depth_image2 = None
        self.capture2 = True
        self.traffic_light2 = None
        self.transformation_matrix2 = None
        self.camera_pose2 = None

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def get_traffic_light(self, n):
        # Returns a random traffic light( note: not a pole but light)
        traffic_lights = self.world.get_actors().filter("*light*")
        traffic_lights = list(traffic_lights)
        traffic_lights = sorted(traffic_lights, key=lambda x: x.id)
        light_ids = [x.id for x in traffic_lights]
        print(light_ids)
        return traffic_lights[n]

    def get_traffic_light_by_id(self, id):
        # Returns a random traffic light( note: not a pole but light)
        traffic_lights = self.world.get_actors().filter("*light*")
        # print(traffic_lights)
        # print(id)
        light = [x for x in traffic_lights if x.id == id]
        # print(light)
        return light[0]

    def get_camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(VIEW_WIDTH))
        camera_bp.set_attribute("image_size_y", str(VIEW_HEIGHT))
        camera_bp.set_attribute("fov", str(VIEW_FOV))
        return camera_bp

    def get_depth_camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find("sensor.camera.depth")
        camera_bp.set_attribute("image_size_x", str(VIEW_WIDTH))
        camera_bp.set_attribute("image_size_y", str(VIEW_HEIGHT))
        camera_bp.set_attribute("fov", str(VIEW_FOV))
        return camera_bp

    def camera_listener(self, image):
        """
        Camera listener callback
        """
        self.image = image
        image.convert(cc.Raw)
        i = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        i = np.reshape(i, (image.height, image.width, 4))

        cvim = cv2.cvtColor(i, cv2.COLOR_BGRA2BGR)
        self.cv_image = cvim

    def camera_listener2(self, image):
        """
        Camera listener callback
        """
        self.image2 = image
        image.convert(cc.Raw)
        i = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        i = np.reshape(i, (image.height, image.width, 4))

        cvim = cv2.cvtColor(i, cv2.COLOR_BGRA2BGR)
        self.cv_image2 = cvim

    def depth_camera_listener(self, image):
        """
        Depth Camera listener callback
        """
        # print("IN depth listener")
        # ref: https://github.com/carla-simulator/data-collector/blob/master/carla/image_converter.py
        image.convert(cc.Raw)
        # raw image to 4 dim numpy rgba image
        i = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        i = np.reshape(i, (image.height, image.width, 4))

        # 4 dim rgba image to depth image with depth
        i = i.astype(np.float32)
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        # ref https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
        normalized_depth = np.dot(i[:, :, :3], [65536.0, 256.0, 1.0])
        normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        image = normalized_depth * 1000
        self.depth_image = image

    def choose_traffic_light(self, fixed_traffic_light=None):
        # consecutive ids(in groups of 2) are opposite pairs, eg 55,56 is a pair, 57,58 is a pair

        # good_lights = [55, 55, 55, 56, 57, 58, 69, 72, 70, 71, 59, 62, 60, 61]
        if fixed_traffic_light is not None:
            traffic_light_id = fixed_traffic_light
        else:
            good_lights = [65, 54, 62]  # 56, 57, 58, 69, 72, 70, 71]
            traffic_light_id = random.choice(good_lights)

        print("using traffic light id=", traffic_light_id)
        traffic_light = self.get_traffic_light_by_id(traffic_light_id)
        rel_transform = carla.Transform(
            carla.Location(x=-5, y=2, z=7), carla.Rotation(pitch=-10, roll=0, yaw=130)
        )
        # for tf_id 53
        if traffic_light_id == 53:
            rel_transform = carla.Transform(
                carla.Location(x=-5, y=2, z=7),
                carla.Rotation(pitch=-10, roll=0, yaw=132),
            )
        return traffic_light, rel_transform

    def setup_camera(self, fixed_traffic_light1=None, fixed_traffic_light2=None):
        # Spawn a camera on a random traffic light
        camera_bp = self.get_camera_blueprint()
        depth_camera_bp = self.get_depth_camera_blueprint()

        traffic_light, rel_transform = self.choose_traffic_light(fixed_traffic_light1)

        self.traffic_light = traffic_light
        self.camera_pose = rel_transform
        # embed()
        self.camera = self.world.spawn_actor(
            camera_bp, rel_transform, attach_to=traffic_light
        )
        self.depth_camera = self.world.spawn_actor(
            depth_camera_bp, rel_transform, attach_to=traffic_light
        )

        rel_transform2 = carla.Transform(
            carla.Location(x=-30, y=50, z=100),
            carla.Rotation(pitch=-90, roll=180, yaw=0),
        )

        # sleep(0.1)
        # weak_self = weakref.ref(self)
        self.camera.listen(self.camera_listener)
        self.depth_camera.listen(self.depth_camera_listener)
        self.calibration = self.compute_intrinsics(self.camera)

        # get the bev camera and store its world-to-camera transform
        rel_transform_bev = carla.Transform(
            carla.Location(x=-30, y=50, z=100),
            carla.Rotation(pitch=-90, roll=180, yaw=0),
        )

        self.camera_bev = self.world.spawn_actor(
            camera_bp, rel_transform_bev, attach_to=traffic_light
        )
        sleep(0.5)
        self.bev_sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(
            self.camera_bev.get_transform()
        )
        self.camera_bev.destroy()

        traffic_light, rel_transform = self.choose_traffic_light(fixed_traffic_light2)
        self.traffic_light2 = traffic_light
        self.camera_pose2 = rel_transform
        # embed()
        self.camera2 = self.world.spawn_actor(
            camera_bp, rel_transform, attach_to=traffic_light
        )
        self.depth_camera2 = self.world.spawn_actor(
            depth_camera_bp, rel_transform, attach_to=traffic_light
        )

        # sleep(0.1)
        # weak_self = weakref.ref(self)
        self.camera2.listen(self.camera_listener2)
        self.depth_camera2.listen(self.depth_camera_listener)
        self.calibration2 = self.compute_intrinsics(self.camera2)

    def compute_intrinsics(self, cam):
        """
        Computes intrinsics and returns a 3x3 K intrinsics matrix
        """
        ImageSizeX = int(cam.attributes["image_size_x"])
        ImageSizeY = int(cam.attributes["image_size_y"])
        camFOV = float(cam.attributes["fov"])

        focal_length = ImageSizeX / (2 * tan(camFOV * pi / 360))
        center_X = ImageSizeX / 2
        center_Y = ImageSizeY / 2
        intrinsics = np.array(
            [[focal_length, 0, center_X], [0, focal_length, center_Y], [0, 0, 1]]
        )
        return intrinsics

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1.0, min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1.0, max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    # @staticmethod
    # def set_image(weak_self, img):
    #     """
    #     Sets image coming from camera sensor.
    #     The self.capture flag is a mean of synchronization - once the flag is
    #     set, next coming image will be stored.
    #     """

    #     self = weak_self()
    #     if self.capture:
    #         self.image = img
    #         self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def apply_batch(self, *args, **kwargs):
        self.client.apply_batch(*args, **kwargs)

    def get_random_clear_weather(self):
        # return carla.WeatherParameters(
        #     cloudyness=random.uniform(0, 20),
        #     precipitation=random.uniform(0, 1),
        #     precipitation_deposits=0.0,
        #     wind_intensity=0.0,
        #     sun_azimuth_angle=0.0,
        #     sun_altitude_angle=random.uniform(40, 90),
        #     fog_density=0.0,# not in 0.9.6
        #     fog_distance=0.0,# not in 0.9.6
        #     wetness=0.0,#not in 0.9.6
        # )
        return carla.WeatherParameters(
            cloudiness=random.uniform(0, 20),
            precipitation=random.uniform(0, 1),
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=random.uniform(40, 90),
            fog_density=0.0,  # not in 0.9.6
            fog_distance=0.0,  # not in 0.9.6
            wetness=0.0,  # not in 0.9.6
            fog_falloff=0.2,
        )

    def is_car_or_walker(self, agent):
        type_id = agent.type_id
        return ".pedestrian." in type_id or "vehicle." in type_id

    def reset_weather(self):
        self.weather = self.get_random_clear_weather()
        self.world.set_weather(self.weather)

    def remove_vehicles_walkers(self):
        actors = self.world.get_actors()
        actor_ids = [actor.id for actor in actors if self.is_car_or_walker(actor)]
        self.apply_batch([carla.command.DestroyActor(x) for x in actor_ids])

    def spawn_vehicles_walkers(self, safe=False):
        # TODO Figure out a better way to randomize this
        # Current issue -> the number is mostly fixed because of rendering issues
        # (reason for      with difference in weather, camera location. Also since the vehicles
        #   deferral)      are spawned and moving randomly, a fixed global count of vehicle
        #                  still gives us randomness in the camera view as
        #                  the movement of agents varies

        # load behavior config
        with open(self.files["behavior_config"]) as f:
            conf = json.load(f)

        number_of_cars = conf["number_of_vehicles"]
        number_of_pedestrians = conf["number_of_walkers"]
        # #Tracking
        # number_of_cars = 10
        # number_of_pedestrians = 0
        vehicles_list, walkers_list, all_actors = spawn.spawn(
            self.client, self.world, number_of_cars, number_of_pedestrians, safe=safe
        )
        for actor in self.world.get_actors():
            if "vehicle." in actor.type_id:
                # print(actor.type_id)
                tm_port = self.tm.get_port()
                actor.set_autopilot(True, tm_port)
                self.tm.ignore_lights_percentage(
                    actor, conf["ignore_lights_percentage"]
                )
        self.tm.set_global_distance_to_leading_vehicle(
            conf["global_distance_to_leading_vehicle"]
        )
        # self.tm.vehicle_percentage_speed_difference(actor,-20)

    def set_fixed_framerate(self, FPS):
        # Set fixed framerate
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1 / FPS
        self.world.apply_settings(settings)

    def setup_videowriter(self, FPS):
        # output_file_name = os.path.join(
        #     self.output_folder+"/videos", "c_" + str(self.intersection_id).zfill(3) + "_" + str(self.side_id).zfill(3) + "_" + str(self.round_).zfill(3) + ".mp4"
        # )
        output_file_name1 = self.files["video"]
        output_file_name2 = self.files["video_2"]

        fourcc = cv2.VideoWriter_fourcc(*"mjpg")
        self.videowriter = cv2.VideoWriter(
            output_file_name1, fourcc, FPS, (VIEW_WIDTH, VIEW_HEIGHT)
        )
        self.videowriter2 = cv2.VideoWriter(
            output_file_name2, fourcc, FPS, (VIEW_WIDTH, VIEW_HEIGHT)
        )

    def init_client(
        self,
        hoststr,
        port,
        tm_port,
        round_name,
        output_folder,
        conf,
        intersection_1,
        intersection_2,
        FPS=5,
        safe=False,
        spawn_actors=True,
    ):
        # client.init_client(
        #             hoststr, port, tm_port, round_name, output_folder, conf, intersection_1, intersection_2, FPS, safe, spawn_actors=SPAWN_ACTORS
        #         )

        #     intersection_id,
        # side_id,
        # round_,
        fixed_traffic_light1 = intersection_1["traffic_light_id"]
        fixed_traffic_light2 = intersection_2["traffic_light_id"]

        self.client = carla.Client(hoststr, port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.tm = self.client.get_trafficmanager(tm_port)
        self.set_synchronous_mode(False)
        self.set_fixed_framerate(FPS)
        self.reset_weather()
        sleep(2)
        self.files = conf
        if spawn_actors:
            self.spawn_vehicles_walkers(safe)
            sleep(5)
        self.setup_camera(fixed_traffic_light1, fixed_traffic_light2)
        self.fixed_traffic_light = fixed_traffic_light1
        self.fixed_traffic_light2 = fixed_traffic_light2
        self.metadata_all = []
        self.metadata_all2 = []
        self.round_name = round_name
        self.output_folder = output_folder
        self.setup_videowriter(FPS)
        sleep(3)
        self.set_synchronous_mode(True)
        self.verify_dp = None
        self.gt_LVD = None
        self.gt_TLJ = None
        self.tl_list = intersection_1["neighbors"]
        self.intersection_id = intersection_1["intersection_id"]
        self.round_ = int(conf["video"][-7:-4])
        self.side_id1 = intersection_1["side"]
        self.side_id2 = intersection_2["side"]

    def de_init_client(self):
        self.remove_vehicles_walkers()
        self.videowriter.release()

        # with open(
        #     os.path.join(self.output_folder+"/metadata", "cgt_" + str(self.intersection_id).zfill(3) + "_" + str(self.side_id).zfill(3) + "_" + str(self.round_).zfill(3) + ".json"), "w"
        # ) as f:
        #     json.dump(self.metadata_all, f)

        with open(
            os.path.join(
                self.output_folder + "/metadata/aggregated",
                "cgt_"
                + str(self.intersection_id).zfill(3)
                + "_"
                + str(self.side_id1).zfill(3)
                + "_"
                + str(self.round_).zfill(3)
                + ".json",
            ),
            "w",
        ) as f:
            json.dump(self.metadata_all, f)

        with open(
            os.path.join(
                self.output_folder + "/metadata/aggregated",
                "cgt_"
                + str(self.intersection_id).zfill(3)
                + "_"
                + str(self.side_id2).zfill(3)
                + "_"
                + str(self.round_).zfill(3)
                + ".json",
            ),
            "w",
        ) as f:
            json.dump(self.metadata_all2, f)

        # write ground truth params
        gt_params_file = self.files["params_gt"]
        with open(gt_params_file, "w") as f:
            json.dump(
                {
                    "TLV": self.gt_TLJ,
                    "LVD": self.gt_LVD,
                },
                f,
            )

    def game_loop(self, number_of_frames=None, num_camera=1):
        """
        Main program loop.
        """
        # For each iteration, save the image and bounding boxes(xmin,ymin,xmax,ymax,classname)

        try:
            pygame.init()

            self.verify_dp = verify_dp_params(self.world, self.tm, [45, 83, 56, 84])
            # self.client = carla.Client(hoststr, port)
            # self.client.set_timeout(5.0)
            # self.world = self.client.get_world()

            # self.setup_car()
            # self.setup_camera()

            self.display = pygame.display.set_mode(
                (VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            pygame_clock = pygame.time.Clock()

            # self.set_synchronous_mode(True)
            all_actors = self.world.get_actors()
            # embed()
            vehicles = all_actors.filter("vehicle.*")
            peds = all_actors.filter("walker.pedestrian.*")
            # lights = all_actors.filter("*traffic_light*") # no boxes for lights :(

            filtered_agents = [v for v in vehicles] + [
                p for p in peds
            ]  # + [l for l in lights]

            SAVE_DATA = True
            if self.files["duration"] > .6:
                SAVE_FRAME_WISE_DATA = False 
            else:
                SAVE_FRAME_WISE_DATA = True 
            
            DEBUG_DRAW = False
            sensor_world_matrix1 = ClientSideBoundingBoxes.get_matrix(
                self.camera.get_transform()
            )
            traffic_light_id = self.traffic_light.id

            sensor_world_matrix2 = ClientSideBoundingBoxes.get_matrix(
                self.camera2.get_transform()
            )
            traffic_light_id2 = self.traffic_light2.id

            if SAVE_DATA:

                config_file1 = os.path.join(
                    self.output_folder + "/metadata/aggregated",
                    "capture_data"
                    + "_c_"
                    + str(self.intersection_id).zfill(3)
                    + "_"
                    + str(self.side_id1).zfill(3)
                    + ".json",  # Saves
                )
                with open(config_file1, "w") as f:
                    json.dump(
                        {
                            "sensor_world_matrix": sensor_world_matrix1.tolist(),
                            "bev_sensor_world_matrix": self.bev_sensor_world_matrix.tolist(),
                            "traffic_light_id": traffic_light_id,
                            "map_name": self.world.get_map().name,
                            "weather": str(self.weather),
                            "intrinsics": self.calibration.tolist(),
                        },
                        f,
                    )

                config_file2 = os.path.join(
                    self.output_folder + "/metadata/aggregated",
                    "capture_data"
                    + "_c_"
                    + str(self.intersection_id).zfill(3)
                    + "_"
                    + str(self.side_id2).zfill(3)
                    + ".json",  # Saves
                )
                with open(config_file2, "w") as f:
                    json.dump(
                        {
                            "sensor_world_matrix": sensor_world_matrix2.tolist(),
                            "bev_sensor_world_matrix": self.bev_sensor_world_matrix.tolist(),
                            "traffic_light_id": traffic_light_id2,
                            "map_name": self.world.get_map().name,
                            "weather": str(self.weather),
                            "intrinsics": self.calibration2.tolist(),
                        },
                        f,
                    )

            k = 0
            loop_cond = (
                lambda x: True if number_of_frames is None else x < number_of_frames
            )
            if number_of_frames:
                pbar = tqdm(total=number_of_frames)
            else:
                pbar = tqdm()

            # Folder for storing metadata - used by tracking
            output_folder1 = (
                self.output_folder
                + "/metadata/frame_wise/"
                + "c_"
                + str(self.intersection_id).zfill(3)
                + "_"
                + str(self.side_id1).zfill(3)
                + "_"
                + str(self.round_).zfill(3)
            )
            if not os.path.exists(output_folder1):
                os.makedirs(output_folder1)
                output_folder1_files = glob.glob(output_folder1)
            else:
                output_folder1_files = glob.glob(output_folder1)
                if len(output_folder1_files) > 0:
                    raise Exception(
                        f"Found {len(output_folder1_files)} files in {output_folder1}, please manually clean it up and rerun"
                    )

            output_folder2 = (
                self.output_folder
                + "/metadata/frame_wise/"
                + "c_"
                + str(self.intersection_id).zfill(3)
                + "_"
                + str(self.side_id2).zfill(3)
                + "_"
                + str(self.round_).zfill(3)
            )
            if not os.path.exists(output_folder2):
                os.makedirs(output_folder2)
                output_folder2_files = glob.glob(output_folder2)
            else:
                output_folder2_files = glob.glob(output_folder2)
                if len(output_folder2_files) > 0:
                    raise Exception(
                        f"Found {len(output_folder2_files)} files in {output_folder2}, please manually clean it up and rerun"
                    )
            if not os.path.exists(output_folder1):
                os.makedirs(output_folder1)

            while loop_cond(k):
                self.verify_dp.update_observed_params()
                print(
                    "TLV percentage: ", self.verify_dp.observed_percentage_jumping_light
                )
                if self.verify_dp.observed_least_lvd == 1000:
                    print("Minimum LVD: No pairs observed yet")
                else:
                    print("Minimum LVD: ", self.verify_dp.observed_least_lvd)
                k += 1
                self.world.tick()
                snap = self.world.get_snapshot()
                frameid = snap.timestamp.frame
                timestamp = int(snap.timestamp.elapsed_seconds * 1000)  # milliseconds

                self.capture = True
                pygame_clock.tick_busy_loop(20)

                self.render(self.display)
                bounding_boxes, metadata = ClientSideBoundingBoxes.get_agent_metadata(
                    filtered_agents, self.camera, self.calibration
                )
                bounding_boxes2, metadata2 = ClientSideBoundingBoxes.get_agent_metadata(
                    filtered_agents, self.camera2, self.calibration2
                )
                cam_loc_obj = self.camera.get_location()
                cam_loc_obj2 = self.camera2.get_location()

                final_metadata = ClientSideBoundingBoxes.draw_2d_bounding_boxes(
                    self.display,
                    bounding_boxes,
                    metadata,
                    self,
                    cam_loc_obj,
                )
                final_metadata2 = ClientSideBoundingBoxes.draw_2d_bounding_boxes(
                    self.display,
                    bounding_boxes2,
                    metadata2,
                    self,
                    cam_loc_obj2,
                )
                if DEBUG_DRAW:
                    debug = self.world.debug
                    for m in final_metadata:
                        posx = m["location"]["x"]
                        posy = m["location"]["y"]
                        posz = m["location"]["z"]
                        loc = carla.Location(int(posx), int(posy), int(posz))
                        classname = m["classname"]
                        c = class_color[classname]
                        debug.draw_point(
                            loc,
                            size=0.05,
                            color=carla.Color(c[0], c[1], c[2]),
                            life_time=20.0,
                        )
                    # embed()
                # save boxes here
                if SAVE_DATA:
                    if SAVE_FRAME_WISE_DATA:
                        filename = os.path.join(
                            output_folder1,
                            str(frameid)
                            + "_"
                            + str(timestamp),  # Framewise - image, final meta data
                        )
                        self.image.save_to_disk(filename)

                        with open(filename + ".json", "w") as f:
                            json.dump(final_metadata, f)

                        filename = os.path.join(
                            output_folder2,
                            str(frameid)
                            + "_"
                            + str(timestamp),  # Framewise - image, final meta data
                        )
                        self.image2.save_to_disk(filename)

                        with open(filename + ".json", "w") as f:
                            json.dump(final_metadata2, f)

                    # cv2.imshow("image", self.cv_image)
                    # cv2.waitKey(0)
                    self.videowriter.write(self.cv_image)
                    self.videowriter2.write(self.cv_image2)
                    # filename = f'{output_folder}/{frameid}_{timestamp}' #py36
                    frame_metadata = {
                        "frame_id": frameid,
                        "timestamp": timestamp,
                        "metadata": final_metadata,
                    }
                    frame_metadata2 = {
                        "frame_id": frameid,
                        "timestamp": timestamp,
                        "metadata": final_metadata2,
                    }
                    self.metadata_all.append(frame_metadata)
                    self.metadata_all2.append(frame_metadata2)
                pygame.display.flip()

                pygame.event.pump()
                pbar.update(1)
            pbar.close()

        finally:
            self.gt_TLJ = self.verify_dp.observed_percentage_jumping_light
            self.gt_LVD = self.verify_dp.observed_least_lvd
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.depth_camera.destroy()
            self.camera2.destroy()
            self.depth_camera2.destroy()
            pygame.quit()


# -----------------------------------------------------------------------------------------


def label_traffic_lights_debug(world):
    tls = list(world.get_actors().filter("*light*"))
    dbg = world.debug

    def offset_loc(loc):
        return carla.Location(loc.x, loc.y + 1, loc.z)

    for t in tls:
        dbg.draw_string(
            offset_loc(t.get_location()),
            "ID=" + str(t.id),
            color=carla.Color(255, 255, 255),
            life_time=60,
        )
        # dbg.draw_point(
        #     t.get_location(), 0.5, color=carla.Color(255, 0, 0), life_time=0,
        # )


def multiple_rounds_main(conf):

    with open(conf["intersection_config"]) as f:
        intersection_1 = json.load(f)

    with open(conf["intersection_config_2"]) as f:
        intersection_2 = json.load(f)

    tl_id1 = intersection_1["traffic_light_id"]
    tl_id2 = intersection_2["traffic_light_id"]

    if local:
        hoststr = "127.0.0.1"
        port = 2000
        tm_port = 8000
    else:
        hoststr = "dumbledore"
        port = 2222
        tm_port = 8000

    # Default
    # safe=False
    # number_of_rounds = 10
    # mins_per_round = 2
    # FPS = 2

    # Tracking
    safe = True  # no spawning of bicycle riders

    FIXED_TRAFFIC_LIGHTS = True
    SPAWN_ACTORS = True

    if FIXED_TRAFFIC_LIGHTS:
        fixed_traffic_lights1 = [tl_id1]
        fixed_traffic_lights2 = [tl_id2]
        rounds_per_light = 1
    else:
        number_of_rounds = 4

    mins_per_round = conf["duration"]
    FPS = 20.0
    number_of_frames = int(mins_per_round * 60 * FPS)

    if number_of_frames < 1:
        return False

    now = datetime.now()

    # output_folder = now.strftime("%d %B %Y %H_%M_%S") + " " + str(FPS) + "FPS"
    # output_folder = os.path.join("../new_output_test_data", output_folder)
    # output_folder = "./../../detector/data/raw_data/svd_encore"
    output_folder = "./../data"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # SAVE_DATA = False
    # DEBUG_DRAW = False
    try:
        client = BasicSynchronousClient()
        if FIXED_TRAFFIC_LIGHTS:
            number_of_rounds = len(fixed_traffic_lights1) * rounds_per_light
            traffic_lights1 = fixed_traffic_lights1 * number_of_rounds
            traffic_lights2 = fixed_traffic_lights2 * number_of_rounds
        for sc_id in range(number_of_rounds):
            if FIXED_TRAFFIC_LIGHTS:
                traffic_light1 = traffic_lights1[sc_id]
                traffic_light2 = traffic_lights2[sc_id]
                round_name = "demo_tfl_" + str(traffic_light1)
                round_name2 = "demo_tfl_" + str(traffic_light2)
                # round_name = "round_" + str(sc_id) + "_" + str(traffic_light)
            else:
                traffic_light = None
                round_name = "round_" + str(sc_id)
            print("capturing round = ", round_name)
            client.init_client(
                hoststr,
                port,
                tm_port,
                round_name,
                output_folder,
                conf,
                intersection_1,
                intersection_2,
                FPS,
                safe,
                spawn_actors=SPAWN_ACTORS,
            )
            client.game_loop(number_of_frames)
            client.de_init_client()
    finally:
        if client is not None:
            client.de_init_client()
            print("EXIT")

    if os.path.exists(conf["TL_status"]):
        os.remove(conf["TL_status"])
    if os.path.exists(conf["TL_status_2"]):
        os.remove(conf["TL_status_2"])
    if os.path.exists(conf["TL_status_combined"]):
        os.remove(conf["TL_status_combined"])
    if os.path.exists(conf["trajectory_pred"]):
        os.remove(conf["trajectory_pred"])
    if os.path.exists(conf["trajectory_pred_2"]):
        os.remove(conf["trajectory_pred_2"])
    if os.path.exists(conf["trajectory_pred_combined"]):
        os.remove(conf["trajectory_pred_combined"])
    if os.path.exists(conf["params_pred"]):
        os.remove(conf["params_pred"])


if __name__ == "__main__":

    run_id = 20
    with open("../data/runs/run_" + str(run_id) + ".json") as f:
        conf = json.load(f)
    multiple_rounds_main(conf)

    # client = carla.Client("127.0.0.1", 2000)
    # client.set_timeout(5.0)
    # world = client.get_world()
    # label_traffic_lights_debug(world)
    # multiple_rounds_main(tl_id1, tl_id2, intersection_id, side_id, round_, run_id)