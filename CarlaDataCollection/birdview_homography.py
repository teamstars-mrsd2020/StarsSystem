#!/usr/bin/env python

import glob
import json
import os
import random
import sys
import weakref
from time import sleep

from IPython import embed
from numpy import cos, pi, sin, tan

import carla
from carla import ColorConverter as cc

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
VIEW_WIDTH = 1920
VIEW_HEIGHT = 1080
# VIEW_WIDTH = 1024
# VIEW_HEIGHT = 768
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

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
            if attr["number_of_wheels"] == 2:
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
    def get_bounding_boxes(actors, camera, calibration):
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
    def get_bounding_2d_boxes_coordinates(bounding_boxes, type_object):

        coordinates = []

        for bbox in bounding_boxes:

            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]

            # print("points shape:", len(points))
            # print("type points:", type(points))
            # print("one point type:", type(points[0]))
            # print(points)
            min_x = points[0][0]
            min_y = points[0][1]
            max_x = points[0][0]
            max_y = points[0][1]
            for point in points:
                if min_x >= 0:
                    if point[0] >= 0 and point[0] <= min_x:
                        min_x = point[0]
                else:
                    if point[0] >= 0:
                        min_x = point[0]
                if point[0] >= 0 and point[0] >= max_x:
                    max_x = point[0]
                if point[1] <= min_y:
                    min_y = point[1]
                if point[1] >= max_y:
                    max_y = point[1]
            if min_x < 0 or min_x == max_x:
                pass
                # print()
                # print("unable to find extrema values:", min_x, max_x, min_y, max_y)
            else:
                # print("rectangle points:", min_x, max_x, min_y, max_y)
                point1 = (min_x, min_y)
                point2 = (min_x, max_y)
                point3 = (max_x, max_y)
                point4 = (max_x, min_y)

                if type_object == 1:

                    if ((max_x - min_x) >= 12) and ((max_y - min_y) >= 20):

                        coordinates.append(point1 + point3)
                    else:
                        print(
                            "car coordinates too far from the scene:",
                            min_x,
                            max_x,
                            min_y,
                            max_y,
                        )
                else:
                    if ((max_x - min_x) >= 12) and ((max_y - min_y) >= 12):

                        coordinates.append(point1 + point3)
                    else:
                        print(
                            "walker coordinates too far from the scene:",
                            min_x,
                            max_x,
                            min_y,
                            max_y,
                        )

                pygame.draw.line(bb_surface, BB_COLOR, point1, point2)
                pygame.draw.line(bb_surface, BB_COLOR, point2, point3)
                pygame.draw.line(bb_surface, BB_COLOR, point3, point4)
                pygame.draw.line(bb_surface, BB_COLOR, point4, point1)
        # print(coordinates)
        return coordinates

    @staticmethod
    def draw_2d_bounding_boxes(display, bounding_boxes, metadata):
        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        final_metadata = []
        for idx, bounding_box in enumerate(bounding_boxes):
            metadat = metadata[idx]
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

            area = w * h
            AREA_THRES = 150
            if (
                min_x >= 0
                and min_y >= 0
                and max_x <= VIEW_WIDTH
                and max_y <= VIEW_HEIGHT
                and area > AREA_THRES
            ):

                # bbox_2d = [x1, x2, w, h]

                point1 = x1
                point2 = x2
                point3 = (x1[0], x2[1])
                point4 = (x2[0], x1[1])

                pygame.draw.line(bb_surface, BB_COLOR, point1, point3)
                pygame.draw.line(bb_surface, BB_COLOR, point2, point3)
                pygame.draw.line(bb_surface, BB_COLOR, point2, point4)
                pygame.draw.line(bb_surface, BB_COLOR, point4, point1)

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

    # @staticmethod
    # def get_bounding_box(vehicle, camera):
    #     """
    #     Returns 3D bounding box for a vehicle based on camera view.
    #     """

    #     bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
    #     cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
    #     # embed()
    #     cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    #     embed()
    #     bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
    #     camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
    #     # print("\n\ncamera_bbox")

    #     # print(camera_bbox)

    #     # ClientSideBoundingBoxes.draw_2dbounding_boxes_harsh(camera_bbox[:,:2])
    #     # print("\n\ncamera_calibration")
    #     # print(camera.calibration)
    #     return camera_bbox
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
        cords = np.dot(world_sensor_matrix, cords)
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

    def __init__(self, intersection_id):
        self.client = None
        self.world = None
        self.camera = None
        self.camera2 = None
        self.car = None
        # self.traffic_light_id= 62 #[55, 65, 54, 62,61] Get Traffic light on side 0
        self.display = None
        self.image = None
        self.capture = True
        self.image_saved = False
        self.image2_saved = False
        self.intersection_id = intersection_id
        path = (
            "./../data/configs/intersection_config/c_"
            + str(intersection_id).zfill(3)
            + "_"
            + str(0).zfill(3)
            + ".json"
        )
        with open(
            path,
        ) as f:
            config = json.load(f)
        self.traffic_light_id = config["traffic_light_id"]

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(VIEW_WIDTH))
        camera_bp.set_attribute("image_size_y", str(VIEW_HEIGHT))
        camera_bp.set_attribute("fov", str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """
        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def get_traffic_light_by_id(self, id):
        # Returns a random traffic light( note: not a pole but light)
        traffic_lights = self.world.get_actors().filter("*light*")
        light = [x for x in traffic_lights if x.id == id]
        return light[0]

    def get_weather(self):
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
            cloudiness=100,  # random.uniform(0, 20),
            precipitation=random.uniform(0, 1),
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            sun_azimuth_angle=90,
            sun_altitude_angle=90,  # random.uniform(40, 90),
            fog_density=0.0,  # not in 0.9.6
            fog_distance=0.0,  # not in 0.9.6
            wetness=0.0,  # not in 0.9.6
            fog_falloff=0.2,
        )

    def get_camera_blueprint(self):
        """
        Returns camera blueprint.
        """
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(VIEW_WIDTH))
        camera_bp.set_attribute("image_size_y", str(VIEW_HEIGHT))
        camera_bp.set_attribute("fov", str(VIEW_FOV))
        return camera_bp

    def camera_listener(self, image):
        """
        Camera listener callback
        """
        # self.image = image
        image.convert(cc.Raw)
        """

        if(not self.image_saved):
            image.save_to_disk('cameras/cam_view_'+str(self.traffic_light_id)+'_%08d' % image.frame_number)
            self.image_saved=True"""

    def camera2_listener(self, image):
        """
        Camera2 listener callback
        """

        self.image = image
        image.convert(cc.Raw)
        if not self.image2_saved:
            image.save_to_disk(
                "./../data/HDMaps/c_"
                + str(self.intersection_id).zfill(3)
                + "/bev_frame.jpg"
            )
            self.image2_saved = True

    def get_homography(self):
        # U1= [K [R1|t1]@[R2|t2]^-1 @ K^-1] U2 => U1=H@U2
        camera1 = self.camera
        camera2 = self.camera2
        T1 = camera1.get_transform()
        T2 = camera2.get_transform()
        Tmat1 = ClientSideBoundingBoxes.get_matrix(T1)[:3]
        Tmat2 = ClientSideBoundingBoxes.get_matrix(T2)[:3]  # 3x3
        Tmat1 = np.delete(Tmat1, 2, 1)
        Tmat2 = np.delete(Tmat2, 2, 1)
        # print(Tmat1)
        # print(Tmat2)
        K = self.calibration  # 3x3
        # print(K.shape)
        H = K @ Tmat1 @ np.linalg.inv(Tmat2) @ np.linalg.inv(K)
        return H  # 3x3

    def setup_camera(self):
        # Spawn a camera on a random traffic light
        camera_bp = self.get_camera_blueprint()

        num = self.traffic_light_id  # duh! TODO
        traffic_light = self.get_traffic_light_by_id(num)
        # print(traffic_light.get_location().distance(self.get_traffic_light_by_id(56).get_location()))
        # print(traffic_light.get_location(),self.get_traffic_light_by_id(56).get_location())

        rel_transform = carla.Transform(
            carla.Location(x=-5, y=2, z=7), carla.Rotation(pitch=-10, roll=0, yaw=130)
        )
        self.camera = self.world.spawn_actor(
            camera_bp, rel_transform, attach_to=traffic_light
        )

        rel_transform2 = carla.Transform(
            carla.Location(x=-30, y=50, z=100),
            carla.Rotation(pitch=-90, roll=180, yaw=0),
        )

        self.camera2 = self.world.spawn_actor(
            camera_bp, rel_transform2, attach_to=traffic_light
        )
        sleep(0.5)
        T = ClientSideBoundingBoxes.get_matrix(self.camera2.get_transform())

        # embed()
        weak_self = weakref.ref(self)
        self.camera.listen(self.camera_listener)
        self.camera2.listen(self.camera2_listener)

        self.calibration = self.compute_intrinsics()
        self.H = self.get_homography()
        print(self.H)
        print(np.linalg.inv(self.H))

    def compute_intrinsics(self):
        """
        Computes intrinsics and returns a 3x3 K intrinsics matrix
        """
        ImageSizeX = int(self.camera.attributes["image_size_x"])
        ImageSizeY = int(self.camera.attributes["image_size_y"])
        camFOV = float(self.camera.attributes["fov"])

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

        # control = car.get_control()
        # control.throttle = 0
        # if keys[K_w]:
        #     control.throttle = 1
        #     control.reverse = False
        # elif keys[K_s]:
        #     control.throttle = 1
        #     control.reverse = True
        # if keys[K_a]:
        #     control.steer = max(-1.0, min(control.steer - 0.05, 0))
        # elif keys[K_d]:
        #     control.steer = min(1.0, max(control.steer + 0.05, 0))
        # else:
        #     control.steer = 0
        # control.hand_brake = keys[K_SPACE]

        # car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

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

    def game_loop(self):
        """
        Main program loop.
        """
        ## For each iteration, save the image and bounding boxes(xmin,ymin,xmax,ymax,classname)

        try:
            pygame.init()

            self.client = carla.Client("127.0.0.1", 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            self.weather = self.get_weather()
            # print(self.weather)
            self.world.set_weather(self.weather)

            # self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode(
                (VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            all_actors = self.world.get_actors()
            # embed()
            vehicles = all_actors.filter("vehicle.*")
            peds = all_actors.filter("walker.pedestrian.*")
            # lights = all_actors.filter("*traffic_light*") # no boxes for lights :(
            # # TODO: detect the traffic light id in the current view and store the state.

            filtered_agents = [v for v in vehicles] + [
                p for p in peds
            ]  # + [l for l in lights]

            # TODO: Add random scenario generator here

            scenario = "test1"
            # output_folder = f'output_data/{scenario}' # only python3.6
            output_folder = "output_data/" + scenario
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            SAVE_DATA = True
            while True:
                self.world.tick()
                snap = self.world.get_snapshot()
                frameid = snap.timestamp.frame
                timestamp = int(snap.timestamp.elapsed_seconds * 1000)  # milliseconds

                self.capture = True
                pygame_clock.tick_busy_loop(20)

                self.render(self.display)
                bounding_boxes, metadata = ClientSideBoundingBoxes.get_bounding_boxes(
                    filtered_agents, self.camera, self.calibration
                )
                final_metadata = ClientSideBoundingBoxes.draw_2d_bounding_boxes(
                    self.display, bounding_boxes, metadata
                )

                # save boxes here
                if SAVE_DATA:
                    # filename = f'{output_folder}/{frameid}_{timestamp}' #py36
                    filename = output_folder + "/" + str(frameid) + "_" + str(timestamp)
                    self.image.save_to_disk(filename)
                    with open(filename + ".json", "w") as f:
                        json.dump(final_metadata, f)
                pygame.display.flip()

                pygame.event.pump()
                if self.control(self.car):
                    return
                break

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            # self.car.destroy()
            pygame.quit()


def get_bev(intersection_id):
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    try:
        client = BasicSynchronousClient(intersection_id)
        client.game_loop()
    finally:
        print("EXIT")


if __name__ == "__main__":
    intersection_id = 2
    get_bev(intersection_id)
