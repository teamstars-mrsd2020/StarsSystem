import carla
import glob
import os
import sys
import time
import numpy as np
import argparse
import logging
import random
import json
import math


class test_tm_params:
    def __init__(self, carla_world, traffic_manager):

        self.world = carla_world
        self.tm = traffic_manager
        self.map = self.world.get_map()

        # self.world.on_tick(self.update_observed_params)

        # variables related to jumping traffic light
        self.number_of_lights_jumped = 0
        self.number_of_red_lights_encountered = 0
        self.vehicles_in_transition_set = set()
        self.stopped_vehicles_set = set()
        self.observed_percentage_jumping_light = 0

        # variables related to lead vehicle distance
        self.min_lvd_pair = ()
        self.proximity_vehicle_threshold = 10.0
        self.vehicles_near_junction = []
        self.least_lvd = 1000

    def update_percentage_running_light(self):

        vehicle_list = self.world.get_actors().filter("vehicle*")
        for v in vehicle_list:
            # if vehicle is already added to at_junction group, check if it is still in junction
            # if yes, then skip this vehicle
            # if no, it  means it moved out of junction and hence remove the vehicle from set
            if v.id in self.vehicles_in_transition_set:
                if not (
                    self.world.get_map().get_waypoint(v.get_location()).get_junction()
                ):
                    self.vehicles_in_transition_set.remove(v.id)
                    print("Popping vehicle from transition set")
                else:
                    continue

            # check if vehicle at traffic light
            if v.is_at_traffic_light():
                traffic_light = v.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    # check if the vehicle in the junction, which means traffic light jumped
                    vehicle_wp = self.world.get_map().get_waypoint(v.get_location())
                    if vehicle_wp.get_junction():
                        self.number_of_lights_jumped += 1
                        self.number_of_red_lights_encountered += 1
                        self.vehicles_in_transition_set.add(v.id)
                        print("Inside junction")
                        print(v.type_id, ": broke traffic light")
                        # spectator = self.world.get_spectator()
                        # world_snapshot = self.world.get_snapshot()
                        # actor_snapshot = world_snapshot.find(v.id)
                        # spectator_transform = vehicle_wp.previous(15)[0].transform
                        # # spectator_transform = carla.Transform(spectator_transform.location + carla.Location(z = 10), carla.Rotation(pitch=-45))
                        # spectator_transform.location += carla.Location(x=0, y=0, z=3.0)
                        # spectator.set_transform(spectator_transform)
                    else:
                        if v.id in self.stopped_vehicles_set:
                            continue
                        # check if vehicle first in lane and adhering to red light
                        vel = v.get_velocity()
                        if np.sqrt(vel.x ** 2 + vel.y ** 2) < 0.05:

                            # blue = carla.Color(0,0,255)
                            # red = carla.Color(255,0,0)
                            # green = carla.Color(0,255,0)
                            # white = carla.Color(0,0,0)

                            # debug = world.debug
                            # draw_waypoint_union(debug, vehicle_wp, vehicle_wp.next(1)[0], red if vehicle_wp.next(1)[0].is_junction else blue, 60)
                            # vector = v.get_velocity()
                            # debug.draw_string(vehicle_wp.transform.location, str('%15.0f km/h' % (3.6 * np.sqrt(vector.x**2 + vector.y**2 + vector.z**2))), False, green, 60)
                            # draw_transform(debug, vehicle_wp.transform, white, 60)

                            self.stopped_vehicles_set.add(v.id)
                            self.number_of_red_lights_encountered += 1
                            print("Vehicle stopped: ", v.id)
                    # print("Number of red light encounters: ", number_of_red_light_encounters)

            if v.id in self.stopped_vehicles_set:
                if v.id in self.vehicles_in_transition_set:
                    self.stopped_vehicles_set.remove(v.id)
                    print("Popping vehicle from stopped set: ", v.id)

        if self.number_of_red_lights_encountered == 0:
            self.observed_percentage_jumping_light = 0
        else:
            self.observed_percentage_jumping_light = (
                self.number_of_lights_jumped / self.number_of_red_lights_encountered
            ) * 100

    def is_within_distance_ahead(
        self, target_transform, current_transform, max_distance
    ):
        """
        Check if a target object is within a certain distance in front of a reference object.
        :param target_transform: location of the target object
        :param current_transform: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :return: True if target object is within max_distance ahead of the reference object
        """
        target_vector = np.array(
            [
                target_transform.location.x - current_transform.location.x,
                target_transform.location.y - current_transform.location.y,
            ]
        )
        norm_target = np.linalg.norm(target_vector)

        # If the vector is too short, we can simply stop here
        if norm_target < 0.001:
            return True, norm_target

        if norm_target > max_distance:
            return False, None

        fwd = current_transform.get_forward_vector()
        forward_vector = np.array([fwd.x, fwd.y])
        d_angle = math.degrees(
            math.acos(
                np.clip(np.dot(forward_vector, target_vector) / norm_target, -1.0, 1.0)
            )
        )

        if d_angle < 90.0:
            return True, norm_target

        return False, None

    def _is_vehicle_hazard(self, ego_vehicle, vehicle_list):
        """
        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = ego_vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == ego_vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self.map.get_waypoint(
                target_vehicle.get_location()
            )
            if (
                target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id
                or target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id
            ):
                continue

            is_lead, dist = self.is_within_distance_ahead(
                target_vehicle.get_transform(),
                ego_vehicle.get_transform(),
                self.proximity_vehicle_threshold,
            )

            if is_lead:
                if dist < self.least_lvd:
                    self.least_lvd = dist

                    # target vehicle is leader, ego vehicle is follower
                    self.min_lvd_pair = (target_vehicle.id, ego_vehicle.id)
                    # uncomment to see where the minimum distance is recorded
                    # spectator = self.world.get_spectator()
                    # world_snapshot = self.world.get_snapshot()
                    # actor_snapshot = world_snapshot.find(ego_vehicle.id)
                    # spectator_transform = ego_vehicle_waypoint.transform
                    # spectator_transform = carla.Transform(spectator_transform.location + carla.Location(z=50), carla.Rotation(pitch=-90))
                    # # spectator_transform.location += carla.Location(x=0, y=0, z=3.0)
                    # spectator.set_transform(spectator_transform)

    def get_vehicles_near_junction(self):
        def dist(tfl, v):
            return tfl.get_location().distance(v.get_location())

        for v in self.vehicle_list:
            for tfl in self.traffic_light_list:
                if dist(tfl, v) < 40:
                    self.vehicles_near_junction.append(v)
                    continue

    def update_lvd(self):
        def dist(v, ego_v):
            return v.get_location().distance(ego_v.get_location())

        for vehicle in self.vehicle_list:
            # filter vehicles which are closer to this vehicle
            filtered_vehicle_list = [
                v
                for v in self.vehicle_list
                if dist(v, vehicle) < 45 and v.id != vehicle.id
            ]
            # check if nearby vehicle is an obstacle ahead, then update the minimum distance
            self._is_vehicle_hazard(vehicle, filtered_vehicle_list)

    def update_observed_params(self):
        self.min_lvd_pair = ()
        self.traffic_light_list = self.world.get_actors().filter(
            "traffic.traffic_light"
        )
        self.vehicle_list = self.world.get_actors().filter("*vehicle*")
        self.update_percentage_running_light()
        self.update_lvd()

        return self.vehicles_in_transition_set, self.min_lvd_pair


if __name__ == "__main__":

    # update the behavioral parameters in config file
    config_file = "./config.json"

    config = {}
    # ----- Load config file ------------------
    if os.path.isfile(config_file):
        config = json.load(open(config_file))
    else:
        print("Error: Please provide the config file")

    try:
        carla_params = config["carla_params"]
        tm_params = config["tm_params"]

        client = carla.Client(carla_params["host"], carla_params["port"])
        client.set_timeout(2.0)

        world = client.get_world()
        tm = client.get_trafficmanager(tm_params["tm_port"])

        test_tm_params(world, tm)

    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")
