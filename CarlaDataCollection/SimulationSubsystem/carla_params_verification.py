#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import time
import numpy as np

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


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

import carla

from carla import VehicleLightState as vls
from CarlaDataCollection.SimulationSubsystem.demo_view import HUD
from CarlaDataCollection.SimulationSubsystem.demo_view import CameraManager

import argparse
import logging
import random
import json
import math

from .test_tm_params import test_tm_params


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def draw_transform(debug, trans, col=carla.Color(255, 0, 0), lt=-1):
    debug.draw_arrow(
        trans.location,
        trans.location + trans.get_forward_vector(),
        thickness=0.05,
        arrow_size=0.1,
        color=col,
        life_time=lt,
    )


def draw_waypoint_union(debug, w0, w1, color=carla.Color(255, 0, 0), lt=5):
    debug.draw_line(
        w0.transform.location + carla.Location(z=0.25),
        w1.transform.location + carla.Location(z=0.25),
        thickness=0.1,
        color=color,
        life_time=lt,
        persistent_lines=False,
    )
    debug.draw_point(
        w1.transform.location + carla.Location(z=0.25), 0.1, color, lt, False
    )


def highlight_danger_actors(world, actor_ids):

    debug = world.debug
    for id in actor_ids:
        actor = world.get_actor(id)
        color = carla.Color(255, 0, 0)
        lt = 1  # life-time
        debug.draw_point(
            actor.get_location() + carla.Location(z=0.25), 0.1, color, lt, False
        )


class CarlaParamsVerifiction:
    def __init__(self, config, save_frames):

        self.config = config
        self.save_frames = save_frames
        self.world = None

    def highlight_traffic_lights(self):

        # highlight traffic lights
        debug = self.world.debug
        traffic_light_list = self.world.get_actors().filter("traffic.traffic_light")
        for tf in traffic_light_list:
            color = carla.Color(255, 0, 0)
            lt = 1  # life-time
            debug.draw_point(
                tf.get_location() + carla.Location(z=0.25), 0.1, color, lt, False
            )

    def highlight_display(self, tfl_violators, lvd_pair):

        # highlight traffic light violators
        debug = self.world.debug
        for id in tfl_violators:
            actor = self.world.get_actor(id)
            color = carla.Color(255, 0, 0)
            lt = 1  # life-time
            debug.draw_point(
                actor.get_location() + carla.Location(z=0.25), 0.1, color, lt, False
            )

        if len(lvd_pair):
            # highlight lvd pairs
            leader = self.world.get_actor(lvd_pair[0])
            follower = self.world.get_actor(lvd_pair[1])
            color = carla.Color(0, 0, 255)
            lt = 1  # life-time

            debug.draw_arrow(
                follower.get_location() + carla.Location(z=0.25),
                leader.get_location() + carla.Location(z=0.25),
                thickness=2,
                arrow_size=2,
                color=color,
                life_time=lt,
            )

    def run(self):
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

        vehicles_list = []
        walkers_list = []
        all_id = []
        client = carla.Client(self.config["host"], self.config["port"])
        client.set_timeout(10.0)
        synchronous_master = False
        ticks = 0
        try:
            self.world = client.load_world(self.config["town"])
            self.world.set_weather(
                getattr(carla.WeatherParameters, self.config["weather"])
            )

            tm = client.get_trafficmanager(self.config["tm_port"])

            test_tm = test_tm_params(self.world, tm)

            simulation_time = self.config["simulation_time"]

            tm.set_global_distance_to_leading_vehicle(
                self.config["global_distance_to_leading_vehicle"]
            )
            tm.global_percentage_speed_difference(
                self.config["global_percentage_speed_difference"]
            )

            if self.config["no_render"]:
                settings = self.world.get_settings()
                settings.no_rendering_mode = True
                self.world.apply_settings(settings)

            if self.config["hybrid"]:
                tm.set_hybrid_physics_mode(True)

            if self.config["sync"]:
                settings = self.world.get_settings()
                tm.set_synchronous_mode(True)
                if not settings.synchronous_mode:
                    synchronous_master = True
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                    self.world.apply_settings(settings)
                else:
                    synchronous_master = False

            blueprints = self.world.get_blueprint_library().filter("vehicle.*")
            blueprints = [
                x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
            ]
            blueprintsWalkers = self.world.get_blueprint_library().filter(
                "walker.pedestrian.*"
            )

            spawn_points = self.world.get_map().get_spawn_points()
            number_of_spawn_points = len(spawn_points)

            if self.config["number_of_vehicles"] < number_of_spawn_points:
                random.shuffle(spawn_points)
            elif self.config["number_of_vehicles"] > number_of_spawn_points:
                msg = "requested %d vehicles, but could only find %d spawn points"
                logging.warning(
                    msg, self.config["number_of_vehicles"], number_of_spawn_points
                )
                self.config["number_of_vehicles"] = number_of_spawn_points

            # @todo cannot import these directly.
            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            SetVehicleLightState = carla.command.SetVehicleLightState
            FutureActor = carla.command.FutureActor

            # --------------
            # Spawn vehicles
            # --------------
            batch = []
            for n, transform in enumerate(spawn_points):
                if n >= self.config["number_of_vehicles"]:
                    break
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute("color"):
                    color = random.choice(
                        blueprint.get_attribute("color").recommended_values
                    )
                    blueprint.set_attribute("color", color)
                if blueprint.has_attribute("driver_id"):
                    driver_id = random.choice(
                        blueprint.get_attribute("driver_id").recommended_values
                    )
                    blueprint.set_attribute("driver_id", driver_id)
                blueprint.set_attribute("role_name", "autopilot")

                # prepare the light state of the cars to spawn
                light_state = vls.NONE

                # spawn the cars and set their autopilot and light state all together
                batch.append(
                    SpawnActor(blueprint, transform)
                    .then(SetAutopilot(FutureActor, True, tm.get_port()))
                    .then(SetVehicleLightState(FutureActor, light_state))
                )

            for response in client.apply_batch_sync(batch, synchronous_master):
                if response.error:
                    logging.error(response.error)
                else:
                    vehicles_list.append(response.actor_id)

            # --------------- set danger car params ---------------------------------
            actor_list = self.world.get_actors().filter("vehicle.*")
            print(actor_list)
            number_of_danger_vehicles = int(
                ((self.config["percentageTrafficBreakingLight"]) / 100)
                * self.config["number_of_vehicles"]
            )
            danger_vehicles = np.random.choice(
                self.config["number_of_vehicles"],
                number_of_danger_vehicles,
                replace=False,
            )
            for i in range(number_of_danger_vehicles):
                print(
                    "Setting jumping light percentage to ",
                    float(self.config["ignore_lights_percentage"]),
                    " for vehicle: ",
                    actor_list[int(danger_vehicles[i])].type_id,
                )
                tm.ignore_lights_percentage(
                    actor_list[int(danger_vehicles[i])],
                    float(self.config["ignore_lights_percentage"]),
                )
                # tm.distance_to_leading_vehicle(actor_list[int(danger_vehicles[i])],0)
                # tm.vehicle_percentage_speed_difference(actor_list[int(danger_vehicles[i])],config["vehicle_percentage_speed_difference"])

            # -------------
            # Spawn Walkers
            # -------------
            # some settings
            percentagePedestriansRunning = self.config[
                "percentagePedestriansRunning"
            ]  # how many pedestrians will run
            percentagePedestriansCrossing = self.config[
                "percentagePedestriansCrossing"
            ]  # how many pedestrians will walk through the road
            # 1. take all the random locations to spawn
            spawn_points = []
            for i in range(self.config["number_of_walkers"]):
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc != None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)
            # 2. we spawn the walker object
            batch = []
            walker_speed = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprintsWalkers)
                # set as not invincible
                if walker_bp.has_attribute("is_invincible"):
                    walker_bp.set_attribute("is_invincible", "false")
                # set the max speed
                if walker_bp.has_attribute("speed"):
                    if random.random() > percentagePedestriansRunning:
                        # walking
                        walker_speed.append(
                            walker_bp.get_attribute("speed").recommended_values[1]
                        )
                    else:
                        # running
                        walker_speed.append(
                            walker_bp.get_attribute("speed").recommended_values[2]
                        )
                else:
                    print("Walker has no speed")
                    walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_point))
            results = client.apply_batch_sync(batch, True)
            walker_speed2 = []
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list.append({"id": results[i].actor_id})
                    walker_speed2.append(walker_speed[i])
            walker_speed = walker_speed2
            # 3. we spawn the walker controller
            batch = []
            walker_controller_bp = self.world.get_blueprint_library().find(
                "controller.ai.walker"
            )
            for i in range(len(walkers_list)):
                batch.append(
                    SpawnActor(
                        walker_controller_bp, carla.Transform(), walkers_list[i]["id"]
                    )
                )
            results = client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list[i]["con"] = results[i].actor_id
            # 4. we put altogether the walkers and controllers id to get the objects from their id
            for i in range(len(walkers_list)):
                all_id.append(walkers_list[i]["con"])
                all_id.append(walkers_list[i]["id"])
            all_actors = self.world.get_actors(all_id)

            # wait for a tick to ensure client receives the last transform of the walkers we have just created
            if not self.config["sync"] or not synchronous_master:
                self.world.wait_for_tick()
            else:
                self.world.tick()

            # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
            # set how many pedestrians can cross the road
            self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
            for i in range(0, len(all_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation()
                )
                # max speed
                all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

            print(
                "spawned %d vehicles and %d walkers, press Ctrl+C to exit."
                % (len(vehicles_list), len(walkers_list))
            )

            # example of how to use parameters
            # tm.global_percentage_speed_difference(30.0)
            pygame.init()
            pygame.font.init()

            display = pygame.display.set_mode(
                (1080, 720), pygame.HWSURFACE | pygame.DOUBLEBUF
            )

            time.sleep(1)

            hud = HUD(1080, 720)
            camera_manager = CameraManager(hud, 2.2, self.world)
            camera_manager.transform_index = 0
            camera_manager.set_sensor(0, notify=False)
            self.world.on_tick(hud.on_world_tick)

            clock = pygame.time.Clock()

            # spectator = self.world.get_spectator()
            # world_snapshot = self.world.get_snapshot()
            # spectator.set_transform(
            #     carla.Transform(
            #         carla.Location(0.0, 0.0, 50.0), carla.Rotation(-90.0, 90.0, 0.0)
            #     )
            # )
            frame_num = 0
            image_num = 0
            while True:
                frame_num += 1
                clock.tick_busy_loop(60)
                hud.tick(self.world, clock)
                tfl_violators, lvd_pair = test_tm.update_observed_params()
                # test_tm.update_observed_params()
                self.highlight_display(tfl_violators, lvd_pair)
                lvd = test_tm.least_lvd
                tlv = test_tm.observed_percentage_jumping_light
                # tlv = 0
                # lvd = 0
                print("Current percentage for jumping red light: ", tlv)
                print("Current minimum distance between leading vehicle: ", lvd)
                camera_manager.render(display)
                hud.render(display, tlv, lvd)
                pygame.display.flip()

                if self.save_frames:
                    if frame_num % 3:
                        image_num += 1
                        filename = (
                            "/home/stars/StarsSystem/CarlaDataCollection/SimulationSubsystem/Test4/%03d.png"
                            % image_num
                        )
                        pygame.image.save(display, filename)

                world_snapshot = self.world.get_snapshot()
                print("Elapsed Time: ", world_snapshot.timestamp.elapsed_seconds)
                if world_snapshot.timestamp.elapsed_seconds > simulation_time:
                    break
                ticks += 1
                if self.config["sync"] and synchronous_master:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()

        finally:

            if self.config["sync"] and synchronous_master:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)

            world_snapshot = self.world.get_snapshot()
            # print("Elapsed Time: ", world_snapshot.timestamp.elapsed_seconds)
            # print("Total number of red light encounters: ", number_of_red_light_encounters)
            # print("Total number of red ligths jumped: ", number_of_lights_jumped)
            # print('\ndestroying %d vehicles' % len(vehicles_list))
            # print("Total ticks: ", ticks)
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

            # stop walker controllers (list is [controller, actor, controller, actor ...])
            for i in range(0, len(all_id), 2):
                all_actors[i].stop()

            print("\ndestroying %d walkers" % len(walkers_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

            time.sleep(0.5)

            return test_tm.observed_percentage_jumping_light, test_tm.least_lvd


if __name__ == "__main__":

    # update the behavioral parameters in config file
    carla_config_file = "../../data/configs/carla_config.json"
    behavior_config_file = "../../data/configs/behavior_config/moderate_town.json"

    config_data = {}
    # ----- Load config file ------------------
    if os.path.isfile(carla_config_file):
        carla_config_data = json.load(open(carla_config_file))
    else:
        print("Error: Please provide the carla config file")
        sys.exit()

    # ----- Load config file ------------------
    if os.path.isfile(behavior_config_file):
        behavior_config_data = json.load(open(behavior_config_file))
    else:
        print("Error: Please provide the behavior config file")
        sys.exit()

    config_data.update(carla_config_data)
    config_data.update(behavior_config_data)

    try:
        carla_params_verification(config_data)
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")
