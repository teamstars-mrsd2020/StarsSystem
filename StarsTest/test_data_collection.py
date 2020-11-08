import pytest
import json 
import os
import pandas as pd
import sys
sys.path.append("../")

def get_config(run):
    with open("../data/runs/run_" + str(run) + ".json") as f:
        conf = json.load(f)
    return conf

def get_config_file(file_path):
    with open(file_path) as f:
        conf = json.load(f)
    return conf

def check_file_and_size(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return True
    else:
        return False

def get_change(gt, obs):
    return (abs(gt - obs)/gt)*100


# @pytest.mark.incremental
class TestDataCollection:

    def test_get_conf(self, run):
        assert get_config(run) is not None
        print(get_config(run))

    def test_exist_behavior_config(self, run):
        b_config_path = get_config(run)["behavior_config"]
        assert check_file_and_size(b_config_path)

    def test_carla_params_verification(self, run):
        # Run Carla in the background
        
        # Read behavior config
        cfg = get_config(run)       

        behavior_cfg_path = cfg["behavior_config"]
        behavior_cfg = get_config_file(behavior_cfg_path)
        
        carla_cfg_path = cfg["carla_config"]
        carla_cfg = get_config_file(carla_cfg_path)
        print(carla_cfg)

        config_data = {} 
        config_data.update(carla_cfg)
        config_data.update(behavior_cfg)

        from CarlaDataCollection.SimulationSubsystem.carla_params_verification import carla_params_verification
        # Pass config to data collection functions
        obs_paramTLJ, obs_paramLVD = carla_params_verification(config_data)

        gt_paramLVD = behavior_cfg["global_distance_to_leading_vehicle"]
        gt_paramTLJ = behavior_cfg["ignore_lights_percentage"]
        
        LVD_change = get_change(gt_paramLVD, obs_paramLVD) 
        TLJ_change = get_change(gt_paramTLJ, obs_paramTLJ) 
        LVD_tol = 20
        TLJ_tol = 20

        print("LVD change : ", LVD_change , "%")
        print("TLJ change : ", TLJ_change, "%")

        assert (LVD_change <= LVD_tol or TLJ_change <= TLJ_tol)

    # def test_get_TL_status(self, run):
    #     cfg = get_config(run)
    #     video_path = cfg["video"]
        
    #     csv_save_path = cfg["TL_status"]
    #     intersection_cfg_path = cfg["intersection_config"]
    #     intersection_cfg = get_config_file(intersection_cfg_path)
    #     bboxes = intersection_cfg["bboxes"]

    #     # Main Function Under Test
    #     from STARSDataProcessing.traffic_light_classification.infer import get_TL_status
    #     # if not check_file_and_size(csv_save_path):
    #     get_TL_status(video_path, bboxes, csv_save_path, debug=False)


    #     assert check_file_and_size(csv_save_path)


