


def test_1(run):
    pass

# Test for Tracking (MOTA, MOTP)
def test_2(run):
    import json
    from test_stars_data_proc import get_config, check_file_and_size, get_config_file
    import sys
    sys.path.append("../")
    # Running Detection Tracking on video
    from StarsDataProcessing.detection_tracking.detection_tracking import detection_tracking
    cfg = get_config(run)
    detection_tracking(cfg)

# Test for Data Processing Verification (TLV, LVD)
def test_3(run):
    import json
    from test_stars_data_proc import get_config, check_file_and_size, get_config_file
    import sys
    sys.path.append("../")
    # Running Detection Tracking on video
    from StarsDataProcessing.detection_tracking.detection_tracking import detection_tracking
    cfg = get_config(run)
    detection_tracking(cfg)

    # Extracting params from trajectory files
    from StarsDataProcessing.tl_violation.traffic_violation import tlv
    from StarsDataProcessing import extract_params

    tlv = tlv(cfg)
    lvd_stats = extract_params.get_global_stats()

    if check_file_and_size(cfg['params_pred']):
        with open(cfg['params_pred']) as f:
            params = json.load(f)
    else:
        params = {}
    params['LVD'] = lvd_stats
    params["TLV"] = tlv
    
    os.remove(cfg["params_pred"])
    with open(cfg["params_pred"], 'w') as f:
        json.dump(params, f, indent=4)

    intersection_cfg_path = cfg["intersection_config"]
    intersection_cfg = get_config_file(intersection_cfg_path)
    intersection_type = intersection_cfg["type"]
    
    if intersection_type == "carla":
        params_pred = get_config_file(cfg["params_pred"])
        params_gt = get_config_file(cfg["params_gt"])

        dist_param = "min_lvd_global"
        
        if params_gt["TLV"] != 0:
            eps = 0
        else:
            eps = 0.001
        diff_tlv = (params_pred["TLV"] - params_gt["TLV"] - eps) / (params_gt["TLV"] + eps)

        if params_gt["LVD"] != 0:
            eps = 0
        else:
            eps = 0.001
        diff_lvd = (params_pred["LVD"][dist_param]/9.46 - params_gt["LVD"] - eps) / (params_gt["LVD"] + eps)
    
    print("TLV : ",diff_tlv, "LVD : ", diff_lvd)
    assert abs(diff_tlv) <= 0.2
    assert abs(diff_lvd) <= 0.2

# Test for Simulation Verification (TLV, LVD)
def test_4(run):

    import pytest
    import json 
    import os
    import pandas as pd
    import sys
    from test_simulation import get_config, check_file_and_size, get_config_file, get_change
    sys.path.append("../")

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

def test_5():
    assert 0

def test_6():
    pass

def test_7():
    assert 0

