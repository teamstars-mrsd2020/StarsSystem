from StarsTest.test_data_collection import get_config
from test_stars_data_proc import *
import sys


def test_1(run):
    from StarsDataProcessing.detection_modelling import visualize_evaluation

    cfg = get_config(run)
    checkpoints_path = "../StarsDataProcessing/detection_modelling/final_training/output/model_final.pth"
    dataset_folder = cfg["detection_gt"]
    p_vehicle, r_vehicle = visualize_evaluation.run(
        dataset_folder,
        checkpoints_path,
        confidence_thres=0.8,
        iou_thres=0.5,
        show_visualization=True,
    )
    print(f"Final Precision: {p_vehicle}; Final Recall: {r_vehicle}")
    assert p_vehicle > 0.75
    assert r_vehicle > 0.75


# Test for Tracking (MOTA, MOTP)
def test_2(run):
    import json
    from test_stars_data_proc import get_config, check_file_and_size, get_config_file
    import sys

    sys.path.append("../")
    # Running Detection Tracking on video
    from StarsDataProcessing.detection_tracking.detection_tracking import (
        detection_tracking,
    )

    cfg = get_config(run)
    mota, motp = detection_tracking(cfg)

    intersection_cfg_path = cfg["intersection_config"]
    intersection_cfg = get_config_file(intersection_cfg_path)
    intersection_type = intersection_cfg["type"]

    if intersection_type == "carla":
        assert mota >= 0.4
        assert motp >= 0.4
    assert True


# Test for Data Processing Verification (TLV, LVD)
def test_3(run):
    import json
    from test_stars_data_proc import get_config, check_file_and_size, get_config_file
    import sys

    sys.path.append("../")
    # Running Detection Tracking on video
    from StarsDataProcessing.detection_tracking.detection_tracking import (
        detection_tracking,
    )

    cfg = get_config(run)
    detection_tracking(cfg)

    # Extracting params from trajectory files
    from StarsDataProcessing.tl_violation.traffic_violation import tlv
    from StarsDataProcessing import extract_params

    tlv = tlv(cfg)
    lvd_stats = extract_params.get_global_stats()

    if check_file_and_size(cfg["params_pred"]):
        with open(cfg["params_pred"]) as f:
            params = json.load(f)
    else:
        params = {}
    params["LVD"] = lvd_stats
    params["TLV"] = tlv

    os.remove(cfg["params_pred"])
    with open(cfg["params_pred"], "w") as f:
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
        diff_tlv = (params_pred["TLV"] - params_gt["TLV"] - eps) / (
            params_gt["TLV"] + eps
        )

        if params_gt["LVD"] != 0:
            eps = 0
        else:
            eps = 0.001
        diff_lvd = (params_pred["LVD"][dist_param] / 9.46 - params_gt["LVD"] - eps) / (
            params_gt["LVD"] + eps
        )

    print("TLV : ", diff_tlv, "LVD : ", diff_lvd)
    assert abs(diff_tlv) <= 0.2
    assert abs(diff_lvd) <= 0.2


# Test for Simulation Verification (TLV, LVD)
def test_4(run):

    import pytest
    import json
    import os
    import pandas as pd
    import sys
    from test_simulation import (
        get_config,
        check_file_and_size,
        get_config_file,
        get_change,
    )

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

    from CarlaDataCollection.SimulationSubsystem.carla_params_verification import (
        CarlaParamsVerifiction,
    )

    # Pass config to data collection functions, save the image frames
    save_frames = False
    observe_gt = True
    carla_params_verification = CarlaParamsVerifiction(
        config_data, observe_gt, save_frames
    )
    obs_paramTLJ, obs_paramLVD = carla_params_verification.run()

    gt_paramLVD = behavior_cfg["global_distance_to_leading_vehicle"]
    gt_paramTLJ = behavior_cfg["ignore_lights_percentage"]

    LVD_change = get_change(gt_paramLVD, obs_paramLVD)
    TLJ_change = get_change(gt_paramTLJ, obs_paramTLJ)
    LVD_tol = 100
    TLJ_tol = 20

    print("LVD change : ", LVD_change, "%")
    print("TLJ change : ", TLJ_change, "%")

    assert LVD_change <= LVD_tol or TLJ_change <= TLJ_tol


def test_5():
    assert 0


# Test for Simulation FPS
def test_6(run):
    import pytest
    import json
    import os
    import pandas as pd
    import sys
    from test_simulation import (
        get_config,
        check_file_and_size,
        get_config_file,
        get_change,
    )

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

    from CarlaDataCollection.SimulationSubsystem.carla_params_verification import (
        CarlaParamsVerifiction,
    )

    # Pass config to data collection functions, save the image frames
    save_frames = False
    observe_gt = False
    carla_params_verification = CarlaParamsVerifiction(
        config_data, observe_gt, save_frames
    )
    observed_fps = carla_params_verification.run()

    required_fps = 10

    print("FPS : ", observed_fps)

    assert observed_fps > required_fps


def test_7():
    assert 0


def test_fvd_demo1(run):
    # Carla
    test_obj = TestSTARSDataProc()
    cfg = get_config(run)

    if cfg["eval"]:
        # Get p,r for one side
        # test_1(run)
        # Get trajectory for one side
        # Get mota,motp for side 0 using GT
        test_2(run)
        # Get trajectory for other side
        from StarsDataProcessing.detection_tracking.detection_tracking import (
            detection_tracking,
        )

        sys.path.append("../StarsDataProcessing/detection_tracking")
        cfg["intersection_config"] = cfg["intersection_config_2"]
        cfg["video"] = cfg["video_2"]
        cfg["trajectory_pred"] = cfg["trajectory_pred_2"]
        detection_tracking(cfg)
        assert check_file_and_size(cfg["trajectory_pred"])

        # combine both
        cfg2 = get_config(run)

        traj_combined = combine(cfg2["trajectory_pred"], cfg2["trajectory_pred_2"])
        traj_combined.to_csv(cfg2["trajectory_pred_combined"], index=False)
        assert check_file_and_size(cfg2["trajectory_pred_combined"])

    else:
        # get trajectories
        test_obj.test_get_trajectory(run)

    # Get TL status for both
    test_obj.test_get_TL_status(run)

    # compute TLV, LVD using both sides
    test_obj.test_get_TLV(run)
    test_obj.test_get_LVD(run)

    if cfg["eval"]:
        # Compare params with GT
        test_obj.test_validate_params_wrt_gt(run)

    assert True


def test_fvd_demo2(run):

    test_obj = TestSTARSDataProc()
    test_obj.test_get_trajectory(run)
    test_obj.test_get_TL_status(run)
    test_obj.test_get_TLV(run)
    test_obj.test_get_LVD(run)

    assert True


def test_fvd_demo3_1(run):
    test_4(run)
    assert True


def test_fvd_demo3_2(run):
    test_6(run)
    assert True


def test_fvd_demo4():
    assert 0


# Test for Tracking (MOTA, MOTP)
def test_temp_2(run):
    import json
    from test_stars_data_proc import get_config, check_file_and_size, get_config_file
    import sys

    sys.path.append("../")
    # Running Detection Tracking on video
    from StarsDataProcessing.detection_tracking.detection_tracking_copy import (
        detection_tracking,
    )

    cfg = get_config(run)
    detection_tracking(cfg)
