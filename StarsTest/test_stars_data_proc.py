# Inputs:
#   Video
#   HDMap
#   Intersection Config
#
# Outputs:
#   Trajectories
#   Params = {"LVD": x.x, "TLV": y.y}
#
# Process:
#   Get Trajectory
#   Get TL Status
#   Get TLV
#   Get LVD
#   Compare with GT if type=="carla"

import pytest
import json
import os
import pandas as pd
import sys
import cv2

sys.path.append("../")
sys.path.append("../StarsDataProcessing/detection_tracking/")


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


def combine(file1, file2):
    traj_1 = pd.read_csv(file1)
    traj_2 = pd.read_csv(file2)

    combined = pd.concat([traj_1, traj_2], axis=0, join="inner")

    return combined


# @pytest.mark.incremental
class TestSTARSDataProc:
    def test_get_conf(self, run):
        assert get_config(run) is not None
        print(get_config(run))

    def test_exist_video(self, run):
        video_path = get_config(run)["video"]
        assert check_file_and_size(video_path)

    def test_exist_HDMap(self, run):
        hdmap_path = get_config(run)["HD_map"]
        assert check_file_and_size(hdmap_path + "/annotations.starsjson")
        assert check_file_and_size(
            hdmap_path + "/bev_frame.jpg"
        )  # or check_file_and_size(hdmap_path + "/bev_frame.png")

    def test_exist_iconf(self, run):
        iconfig_path = get_config(run)["intersection_config"]
        assert check_file_and_size(iconfig_path)

    def test_get_trajectory(self, run):

        from StarsDataProcessing.detection_tracking.detection_tracking import (
            detection_tracking,
        )

        sys.path.append("../StarsDataProcessing/detection_tracking")
        cfg = get_config(run)

        detection_tracking(cfg)
        assert check_file_and_size(cfg["trajectory_pred"])

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

    def test_get_TL_status(self, run):
        cfg = get_config(run)
        video_path = cfg["video"]

        csv_save_path = cfg["TL_status"]
        intersection_cfg_path = cfg["intersection_config"]
        intersection_cfg = get_config_file(intersection_cfg_path)
        bboxes = intersection_cfg["bboxes"]

        # Main Function Under Test
        from StarsDataProcessing.traffic_light_classification.infer import get_TL_status

        # if not check_file_and_size(csv_save_path):
        get_TL_status(video_path, bboxes, csv_save_path, second_TL=True, debug=False)

        assert check_file_and_size(csv_save_path)

        # if int(run) >= 20:
        video_path = cfg["video_2"]
        csv_save_path = cfg["TL_status_2"]
        intersection_cfg_path = cfg["intersection_config_2"]
        intersection_cfg = get_config_file(intersection_cfg_path)
        bboxes = intersection_cfg["bboxes"]
        get_TL_status(video_path, bboxes, csv_save_path, debug=False)

        assert check_file_and_size(csv_save_path)
        # combine both of them
        traj_1 = pd.read_csv(cfg["TL_status"])
        traj_2 = pd.read_csv(cfg["TL_status_2"])

        len1 = traj_1.shape[0]
        len2 = traj_2.shape[0]

        len = min(len1, len2)

        # traj_1[:len]

        df = pd.concat([traj_1[:len], traj_2[:len]], axis=0, join="inner", ignore_index = True)
        # df = combine(cfg["TL_status"], cfg["TL_status_2"])
        df.to_csv(cfg["TL_status_combined"], index=False)

        assert check_file_and_size(cfg["TL_status_combined"])

    def test_get_TLV(self, run):
        cfg = get_config(run)
        from StarsDataProcessing.tl_violation.traffic_violation_copy import tlv

        cfg["TL_status"] = cfg["TL_status_combined"]
        cfg["trajectory_pred"] = cfg["trajectory_pred_combined"]
        file_name = os.path.splitext(os.path.split(cfg["trajectory_pred_combined"])[1])[
            0
        ]
        fps = cfg["fps"]

        num, denom = tlv(cfg, fps, "../data/extras/tlv_debug/"+file_name)
        if check_file_and_size(cfg["params_pred"]):
            with open(cfg["params_pred"], "r") as f:
                params = json.load(f)
        else:
            params = {}

        if denom != 0:
            params["TLV"] = 100 * num / denom
        else:
            params["TLV"] = 0.0

        with open(cfg["params_pred"], "w") as f:
            json.dump(params, f, indent=4)

        assert check_file_and_size(cfg["params_pred"])
        assert "TLV" in get_config_file(cfg["params_pred"])

    def test_get_LVD(self, run):
        # read traj csv
        cfg = get_config(run)
        intersection_cfg_path = cfg["intersection_config"]
        intersection_cfg = get_config_file(intersection_cfg_path)
        intersection_type = intersection_cfg["type"]

        if intersection_type == "carla":
            scaling_factor = 9.5
        else:
            scaling_factor = 8.9
        df = pd.read_csv(cfg["trajectory_pred_combined"])
        file_name = os.path.splitext(os.path.split(cfg["trajectory_pred_combined"])[1])[
            0
        ]
        lvd_debug_file = f"../data/extras/lvd_debug/{file_name}"
        # df = pd.read_csv(cfg["trajectory_pred"])
        assert "lane_id" in df.columns, "Missing lane_id column in trajectory_preds"
        from StarsDataProcessing import extract_params

        bev_image = cv2.imread(cfg["HD_map"] + "/bev_frame.jpg")
        # lvd_stats = extract_params.get_global_stats(df)
        fps = cfg["fps"]
        lvd_stats = extract_params.get_global_stats_with_visualization(
            df, bev_image, scaling_factor, fps, lvd_debug_file
        )
        if check_file_and_size(cfg["params_pred"]):
            with open(cfg["params_pred"]) as f:
                params = json.load(f)
        else:
            params = {}
        params["LVD"] = {}
        params["LVD"]["min_lvd_global"] = lvd_stats["min_lvd_global"] / scaling_factor
        params["LVD"]["mean_lvd_global"] = lvd_stats["mean_lvd_global"] / scaling_factor
        with open(cfg["params_pred"], "w") as f:
            json.dump(params, f)
        params_str = json.dumps(params,indent=2)
        print("-------- Extracted Params --------")
        print(params_str)
        print("-------- ---------------- --------")
        assert "LVD" in get_config_file(cfg["params_pred"])

    def test_trajectory_evaluation(self, run):
        cfg = get_config(run)
        intersection_cfg_path = cfg["intersection_config"]
        intersection_cfg = get_config_file(intersection_cfg_path)
        intersection_type = intersection_cfg["type"]

        if intersection_type == "carla":
            pass

    def test_validate_params_wrt_gt(self, run):
        cfg = get_config(run)
        intersection_cfg_path = cfg["intersection_config"]
        intersection_cfg = get_config_file(intersection_cfg_path)
        intersection_type = intersection_cfg["type"]

        if intersection_type == "carla":
            # verify the gt params & pred params
            params_pred = get_config_file(cfg["params_pred"])
            params_gt = get_config_file(cfg["params_gt"])

            eps = 0.00001
            dist_param = "min_lvd_global"
            if params_gt["TLV"] != 0:
                diff_tlv1 = (params_pred["TLV"] - params_gt["TLV"]) / params_gt["TLV"]
            else:
                diff_tlv1 = (params_pred["TLV"] - params_gt["TLV"] - eps) / (
                    params_gt["TLV"] + eps
                )
            if params_pred["TLV"] != 0:
                diff_tlv2 = (params_gt["TLV"] - params_pred["TLV"]) / params_pred["TLV"]
            else:
                diff_tlv2 = (params_gt["TLV"] - params_pred["TLV"] - eps) / (
                    params_pred["TLV"] + eps
                )

            if params_gt["LVD"] != 0:
                diff_lvd = (
                    params_pred["LVD"][dist_param] - params_gt["LVD"]
                ) / params_gt["LVD"]
            else:
                diff_lvd = (params_pred["LVD"][dist_param] - params_gt["LVD"] - eps) / (
                    params_gt["LVD"] + eps
                )

            print(diff_lvd)
            assert min(abs(diff_tlv1), abs(diff_tlv2)) <= 0.2
            assert abs(diff_lvd) <= 0.2
        assert True