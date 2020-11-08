# import json 
# import os
# import pandas as pd
# import sys

# def get_config(run):
#     with open("../data/runs/run_" + str(run) + ".json") as f:
#         conf = json.load(f)
#     return conf

# def get_config_file(file_path):
#     with open(file_path) as f:
#         conf = json.load(f)
#     return conf

# def check_file_and_size(file_path):
#     if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
#         return True
#     else:
#         return False


# def test_get_TL_status( run):
#     # init module for TL Classification 
#     cfg = get_config(run)
#     video_path = cfg["video"]
    
#     csv_save_path = cfg["TL_status"]
#     intersection_cfg_path = cfg["intersection_config"]
#     intersection_cfg = get_config_file(intersection_cfg_path)
#     bboxes = intersection_cfg["bboxes"]

#     sys.path.append("../")
#     from STARSDataProcessing.traffic_light_classification.infer import get_TL_status

#     get_TL_status(video_path, bboxes, csv_save_path, debug=False)

#     assert check_file_and_size(csv_save_path)

# test_get_TL_status(1)