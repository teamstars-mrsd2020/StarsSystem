from stars_data_capture_2 import multiple_rounds_main
from trajectory import create_trajectories_file
from birdview_homography import get_bev
import glob
from os import path
import json


def get_round_number(intersection_id, side_id):
    path = (
        "./../data/videos/c_"
        + str(intersection_id).zfill(3)
        + "_"
        + str(side_id).zfill(3)
        + "*.mp4"
    )
    files = glob.glob(path)
    filenames = []
    for file in files:
        rn = file[-7:-4]
        filenames.append(int(rn))
    filenames.sort()
    if len(filenames) == 0:
        return 1
    return filenames[-1] + 1


if __name__ == "__main__":
    """Records data from carla and stores GT"""

    run_id = 4

    with open("../data/runs/run_" + str(run_id) + ".json") as f:
        conf = json.load(f)
    multiple_rounds_main(conf)

    with open(conf["intersection_config"]) as f:
        intersection_1 = json.load(f)

    with open(conf["intersection_config_2"]) as f:
        intersection_2 = json.load(f)

    intersection_id = intersection_1["intersection_id"]
    round_ = int(conf["video"][-7:-4])
    side_id1 = intersection_1["side"]
    side_id2 = intersection_2["side"]

    create_trajectories_file(intersection_id, side_id1, round_)
    create_trajectories_file(intersection_id, side_id2, round_)

    # check if bev_image exists
    if not path.exists(
        "./../data/HDMaps/c_" + str(intersection_id).zfill(3) + "/bev_frame.png"
    ):
        get_bev(intersection_id)

    # TODO Call function to get ground truth params (TLV, LVD) from carla
