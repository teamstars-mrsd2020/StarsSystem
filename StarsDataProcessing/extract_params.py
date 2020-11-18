import glob
from typing import Tuple, List, Any

import pandas as pd
import numpy as np
import math
from IPython import embed
import copy
import cv2

# import parse_trajectories

ABS_POSITION_THRES = 200  # px
ABS_DIRECTION_THRES_DEGREE = (
    20  # allow 15 to -15 degrees variation for considering "same" direction
)
LATERAL_DISTANCE_THRESHOLD = 10  # px?
STATIONARY_VELOCITY = 1


def extract_speed_stats(agent_df: pd.DataFrame):
    """Calculate mean speeds, given the agent df(including x,y points),

    Args:
        df (pd.DataFrame): dataframe of all points for the agent

    Returns:
        [type]: [description]
    """
    agent_speeds = {
        "mean_vx": agent_df.vx.mean(),
        "mean_vy": agent_df.vy.mean(),
        "mean_abs_v": agent_df.abs_v.mean(),
        "mean_abs_a": agent_df.abs_a.mean(),
        "max_vx": agent_df.vx.max(),
        "max_vy": agent_df.vy.max(),
        "max_abs_v": agent_df.abs_v.max(),
        "max_abs_a": agent_df.abs_a.max(),
        "min_vx": agent_df.vx.min(),
        "min_vy": agent_df.vy.min(),
        "min_abs_v": agent_df.abs_v.min(),
        "min_abs_a": agent_df.abs_a.min(),
    }
    return agent_speeds


def get_trajectory_length(agent_df: pd.DataFrame):
    """Calculate the length of the trajectory, given the agent df(including x,y points),

    Args:
        agent_df (pd.DataFrame): dataframe of all points for the agent

    Returns:
        [type]: [description]
    """
    x = agent_df.x.values
    y = agent_df.y.values

    dist_array = (x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2

    return np.sum(np.sqrt(dist_array))

def get_global_stats_with_visualization(df: pd.DataFrame, bev_image, scaling_factor, fps, lvd_debug_file="LVD_Video"):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(f"{lvd_debug_file}.mp4", fourcc, fps, (bev_image.shape[1], bev_image.shape[0]))
    
    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 1
    TEXT_THICKNESS = 2

    groups = df.groupby("frame_id").groups
    frame_ids = sorted(list(groups.keys()))
    lvd_arr = []
    min_dist = 1000
    for frame_id in frame_ids:
        birdview = copy.deepcopy(bev_image)
        frame_df = df.iloc[groups[frame_id]]

        lvd_arr += calc_frame_level_lead_vehicles_dist(frame_df, frame_id)
        # print(lvd_arr)
        if len(lvd_arr)>0:
            lvd_df_temp = pd.DataFrame(
                lvd_arr,
                columns=[
                    "frame_id",
                    "agent_id",
                    "lead_vehicle_id",
                    "distance",
                    "agent_x",
                    "agent_y",
                    "other_x",
                    "other_y",
                ],
            )
            min_dist = lvd_df_temp.distance.min()#min(min_dist, lvd_arr[-1][3])
            lvd_df_temp = lvd_df_temp[lvd_df_temp.frame_id == frame_id]
        #     min_dist = min(lvd_arr)
            # print(min_dist)
            cv2.putText(
                birdview,
                "LVD: {0:.2f}".format(min_dist/scaling_factor) +"m",
                (900, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            # cv2.
            # cv2.line() 
            for idx,row in lvd_df_temp.iterrows():
                p1 = (int(row.agent_x),int(row.agent_y))
                p2 = (int(row.other_x),int(row.other_y))
                dist = row.distance
                if(dist==min_dist):
                    color = (0,0,255)
                    thickness = 3
                else:
                    color = (0,255,0)
                    thickness = 2
                cv2.line(birdview, p1, p2, color, thickness)

            # Draw green line for all vehicles
            # color the "min" line Red and make it thicker
        else:
            cv2.putText(
            birdview,
            "No pair detected",
            (800, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,)
        
        unique_agents = frame_df.agent_id.unique()
        for id in unique_agents:
            # print(id)
            agent = frame_df[frame_df.agent_id == id].iloc[0]
            cv2.circle(birdview, (int(agent.x),int(agent.y)), 5, ((50.0*id)%256,(2.0*id)%256,0), -1)
            cv2.putText(birdview, str(int(agent.agent_id)), (int(agent.x),int(agent.y)), TEXT_FACE, TEXT_SCALE, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)

        out.write(birdview)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    out.release()
    lvd_df = pd.DataFrame(
        lvd_arr,
        columns=[
            "frame_id",
            "agent_id",
            "lead_vehicle_id",
            "distance",
            "agent_x",
            "agent_y",
            "other_x",
            "other_y",
        ],
    )
    lvd_df.to_csv(f"{lvd_debug_file}.csv")
    # the following lines avoid vehicle A->B and B->A matching errors in the same frame and drops them since the distance values are duplicates and we only need one
    lvd_df["min_agent_id"] = lvd_df[["agent_id", "lead_vehicle_id"]].min(axis=1)
    lvd_df["max_agent_id"] = lvd_df[["agent_id", "lead_vehicle_id"]].max(axis=1)
    lvd_df_filtered = lvd_df.drop_duplicates(
        subset=["frame_id", "min_agent_id", "max_agent_id"]
    )
    mean_lvd_global = lvd_df_filtered.distance.mean()  # per frame and per agent
    min_lvd_global = lvd_df_filtered.distance.min()
    return {"mean_lvd_global": mean_lvd_global, "min_lvd_global": min_lvd_global}

def get_global_stats(df: pd.DataFrame):
    lvd_df = calc_mean_leading_vehicle_dist(df)
    lvd_df.to_csv("lvd_debug_df.csv")
    # the following lines avoid vehicle A->B and B->A matching errors in the same frame and drops them since the distance values are duplicates and we only need one
    lvd_df["min_agent_id"] = lvd_df[["agent_id", "lead_vehicle_id"]].min(axis=1)
    lvd_df["max_agent_id"] = lvd_df[["agent_id", "lead_vehicle_id"]].max(axis=1)
    lvd_df_filtered = lvd_df.drop_duplicates(
        subset=["frame_id", "min_agent_id", "max_agent_id"]
    )
    mean_lvd_global = lvd_df_filtered.distance.mean()  # per frame and per agent
    min_lvd_global = lvd_df_filtered.distance.min()
    return {"mean_lvd_global": mean_lvd_global, "min_lvd_global": min_lvd_global}
    # TODO: check if we need per agent mean lvd? or per frame


def get_agent_data(df: pd.DataFrame):
    """Given a trajectory dataframe, generate agent level parameters for all agents

    Args:
        df (pd.DataFrame): [description]

    Returns:
        dict: For each agent_if(key), return speed stats and trajectory length
    """
    agent_data = {}
    for agent_id in df.agent_id.unique():
        sub_df = df.query(f"agent_id=={agent_id}")
        agent_data[agent_id] = {
            "speed_stats": extract_speed_stats(sub_df),
            "trajectory_length": get_trajectory_length(sub_df),
        }
    return agent_data


def calc_mean_leading_vehicle_dist(df: pd.DataFrame):
    # load the dataframe, group by frame id
    # for each frame id run frame_level_lead_vehicle_dist()
    #       returns a list of agent_id, lead_vehicle_id, distance
    # maintain a global agent_id - lead_vehicle_id map
    #           frame_id, target_agent_id, lead_vehicle, distance
    # for each agent_id-lead_vehicle_pair, store the series of lead vehicle distances

    # we can now calculate per agent mean lead vehicle distance sum(agent_id_lvd_mean)/num_agent_lv_pairs where agent_lvd_mean = sum(agent_lvd)/num_frames

    # or we can calculate a global mean lvd = sum(all_lvds)/total_observations_of_lvd[combination of agent_lv and frames]

    groups = df.groupby("frame_id").groups
    frame_ids = sorted(list(groups.keys()))
    lvd_arr = []
    for frame_id in frame_ids:
        frame_df = df.iloc[groups[frame_id]]

        lvd_arr += calc_frame_level_lead_vehicles_dist(frame_df, frame_id)
    lead_vehicle_df = pd.DataFrame(
        lvd_arr,
        columns=[
            "frame_id",
            "agent_id",
            "lead_vehicle_id",
            "distance",
            "agent_x",
            "agent_y",
            "other_x",
            "other_y",
        ],
    )
    return lead_vehicle_df


def calc_frame_level_lead_vehicles_dist(frame_df: pd.DataFrame, frame_id):

    # return table with columns -> agent_id, lead_vehicle_id, distance
    # for each agent, query for another agent who matches the following criteria
    # direction of velocity is the same (within 20 to -20)
    # radius tolerance
    unique_agents = frame_df.agent_id.unique()
    res = []
    for id in unique_agents:
        # print(id)
        agent = frame_df[frame_df.agent_id == id].iloc[0]
        abs_vel = np.linalg.norm(np.array((agent.vx, agent.vy)))
        # if abs_vel < STATIONARY_VELOCITY:
        #     # this means vehicle is stationary, hence ignore leading vehicle calculations
        #     continue
        # print(agent)
        # possibly avoid this loop and use pandas vectorization operations(esp for distance filtering)
        for idx, other_agent in frame_df[frame_df.agent_id != id].iterrows():
            if is_lead_vehicle(
                (agent.x, agent.y),
                (agent.vx, agent.vy),
                (other_agent.x, other_agent.y),
                (other_agent.vx, other_agent.vy),
                agent.lane_id,
                other_agent.lane_id,
            ):
                dist = np.linalg.norm(
                    np.array([other_agent.x, other_agent.y])
                    - np.array([agent.x, agent.y])
                )
                res.append(
                    [
                        int(frame_id),
                        int(agent.agent_id),
                        int(other_agent.agent_id),
                        dist,
                        agent.x,
                        agent.y,
                        other_agent.x,
                        other_agent.y,
                    ]
                )
    return res


def __is_vector_in_same_dir(
    v1, v2, base_pos, other_pos, tolerance_degrees=ABS_DIRECTION_THRES_DEGREE
):
    # use the cross product of the velocity unit vectors and calculate the angle
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v2_norm > STATIONARY_VELOCITY:
        v1 = v1 / v1_norm
        v2 = v2 / v1_norm
        return (
            abs(np.cross(v1, v2)) < math.sin(math.radians(tolerance_degrees))
        ) and np.dot(v1, v2) > 0
    else:
        #         return False #temporarily disabling matching with static vehicles
        # the other vehicle's velocity is very low(hence noisy velocity direction), use position vector instead of velocity unit vector
        v1 = v1 / v1_norm
        other = other_pos - base_pos
        other = other / np.linalg.norm(other)
        return abs(np.cross(v1, other)) < math.sin(math.radians(tolerance_degrees))


def __is_base_vehicle_behind(base_v, base_pos, other_pos):
    # if dot product is positive implies that the vector from the base agent
    # to the target agent and the base agent's velocity are in the same direction
    # current vehicle is "Ahead"
    base_v = base_v / np.linalg.norm(base_v)
    other = other_pos - base_pos
    other = other / np.linalg.norm(other)
    return np.dot(base_v, other) > 0


def __is_vehicle_in_same_lane(
    base_pos, base_v, other_pos, other_v, base_lane=None, other_lane=None
):
    if (
        base_lane is None
        or other_lane is None
        or pd.isna(base_lane)
        or pd.isna(other_lane)
    ):
        # the velocities should be in same direction, lateral distance should not be very high
        if __is_vector_in_same_dir(base_v, other_v, base_pos, other_pos):
            lateral_dist = np.cross(base_v, other_pos - base_pos)
            return lateral_dist < LATERAL_DISTANCE_THRESHOLD
        else:
            return False
    else:
        return base_lane == other_lane


def is_lead_vehicle(
    base_pos: Tuple[float, float],
    base_v: Tuple[float, float],
    other_pos: Tuple[float, float],
    other_v: Tuple[float, float],
    base_lane: int,
    other_lane: int,
):
    base_pos = np.array(base_pos)
    base_v = np.array(base_v)
    other_pos = np.array(other_pos)
    other_v = np.array(other_v)
    if base_lane == -1 or other_lane == -1:  ## ignore vehicles on intersections
        return False
    # Absolute distance should be in a given range
    dist = np.linalg.norm(other_pos - base_pos)
    # check same lane through maps
    if dist < ABS_POSITION_THRES:
        if __is_vector_in_same_dir(
            base_v, other_v, base_pos, other_pos, ABS_DIRECTION_THRES_DEGREE
        ):
            if __is_base_vehicle_behind(base_v, base_pos, other_pos):
                if __is_vehicle_in_same_lane(
                    base_pos, base_v, other_pos, other_v, base_lane, other_lane
                ):
                    return True
    return False


if __name__ == "__main__":
    intersection_id = "62"
    mapfile = f"../../data/bev_{intersection_id}_.starsjson"
    csvfile = parse_trajectories.parse_trajectory_file(
        f"../../results/tracking_trajectories/demo_tfl_{intersection_id}.npy", mapfile
    )
    temp = "/home/stars/Code/detector/results/tracking_trajectories/df_54_map.csv"
    df = pd.read_csv(temp)
    global_stats = get_global_stats(df)
    agent_wise_data = get_agent_data(df)
