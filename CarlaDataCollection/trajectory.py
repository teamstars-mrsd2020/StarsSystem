import pandas as pd
import json

def create_trajectories_file(intersection_id, side_id, round_):
    column_names = ["frame_id", "agent_id", "x", "y", "z", "vx", "vy", "ax", "ay", "abs_v", "abs_a"]
    traj = pd.DataFrame(columns = column_names)

    # Load the metadata file
    folder = "./../data/metadata/aggregated/"
    filename = "cgt_" + str(intersection_id).zfill(3) + "_" + str(side_id).zfill(3) + "_" + str(round_).zfill(3)
    data = json.load(open(folder + filename + ".json"))
    for frame in data:
        frame_id = frame['frame_id']
        metadata = frame['metadata']
        frame_data = pd.DataFrame(columns = column_names)
        for i, agent in enumerate(metadata):
            agent_id = agent['agentid']
            x = agent["location"]["x"] 
            y = agent["location"]["y"] 
            z = agent["location"]["z"] 
            vx = agent["velocity"]["x"] 
            vy = agent["velocity"]["y"] 
            ax = agent["acceleration"]["x"] 
            ay = agent["acceleration"]["y"] 
            abs_v = (vx ** 2 + vy ** 2) ** 0.5
            abs_a = (ax ** 2 + ay ** 2) ** 0.5
            frame_data.loc[i] = [frame_id, agent_id, x, y, z, vx, vy, ax, ay, abs_v, abs_a]

        traj = traj.append(frame_data)
    
    folder = "./../data/trajectories/"
    traj.to_csv(folder + filename + ".csv", index=False)


if __name__ == "__main__":
    intersection_id = 1
    side_id = 0
    round_ = 1
    create_trajectories_file(intersection_id, side_id, round_)


