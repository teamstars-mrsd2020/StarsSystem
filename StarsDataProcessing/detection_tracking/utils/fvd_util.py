from collections import defaultdict
import numpy as np
import cv2
import ipdb
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import pandas as pd
from utils.tracker_utils import *
from numpy import cos, sin

global_trajectories = defaultdict(list)
gt_global_trajectories = defaultdict(list)


def transform(H, x, y):

    out = np.matmul(np.linalg.inv(H), np.array([[x], [y], [1]]))
    out = out / out[2]
    return out[0][0], out[1][0]


def transform_points(H, points_to_transform, bev_pos_bound, bev_boundary):

    """
    Apply homography H to given points and return a list of transformed points
    Only those which are inside birds eye view bounds
    :param H:
    :param points_to_transform:
    :param bev_pos_bound:
    :param bev_boundary: polygon type boundary
    :return:
    """
    transformed_points = []
    for detection in points_to_transform:

        x, y, obj_id, cls = detection

        x_map, y_map = transform(H, x, y)

        point = Point(x_map, y_map)
        if not bev_boundary.contains(point):
            continue

        # if( (x_map < bev_pos_bound[0]) or
        #     (x_map >= bev_pos_bound[1]) or
        #     (y_map < bev_pos_bound[2]) or
        #     (y_map >= bev_pos_bound[3])):
        #     continue

        transformed_points.append([x_map, y_map, obj_id, cls])

    return transformed_points


def renderhomography(detections, colors, birdview, bev_pos_bound, boundary):
    global global_trajectories

    # add to global trajectories
    for id, state in detections.items():
        x_map, y_map = detections[id][:2]

        obj_id = id
        global_trajectories[obj_id].append((x_map, y_map))
        global_trajectories = clean_global_traj(global_trajectories)

    # plot all the trajectories
    for id in global_trajectories:
        color = colors[int(id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.polylines(
            birdview,
            np.array([global_trajectories[id]], dtype=np.int32),
            False,
            color,
            thickness=5,
        )
    # cv2.rectangle(
    #     birdview,
    #     (bev_pos_bound[0], bev_pos_bound[2]),
    #     (bev_pos_bound[1], bev_pos_bound[3]),
    #     (0,0,0),
    #     thickness=2
    # )
    cv2.polylines(
        birdview, np.array([boundary], dtype=np.int32), False, (0, 0, 0), thickness=2
    )
    return birdview


def render_gt_homography(detections, birdview):
    global gt_global_trajectories

    # add to global trajectories
    for id, state in detections.items():
        x_map, y_map = detections[id][:2]

        obj_id = id
        # print("Id of vehicle : ",id)
        gt_global_trajectories[obj_id].append((x_map, y_map))

    # plot all the trajectories
    for id in gt_global_trajectories:
        cv2.polylines(
            birdview,
            np.array([gt_global_trajectories[id]], dtype=np.int32),
            False,
            (0, 0, 0),
            thickness=5,
        )

    return birdview


def clean_global_traj(global_trajectories):
    # if we have K objects and the length of the path of K points is less than L, remove that id from the dict
    K = 30
    L = 50
    new_glob = defaultdict(list)
    for id in global_trajectories:
        if len(global_trajectories[id]) < K:
            new_glob[id] = global_trajectories[id]
        else:
            length = 0
            for i in range(1, len(global_trajectories[id]) - 1):
                length += np.linalg.norm(
                    np.array(global_trajectories[id][i])
                    - np.array(global_trajectories[id][i - 1])
                )
            if length > L:
                new_glob[id] = global_trajectories[id]
            # else:
            #     ipdb.set_trace()
    return new_glob


def convert_to_evaluator_format(gt_tracks):
    gt_tracks_dict = {}
    for x, y, id, cls in gt_tracks:
        if cls != "vehicle":
            continue
        gt_tracks_dict[id] = [x, y]

    return gt_tracks_dict


def visualize(img, stars_hd_map):

    # with open(stars_map_file) as f:
    #     data = json.load(f)
    data = stars_hd_map

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, len(data))]
    for id_obj in data:
        polygon = np.array(id_obj["polygon"], dtype=np.int32)
        color = colors[int(id_obj["id"])]

        # center lines in red
        start_point_center_line, end_point_center_line = id_obj["center_line"]
        start_point_center_line, end_point_center_line = (
            tuple(map(int, start_point_center_line)),
            tuple(map(int, end_point_center_line)),
        )
        # print(id_obj["id"])
        if id_obj["id"] != "-1":
            img = cv2.line(
                img,
                start_point_center_line,
                end_point_center_line,
                (0, 0, 0),
                thickness=2,
            )

        color = [i * 255 for i in color]
        img = cv2.polylines(
            img,
            np.int32([polygon]),
            isClosed=True,
            color=color,
            thickness=2,
        )
        img = cv2.putText(
            img=img,
            text=f"ID:{id_obj['id']} DIR:{id_obj['direction'][:3]}",
            org=tuple(np.int32((polygon[0] + polygon[2]) / 2)),  # topleft
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # cv2.imshow("Stars Map", img)
    # cv2.imwrite("bev_fifth_craig_map_viz.png", img)

    cv2.waitKey(0)
    return img


def plot_2d_boxes(bev_img, pred_traj, w=20, h=10):
    for id_ in pred_traj:
        state = pred_traj[id_][-1]
        (
            x,
            y,
            vx,
            vy,
        ) = state
        theta = np.arctan2(vy, vx)

        rect = np.array(
            [
                [x - w / 2, y - h / 2],
                [x + w / 2, y - h / 2],
                [x + w / 2, y + h / 2],
                [x - w / 2, y + h / 2],
            ]
        )
        R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

        center = (x, y)
        rect = (center + ((rect - center) @ R.T)).astype(np.int32)
        isClosed = True
        # rect.reshape((-1, 1, 2))
        cv2.polylines(bev_img, [rect], isClosed, (0, 255, 0), 2)
    return bev_img
