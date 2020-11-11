from . import _init_paths

import json
import utils.dp_utils
import numpy as np
import cv2
import os
from tqdm import tqdm
from copy import deepcopy

# from detectron2.utils.logger import setup_logger
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog
from .utils import fvd_util

# from bev_tracker.filter_track_linear_sum import *
# from bev_tracker import tracker_eval
from utils.tracker_utils import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# from IPython import embed
from bev_tracker.tracking import BevTracker


# from detector import Detector
from CenterTrack.src.center_track import Detector

# from detector import Detector


# from mapping import visualize_stars_lane_map
from time import sleep


# -------- DETECTOR AND TRACKER -------------------
# tracker_eval = tracker_eval.tracker_evaluator(max_dist_thresh)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


prev_frame_state = {}


def pre_filter(tracks, FILTER):
    """
    Compute velocity and accelerations for each track

    Each track must be visible for atleast 3 frames for successful
    computation of velocity and acceleration
    Arguments:
        tracks {list of lists} -- detected tracks from the camera view
        FILTER {bool} -- If we want to store raw tracks w/o vel and acc make it False

    Returns:
        dictionary -- {id1: [x1, y1, vx1, vy1, ax1, ay1, id1], id2: [x2, y2,...]}
    """
    # TODO - add time dependence
    valid_frame_state = {}
    for x, y, id, cls in tracks:
        if FILTER:
            if id in prev_frame_state:

                prev_x, prev_y = prev_frame_state[id][0], prev_frame_state[id][1]
                prev_vx, prev_vy = prev_frame_state[id][2], prev_frame_state[id][3]
                # prev_ax, prev_ay = prev_frame_state[id][4], prev_frame_state[id][5]

                vx = x - prev_x
                vy = y - prev_y

                if prev_vx:
                    ax = vx - prev_vx
                    ay = vy - prev_vy
                    valid_frame_state[id] = np.array([x, y, vx, vy, ax, ay])

                else:
                    ax = None
                    ay = None

            else:
                vx = None
                vy = None
                ax = None
                ay = None

            prev_frame_state[id] = [x, y, vx, vy, ax, ay, id]
        else:
            valid_frame_state[id] = np.array([x, y, None, None, None, None])

    return valid_frame_state


class DataProcessing:
    def __init__(self, carla):
        """
        Intialize the member variables of class.
        """
        self.detector = None
        self.tracker_evaluator = None
        self.video_reader = None
        self.tf_matrix = None  # b/w bev image and camera image
        self.gt_tf_matrix = None  # b/w grnd-truth locations and bev camera
        self.intrinsics = None  # intrinsics matrix for bev camera
        self.bev_boundary_polygon = None
        self.bev_boundary = None
        self.trajectories = []
        self.gt_data = None
        self.params = None
        self.pbar = None
        self.recorded_boxes = None
        self.filtered_traj = None

        self.bev_tracker = None
        self.video_writer = None
        self.bev_image = None
        self.carla = carla
        self.hd_map = None

    def load_data(self, tf_id, ip_files):
        """
        Load data from input files.

        Arguments:
            tf_id {integer} -- required for loading corresponding tf cam config
            ip_files {dictionary} -- dictionary of input file paths
        """
        # ----- Load video ------------------
        self.video_reader = cv2.VideoCapture(ip_files["tcam_file"])
        if self.video_reader.isOpened() is False:
            print("Error opening video stream or file")

        # ----- Load bev image file ------------------
        self.bev_image = cv2.imread(ip_files["bev_img_file"])  # 1080 x 1920
        self.map_path = ip_files["map_file"]
        print(os.path.abspath(ip_files["bev_img_file"]))
        print(os.path.abspath(ip_files["map_file"]))

        # import ipdb;ipdb.set_trace()

        # self.bev_image = visualize_stars_lane_map.visualize(
        #     os.path.abspath(ip_files["bev_img_file"]),
        #     os.path.abspath(ip_files["map_file"]),
        # )

        # ----- Load config file ------------------
        if os.path.isfile(ip_files["config_file"]):
            config_data = json.load(open(ip_files["config_file"]))
            if self.carla:
                tf_scene = config_data["carla" + "_" + tf_id]
            else:
                scene = json.load(open(ip_files["intersection_config"]))
                tf_scene = config_data[scene["location"]]

            camera_correspondences = np.asarray(tf_scene["xy_pixels_camera"])
            bev_correspondences = np.asarray(tf_scene["xy_pixels_map"])
            self.tf_matrix, _ = cv2.findHomography(
                bev_correspondences, camera_correspondences
            )
            if self.carla:
                self.gt_tf_matrix = np.asarray(tf_scene["birdview_tf"])
            self.bev_boundary = tf_scene["bev_boundary"]
            self.bev_boundary_polygon = Polygon(tf_scene["bev_boundary"])
            if self.carla:
                self.intrinsics = np.asarray(tf_scene["intrinsics"])
            self.params = config_data["params"]
            self.tracker_evaluator = tracker_eval.tracker_evaluator(
                self.params["max_dist_thresh"]
            )

            # ----- Load detector ------------------
            self.detector = Detector(
                self.params["det_model"],
                ip_files["det_config_file"],
                self.params["det_conf_thresh"],
            )

            # _-Load Tracker----
            self.tracking_params = json.load(open(ip_files["tracking_config_file"]))
            self.tracking_params["map_file"] = ip_files["map_file"]
            self.bev_tracker = BevTracker(
                self.tracking_params, self.bev_boundary_polygon
            )
        else:
            print("Error opening config file")

        # ----- Load ground truth trajectories file if required --------------
        if self.carla:
            if self.params["evaluate_tracker"] & os.path.isfile(ip_files["gt_file"]):
                # df = pd.read_csv(ip_files["gt_file"])
                # self.gt_data = convertDataFrametoTraj(df)
                self.gt_data = json.load(open(ip_files["gt_file"]))
            else:
                if self.params["evaluate_tracker"]:
                    print("Error :" + ip_files["gt_file"] + " not found")

        # ----- Load pre-computed bounding boxes file if required ------------
        if self.params["use_stored_detections"] & os.path.isfile(
            ip_files["detections_file"]
        ):
            result = json.load(open(ip_files["detections_file"]))
            self.recorded_boxes = result["objects"]
        else:
            if self.params["use_stored_detections"]:
                print("Error :" + ip_files["detections_file"] + " not found")
                return

        # ----- Load trajectory data file if required --------------
        if (
            self.params["use_traj_file"]
            & os.path.isfile(ip_files["traj_file"])
            & self.params["use_stored_detections"]
        ):
            df = pd.read_csv(ip_files["traj_file"])
            self.filtered_traj = convertDataFrametoTraj(df)
        else:
            if self.params["use_traj_file"]:
                print("Error: " + ip_files["traj_file"] + " not found")
                return
            if self.params["use_stored_detections"]:
                print(
                    "Error: Can't use filtered trajectory if detection boxes are unavailable"
                )

        # ----- Load bev hdmap file if required ------------
        if self.params["plot_hdmap"] & os.path.isfile(ip_files["map_file"]):
            self.hd_map = json.load(open(ip_files["map_file"]))
        else:
            if self.params["plot_hdmap"]:
                print("Error :" + ip_files["map_file"] + " not found")
                return

    def generate_gt_tracks(self, h, w):
        """
        Extract ground-truth tracks from gt carla locations.

        Transform the gt locations of carla to bev camera frame.
        Ignore those tracks whose bounding boxes lie on the corner of
        the image frame governed by camera_view_offset
        Arguments:
            h {scalar} -- camera_view height
            w {scalar} -- camera_view width

        Returns:
            list of lists -- [[xbev1, ybev1, id, cls], [xbev2, ybev2, id, cls] ..]
        """
        gt_tracks = []
        # import ipdb

        # ipdb.set_trace()
        for gt_detections in self.gt_data[self.pbar.n]["metadata"]:

            world_point = gt_detections["location"]
            x_world, y_world, z_world = (
                world_point["x"],
                world_point["y"],
                world_point["z"],
            )
            bbox_2d = gt_detections["bounding_box2d"]
            x_bbox = (bbox_2d[0] + bbox_2d[2]) / 2
            y_bbox = ((bbox_2d[1] + bbox_2d[3]) / 2 + bbox_2d[3]) / 2
            id, cls = gt_detections["agentid"], gt_detections["classname"]
            point = np.array([x_world, y_world, z_world, 1]).reshape(4, 1)
            x_bev, y_bev = get_bev_coords(
                point, self.gt_tf_matrix, self.intrinsics, self.tf_matrix
            )

            # x_bev_homo, y_bev_homo = svd_util.transform(H_static_map, x_bbox, y_bbox)
            # if ((x_bev < bev_pos_bound[0]) or
            #         (x_bev >= bev_pos_bound[1]) or
            #         (y_bev < bev_pos_bound[2]) or
            #         (y_bev >= bev_pos_bound[3])):
            #     continue
            if not self.bev_boundary_polygon.contains(Point(x_bev, y_bev)):
                continue

            if (
                x_bbox >= self.params["camera_view_offset"]
                and x_bbox <= (w - self.params["camera_view_offset"])
                and y_bbox >= self.params["camera_view_offset"]
                and y_bbox <= (h - self.params["camera_view_offset"])
            ):
                gt_tracks.append([x_bev, y_bev, id, cls])

        return gt_tracks

    def run_processing(self, op_files):
        """
        Implement trajectory generation from a traffic video
        1. Run detection(detectron + sort) on video frame
        2. Transform detections to bird's eye view (BEV)
        3. Track positions of agents in BEV
        4. Plot the trajectories on BEV
        5. Option to store bboxes and trajectories
        """
        # global variables
        # global birdview, bev_pos_bound, metadata

        # ---------- VIDEO WRITER --------------------------
        fourcc = cv2.VideoWriter_fourcc(*"mjpg")
        vid = cv2.VideoWriter(op_files["demo_video_file"], fourcc, 20.0, (1280, 480))

        gt_tracks = None
        tracked_agents_history = None
        mota = None
        motp = None
        result = {"objects": []}  # might be unused in case raw data is available

        # --------- progress bar -------------------------
        total = self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
        self.pbar = tqdm(total)

        # while self.video_reader.isOpened():
        for _ in range(int(total)):

            ret, camera_view = self.video_reader.read()
            points_to_transform = []

            if ret is True:

                # ---- If raw detection data not available run detector and tracker ---
                if not self.params["use_stored_detections"]:
                    # Run detector model
                    raw_detection_boxes, tracked_boxes, _ = self.detector.run_detector(
                        camera_view, self.params["relevant_classes"]
                    )
                else:  # raw detections are available
                    raw_detection_boxes = self.recorded_boxes[self.pbar.n]["raw_boxes"]
                    tracked_boxes = self.recorded_boxes[self.pbar.n]["tracked_boxes"]

                # plot tracked bounding boxes on image_view
                self.camera_view = dp_utils.plot_bboxes(
                    camera_view, tracked_boxes, self.detector.get_classlist()
                )
                # transform points to bird's-eye view
                points_to_transform = dp_utils.extract_bottom_center(
                    tracked_boxes,
                    self.params["camera_view_offset"],
                    self.camera_view.shape[0],
                    self.camera_view.shape[1],
                )

                tracks = dp_utils.transform_points(
                    self.tf_matrix, points_to_transform, self.bev_boundary_polygon
                )

                # track trajectories
                pre_filtered_tracks = pre_filter(tracks, True)
                if self.params["use_traj_file"]:
                    tracks = self.filtered_traj[self.pbar.n]
                else:
                    if self.params["filter_traj"]:
                        tracks, tracked_agents_history = self.bev_tracker.tracker(
                            pre_filtered_tracks
                        )
                        # tracks = tracker(pre_filtered_tracks, self.bev_boundary_polygon)
                    else:
                        tracks = pre_filtered_tracks

                # save trajectories
                if self.params["save_traj_file"] and not self.params["use_traj_file"]:
                    self.trajectories.append(tracks)

                # evaluate tracker
                if self.carla:
                    if self.params["evaluate_tracker"]:
                        gt_tracks = self.generate_gt_tracks(
                            self.camera_view.shape[0], self.camera_view.shape[1]
                        )
                        gt_tracks = dp_utils.convert_to_evaluator_format(gt_tracks)
                        self.tracker_evaluator.validate(tracks, gt_tracks)
                        if self.pbar.n % 10 == 0:
                            self.tracker_evaluator.full_metrics()
                        mota, motp = self.tracker_evaluator.mot_metrics()

                bev_view = dp_utils.renderhomography(
                    tracks,
                    gt_tracks,
                    tracked_agents_history,
                    self.bev_image.copy(),
                    self.bev_boundary,
                    self.params["evaluate_tracker"],
                )
                # plot trajectories
                if self.params["plot_hdmap"]:
                    bev_view = svd_util.visualize(bev_view, self.hd_map)

                # bev_view = dp_utils.renderhomography(
                #     tracks, gt_tracks, self.bev_image, self.bev_boundary,
                #     self.params["evaluate_tracker"])

                # final frame
                joined_view = dp_utils.join_views(
                    bev_view,
                    self.camera_view,
                    self.params["evaluate_tracker"],
                    mota,
                    motp,
                )

                if self.params["realtime_display"]:
                    cv2.namedWindow("STARS Tracking Evaluation", cv2.WINDOW_NORMAL)
                    cv2.imshow("STARS Tracking Evaluation", joined_view)
                    vid.write(joined_view)
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        break
                else:  # only store video
                    vid.write(joined_view)
                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        break

                # ------ Store the plotted result ------------
                if (
                    self.params["save_detections"]
                    and not self.params["use_stored_detections"]
                ):
                    frame_result = {
                        "raw_boxes": raw_detection_boxes,
                        "tracked_boxes": tracked_boxes,
                        "pose2d": pre_filtered_tracks,
                    }
                    (result["objects"]).append(frame_result)

                self.pbar.update(1)

            # Break the loop if video ends
            else:
                continue
                # print("here")
                # break
                # if (self.params["realtime_display"]):
                #     if cv2.waitKey(25) & 0xFF == ord("q"):
                #         break
                # else:
                #     break
        print("Video Reader closed...")
        if self.params["save_detections"] and not self.params["use_stored_detections"]:
            # create raw data file
            image_width, image_height, image_channels = self.camera_view.shape
            result["stats"] = {
                "image_width": image_width,
                "image_height": image_height,
                "image_channels": image_channels,
                "video_name": op_files["demo_video_file"],
                "frame_count": len(result["objects"]),
                "classlist": self.detector.get_classlist(),
            }
            detections_file = op_files["detections_file"]
            json.dump(result, open(f"{detections_file}", "w"), cls=NumpyEncoder)

        if self.params["smooth_trajectories"]:
            self.trajectories = smoothTrajectories(self.trajectories)

            # import ipdb

            # ipdb.set_trace()
            # df = convertTrajectoryToDataFrame(self.trajectories)
            # bev_view = dp_utils.plot_smooth_trajs(
            #     df, self.bev_image.copy(), self.bev_boundary
            # )
            # joined_view = utils.dp_utils.join_views(
            #     bev_view, self.camera_view, self.params["evaluate_tracker"], mota, motp
            # )
            # cv2.imshow("Final Image after smoothing", joined_view)
            # cv2.waitKey(0)

        if self.params["save_traj_file"] and not self.params["use_traj_file"]:
            print("Saving trajectory file..")
            df = convertTrajectoryToDataFrame(self.trajectories, self.map_path)
            df.to_csv(op_files["traj_file"], index=False)

        # When everything is done, release the video capture object
        self.video_reader.release()
        if self.params["realtime_display"]:
            # Closes all the frames
            cv2.destroyAllWindows()


def detection_tracking(run_file, debug=False):

    # ------- Input Files ---------
    if debug:
        folder = "./../"
    else:
        folder = ""

    path_ = folder + run_file["intersection_config"]
    with open(
        path_,
    ) as f:
        config = json.load(f)
    tl_id = config["traffic_light_id"]

    carla = run_file["HD_map"][-5] == "c"

    if carla:
        prefix = "c_"
    else:
        prefix = "r_"

    filename = prefix + run_file["video"][-15:-4]

    ip_files = {}
    ip_files["tcam_file"] = folder + run_file["video"]
    ip_files["bev_img_file"] = folder + run_file["HD_map"] + "/bev_frame.jpg"
    ip_files["map_file"] = folder + run_file["HD_map"] + "/annotations.starsjson"
    print(ip_files["bev_img_file"])

    if debug:
        ip_files["config_file"] = "./config.json"
        ip_files["det_config_file"] = "./det_config.json"
    else:
        ip_files[
            "config_file"
        ] = "../StarsDataProcessing/detection_tracking/config.json"
        ip_files[
            "det_config_file"
        ] = "../StarsDataProcessing/detection_tracking/det_config_test.json"

    if carla:
        ip_files["gt_file"] = folder + run_file["trajectory_gt"]

    ip_files["intersection_config"] = folder + run_file["intersection_config"]
    ip_files["detections_file"] = (
        folder + "../data/extras/detection_boxes/" + filename + ".json"
    )
    ip_files["traj_file"] = folder + run_file["trajectory_pred"]

    ip_files["tracking_config_file"] = (
        folder + "../StarsDataProcessing/detection_tracking/tracking_config.json"
    )

    op_files = {}
    op_files["demo_video_file"] = (
        folder + "../data/extras/annotated_videos/" + filename + ".mp4"
    )
    op_files["traj_file"] = folder + run_file["trajectory_pred"]
    op_files["detections_file"] = (
        folder + "../data/extras/detection_boxes/" + filename + ".json"
    )

    data_processing = DataProcessing(carla)
    data_processing.load_data(str(tl_id), ip_files)
    data_processing.run_processing(op_files)
    data_processing.pbar.close()

    # return data_processing.trajectories # npy array


# def detection_tracking2(run_file, debug = False):

#     # ------- Input Files ---------
#     if debug:
#         folder = "./../"
#     else:
#         folder = ""

#     path_ = folder+ run_file["intersection_config_2"]
#     with open(path_, ) as f:
#         config = json.load(f)
#     tl_id = config["traffic_light_id"]

#     carla = run_file["HD_map"][-5] == "c"

#     if carla:
#         prefix = "c_"
#     else:
#         prefix = "r_"

#     filename = prefix + run_file["video"][-15:-4]

#     ip_files = {}
#     ip_files["tcam_file"] = folder + run_file["video_2"]
#     ip_files["bev_img_file"] = folder + run_file["HD_map"] + "/bev_frame.png"
#     ip_files["map_file"] = folder + run_file["HD_map"] + "/annotations.starsjson"

#     if debug:
#         ip_files["config_file"] = "./config.json"
#         ip_files["det_config_file"] = "./det_config.json"
#     else:
#         ip_files["config_file"] =  "../StarsDataProcessing/detection_tracking/config.json"
#         ip_files["det_config_file"] = "../StarsDataProcessing/detection_tracking/det_config_test.json"

#     if carla:
#         ip_files["gt_file"] = folder + run_file["trajectory_gt"]

#     ip_files["intersection_config"] = folder + run_file["intersection_config_2"]
#     ip_files["detections_file"] = folder + "../data/extras/detection_boxes/" + filename + ".json"
#     ip_files["traj_file"] = folder + run_file["trajectory_pred"]

#     op_files = {}
#     op_files["demo_video_file"] = folder + "../data/extras/annotated_videos/" + filename + ".mp4"
#     op_files["traj_file"] = folder + run_file["trajectory_pred"]
#     op_files["detections_file"] = folder + "../data/extras/detection_boxes/" + filename + ".json"

#     data_processing = DataProcessing(carla)
#     data_processing.load_data(str(tl_id), ip_files)
#     data_processing.run_processing(op_files)
#     return data_processing.trajectories # npy array


if __name__ == "__main__":
    run_id = 5
    cfg = json.load(open("./../../data/runs/run_" + str(run_id) + ".json"))
    # import ipdb;ipdb.set_trace()
    detection_tracking(cfg, True)

    cfg["intersection_config"] = cfg["intersection_config_2"]
    cfg["video"] = cfg["video_2"]
    cfg["trajectory_pred"] = cfg["trajectory_pred_2"]
    detection_tracking(cfg, True)

    print("done")
