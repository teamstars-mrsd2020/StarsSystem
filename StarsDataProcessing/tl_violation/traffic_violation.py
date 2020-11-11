import pandas as pd
from collections import defaultdict 
# from point_loc import get_points, draw_circle
import cv2
import numpy as np
import copy
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.position = [0,0]
        self.tl_relative_posn_prev  = None #List of length = # of tls
        self.tl_relative_posn_now = None #List of length = # of tls
        self.at_intersection = False # Boolen variable to check if vehicle is within the 4 lines.
        self.violate = False
        self.linecrossframe = None
        self.lane_id = None
        self.is_denom = False

    def update_position(self, x, y):
        self.position = [x,y]


    def compute_relative_posn(self, tl_list):
        self.tl_relative_posn_prev = copy.deepcopy(self.tl_relative_posn_now)
        self.tl_relative_posn_now = []

        for tl in tl_list.values():
            coeff = tl.line
            temp = sum([a*b for a,b in zip(coeff,self.position+[1])])
            self.tl_relative_posn_now.append(temp)

    def check_for_break(self, tl_list):
        a = self.tl_relative_posn_prev
        b = self.tl_relative_posn_now

        for i in range(len(a)):
            if b[i]/abs(b[i]) != a[i]/abs(a[i]):
                tl = list(tl_list.keys())[i]
                if tl_list[tl].state.lower() == "red":
                    return True

        return False

    def check_if_at_intersection(self, HDMap):

        for lane in HDMap:
            polygon = Polygon(lane["polygon"])
            if polygon.contains(Point(self.position[0], self.position[1])):
                self.lane_id = int(lane["id"])
                if self.lane_id == -1:
                    self.at_intersection = True
                    return
                break
        self.at_intersection = False


    def set_lane(self, tl_list, HDMap):
        if self.at_intersection:
            self.lane_id = -1
        else:
            dist = [abs(i) for i in self.tl_relative_posn_now]
            self.lane_id = dist.index(min(dist))

class TrafficLight:
    def __init__(self, tl_id):
        self.id = tl_id
        self.line = None
        self.state = "green" # "Red", "Yellow" or "Green"
        self.position = None
        self.startp = None
        self.endp = None
        self.lane = None # Contour for lane
        self.agents = []

    def update_state(self, state):
        self.state = state

    def create_line(self, start_point, end_point):
        self.startp = (start_point[0], start_point[1])
        self.endp = (end_point[0], end_point[1])

        x1,y1 = start_point
        x2,y2 = end_point
        m = (y2-y1)/(x2-x1)
        c = (y2-m*x2)
        self.line = [m,-1,c]

def show_lines(self, birdview, start_point, end_point):
    color = (255, 0, 0) 
    thickness = 2
    image = cv2.line(birdview, start_point, end_point, color, thickness)
    cv2.imshow("Image", image) 
    cv2.waitKey()

def update_all_tl_states(tl_list, df):
    for tl in tl_list.keys():
        
        state = df[df["tl_id"] == tl].iloc[0]['tl_state']
        tl_list[tl].update_state(state)

def create_polygon(birdview, points):
    blank = np.zeros(birdview.shape, np.uint8)
    polygon = cv2.polylines(blank,np.int32([points]),True,(0,255,255),1)
    gray = cv2.cvtColor(polygon, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 120, 255, 1)
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # cv2.drawContours(blank, [cnts[0]], -1, (36, 255, 12), 2)
    # cv2.imshow('image', blank)
    # cv2.waitKey()
    return cnts


def tlv(cfg, debug = False):
    
    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 1
    TEXT_THICKNESS = 2

    if debug:
        folder = "./../"
    else:
        folder = ""

    path_img = folder + cfg["HD_map"] + "/bev_frame.jpg"
    birdview = cv2.imread(path_img) 

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter("video2.mp4", fourcc, 20.0, (1920, 1080))
    out = cv2.VideoWriter("video2.mp4", fourcc, 20.0, (birdview.shape[1], birdview.shape[0]))

    HDMap = json.load(open(folder + cfg["HD_map"]+"/annotations.starsjson"))

    for id_ in HDMap:
        if id_["id"]==str(-1):
            points_list = [[int(x) for x in l] for l in id_["polygon"]]
            break

    #These are the point of traffic light and will also be used to create the lines
    # filename = prefix + str(intersection_id).zfill(3) + "_" + str(side).zfill(3) + "_" + str(round_).zfill(3)
    tl_data = pd.read_csv(folder + cfg["TL_status_combined"])
    tl_list = {}
    tl_count = 4

    start_idx = 0
    thickness = 2
    
    for tl in range(tl_count):
        tl_list[tl] = TrafficLight(tl)
        start = points_list[start_idx]
        tl_list[tl].position = start
        stop_idx = start_idx + 1
        if stop_idx >= tl_count:
            stop_idx = 0
        stop = points_list[stop_idx]
        tl_list[tl].create_line(start, stop)
        start_idx += 1

        # tl_list[tl].lane = create_polygon(birdview, lanes[tl])[0]

    cnts = create_polygon(birdview, points_list)
    # image = cv2.drawContours(birdview, [cnts[0]], -1, (36, 255, 12), 2)

    agent_data = pd.read_csv(folder +cfg["trajectory_pred"])
    # print(agent_data.head())
    violation_count = 0
    denom = 0
    denom_list = []

    frames1 = set(agent_data.frame_id.unique())
    frames2 = set(tl_data.frame_id.unique())

    frames = list(frames1.intersection(frames2))

    agent_list = {}

    for frame in frames:
        #Take subset of dataframe
        update_all_tl_states(tl_list, tl_data[tl_data["frame_id"] == frame])
        state = agent_data[agent_data["frame_id"] == frame]

        bev_frame = copy.deepcopy(birdview)

        for tl in tl_list:
            if tl_list[tl].state.lower() == "red":
                cv2.circle(bev_frame, (tl_list[tl].position[0], tl_list[tl].position[1]), 9, (0, 0, 255), -1)
                cv2.line(bev_frame, tl_list[tl].startp, tl_list[tl].endp, (0, 0, 255), thickness)
                cv2.putText(bev_frame, str(tl), (tl_list[tl].position[0], tl_list[tl].position[1]), TEXT_FACE, TEXT_SCALE, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)
            else:
                cv2.circle(bev_frame, (tl_list[tl].position[0], tl_list[tl].position[1]), 9, (0, 255, 0), -1)
                cv2.line(bev_frame, tl_list[tl].startp, tl_list[tl].endp, (0, 255, 0), thickness)
                cv2.putText(bev_frame, str(tl), (tl_list[tl].position[0], tl_list[tl].position[1]), TEXT_FACE, TEXT_SCALE, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)
            
            tl_list[tl].agents = []

        for index, row in state.iterrows():
            agent_id = row["agent_id"]

            if agent_id not in agent_list.keys():
                agent_list[agent_id] = Agent(agent_id)
                agent_list[agent_id].update_position(row["x"], row["y"])
                agent_list[agent_id].check_if_at_intersection(HDMap)
                agent_list[agent_id].compute_relative_posn(tl_list)
                # agent_list[agent_id].set_lane(tl_list, HDMap)
                # lane = agent_list[agent_id].lane_id
                # if lane != -1:
                #     tl_list[lane].agents.append(agent_id)
                # print(agent_id, lane)

            agent_list[agent_id].update_position(row["x"], row["y"])
            agent_list[agent_id].check_if_at_intersection(HDMap)
            x, y = agent_list[agent_id].position
            agent_list[agent_id].compute_relative_posn(tl_list)

            agent_list[agent_id].set_lane(tl_list, HDMap)
            lane = agent_list[agent_id].lane_id
            if lane != -1:
                tl_list[lane].agents.append(agent_id)
            # print(agent_id, lane)

            if agent_list[agent_id].tl_relative_posn_prev is not None:
                if agent_list[agent_id].check_for_break(tl_list):
                    agent_list[agent_id].linecrossframe = frame
                    
            if agent_list[agent_id].linecrossframe is not None:
                if frame == agent_list[agent_id].linecrossframe+4:
                    if agent_list[agent_id].at_intersection:
                        agent_list[agent_id].violate = True
                        violation_count += 1
                        print("Agent ",agent_id, " violates traffic light")

            if agent_list[agent_id].violate:
                cv2.circle(bev_frame, (int(x),int(y)), 5, ((agent_id*50)%256,(2*agent_id)%256,255), -1)
            else:
                cv2.circle(bev_frame, (int(x),int(y)), 5, ((50*agent_id)%256,(2*agent_id)%256,0), -1)
            
            cv2.putText(bev_frame, str(agent_id), (int(x),int(y)), TEXT_FACE, TEXT_SCALE, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)

        # cv2.namedWindow("STARS Tracking Evaluation", cv2.WINDOW_NORMAL)
        # cv2.imshow("STARS Tracking Evaluation", bev_frame)

        for tl in tl_list.values():
            if tl.state.lower() != "red": continue

            # print(tl_list[tl].agents)
            min_dist = float('inf')
            first_ = -1
            
            for agent_id in tl.agents:
                coeff = tl.line
                dist = abs(sum([a*b for a,b in zip(coeff,agent_list[agent_id].position+[1])]))
                # print(agent_id, dist)
                if dist < min_dist:
                    min_dist = dist
                    first_ = agent_id

            if first_ >= 0:
                if not agent_list[first_].is_denom: 
                    denom += 1
                    agent_list[first_].is_denom = True
                    denom_list.append(first_)
                    # print(first_)

        out.write(bev_frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    out.release()

    print("Total number of violations: ", violation_count)
    if denom!= 0:
        # print(violation_count, denom, denom_list)
        return violation_count, denom
    else:
        return 0,0



if __name__ == "__main__":
    run_id = 3
    cfg = json.load(open("./../../data/runs/run_"+str(run_id)+".json"))
    intersection_id = 1
    side = 0
    round_ = 1
    carla = True
    # TLV = tlv(intersection_id, side, round_, carla)
    TLV = tlv(cfg, True)

    print("Percentage of violation: ", TLV)