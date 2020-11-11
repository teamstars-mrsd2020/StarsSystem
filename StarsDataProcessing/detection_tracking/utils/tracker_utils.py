import numpy as np
import os
import json
import glob
import ipdb
from utils import dp_utils
import pandas as pd
import matplotlib.pyplot as plt
import json
import cv2
from scipy.interpolate import splrep , splev
from mathutils.geometry import intersect_point_line
# from demo_pr3 import H_cam2bev,cam_view
from shapely.geometry import Point, Polygon
def get_closest_point_on_line(point,line):
    """
    Given a point get the closest point to the line segment

    Args:
        point ([tuple]): [description]
        line ([list]): [list of list [[x1,y1],[x2,y2]]]]

    Returns:
        [type]: [description]
    """
    return intersect_point_line(point, *line)


def remove_small_trajectories(tracker_frame_list):
    '''
        Takes in a list of frames
        Each frame is a dictionary with vehicle_id:state
        tracker_frame_list= [{id:x,y,vx,vy,ax,ay,id},
              {...},
              {...},
              {...},]
        Returns post filtered tracker output
    '''
    pass

def get_bev_coords(X,T,K,H):
    # X=np.array([X['x'],X['y'],X['z'],1]).reshape(4,1)
    #X=np.array([-88.10905456542969,120.70973205566406,0.11107280850410461,1]).reshape(4,1)
    # T=np.vstack((T,np.array([0,0,0,1])))
    H = np.linalg.inv(H)
    # ipdb.set_trace()
    sensor_X=np.linalg.inv(T)@X
    sensor_X=np.array([sensor_X[1,0],-sensor_X[2,0],sensor_X[0,0]]).reshape(3,1)
    camera_X=K@sensor_X
    camera_x_w,camera_y_w,_=(camera_X/camera_X[-1])[:,0]
    # ipdb.set_trace()
    # bev_X=H@np.array([camera_x_w,camera_y_w,1]).reshape(3,1)
    # bev_x,bev_y,_=(bev_X/bev_X[-1])[:,0]
    return camera_x_w,camera_y_w

def get_worldXY_coords(x,T,K,H):
    '''
     Returns world coordinates of a point in camera's perspective. Assuming Z of that coordinate to be 0
     x - 3x1 np array - ([x,y,1])
     T - 4x4 Camera Transformation matrix
     K-  3x3 intrinsics
    '''

    '''
    {'z': 0.057462919503450394, 'y': 120.36502838134766, 'x': -88.13628387451172}
    bbox_x = 325, bbox_y = 433
    bev_x = 563.650
    bev_y = 426.613
    
    {'z': 0.11107280850410461, 'x': -88.10905456542969, 'y': 120.70973205566406}
    camera_x = 248.5 
    camera_y = 774 
    K=
    H=array([[-6.99848360e-02, -8.48166323e-01, 7.68253650e+02],  #cameratoBEV
           [-1.73373891e-01, -1.46612293e-01, 4.06120082e+02],
           [-4.79164117e-04, -3.99989659e-04, 1.00000000e+00]])
    "instrinsics": [
    [
      960.0000000000001,
      0.0,
      960.0
    ],
    [
      0.0,
      960.0000000000001,
      540.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ]
    "sensor_world_matrix": [
    [
      0.6330227294094743,
      0.7660440151570704,
      0.11161895400429804,
      -88.98273468017578
    ],
    [
      -0.7544060919176474,
      0.6427881197113354,
      -0.13302210957396599,
      143.85897827148438
    ],
    [
      -0.17364812849127345,
      -0.0,
      0.9848077616832019,
      7.172693729400635
    ]
    '''

    # X=np.array([-88.10905456542969,120.70973205566406,0.11107280850410461,1]).reshape(4,1)
    
    # sensor_X=np.linalg.inv(T)@X
    # sensor_X=np.array([sensor_X[1,0],-sensor_X[2,0],sensor_X[0,0]]).reshape(3,1)
    # camera_X=K@sensor_X
    # camera_x_w,camera_y_w,_=(camera_X/camera_X[-1])[:,0]
    # bev_X=H@np.array([camera_x_w,camera_y_w,1]).reshape(3,1)
    # bev_x,bev_y,_=(bev_X/bev_X[-1])[:,0]
    # return bev_x,bev_y

    bev_X=np.array([x[0],x[1],1]).reshape(3,1)
    camera_X_rev=np.linalg.inv(H)@bev_X
    camera_x_rev,camera_y_rev,_=(camera_X_rev/camera_X_rev[-1])[:,0]
    camera_X_rev=np.array([camera_x_rev,camera_y_rev,1]).reshape(3,1)
    sensor_X_rev=np.linalg.inv(K)@camera_X_rev
    
    # sensor_X_rev_xyz=np.array([sensor_X_rev[2,0],sensor_X_rev[0,0] ,-sensor_X_rev[1,0]]).reshape(3,1)
    #above is same as multiplying by 
    #[[ 0.,  1.,  0.],
    #[-0., -0., -1.],
    #[ 1.,  0.,  0.]]
    T=np.vstack((T,np.array([0,0,0,1])))
    Tyz_x = np.array([[ 0.,  1.,  0.],
    [0., -0., -1.],
    [ 1.,  0.,  0.]] )
    Tyz_x_hom=np.hstack((np.vstack((Tyz_x,np.zeros(3))),np.zeros((4,1))));Tyz_x_hom[-1,-1]=1
    Tyz_x_hom@np.linalg.inv(T)
    T_final=Tyz_x_hom@np.linalg.inv(T)
    T_del=np.delete(T_final,2,1)
    T_del=np.linalg.inv(T_del[:3])
    X=T_del@sensor_X_rev

    # T=np.vstack((T,np.array([0,0,0,1])))
    # X=T@sensor_X_rev_xyz

    # T=np.array(T);K=np.array(K)
    # T = np.delete(T,3,0) #3x4 matrix
    # T = np.delete(T,2,1) #3x3 matrix
    # H=K@T #3x3 matrix TODO: confirm T or inv T
    # X=np.linalg.inv(H)@x #3x1 np array
    X/=X[-1]
    return X[:2]

def _get_lane_id(x, y, id_obj_map):
    if id_obj_map is None:
        return None
    point = Point(x, y)
    for id in id_obj_map:
        polygon = id_obj_map[id]["polygon"]
        assert len(polygon) == 4, f"Found polygon with len!=4 :{id}"
        polygon = Polygon(polygon)
        if point.within(polygon):
            return id
    return None

def convertTrajectoryToDataFrame(trajectoryList: np.ndarray, mapfile:str=None):
    """Parse a npy file of the trajectory and return a dataframe.

    Our trajectory files are stored as a list of dictionaries. Each dictionary corresponds to a given frame.
    We have the [x,y,vx,vy,ax,ay] for each agent in a current frame. The agent id is the dictionary key.
    Example:
    Consider a scenario for 3 frames. Agent 1 is present in all frames. Agent 2 is only present in frame 1. Agent 3 is only present in Frame 2
    The data will look like this,
    ```
    [
        {id1: [x1, y1, vx1, vy1, ax1, ay1], id2: [x2, y2, vx2, vy2, ax2, ay2]},
        {id1: [..], id3: [..]},
        {id1: [..]},
    ]
    ```
    Args:

        filename (str): path to the numpy file

    Returns:

        dataFrame (pandas.DataFrame): dataFrame of input trajectories

        DF["frame-id"].groupby
    """
    # TODO: Convert Pixel to meters per sec
    dataframe_columns = [
        "frame_id",
        "agent_id",
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "abs_v",
        "abs_a",
        "lane_id"
    ]

    if mapfile:
        with open(mapfile) as f:
            mapobj = json.load(f)
            id_obj_map = {x["id"]: x for x in mapobj}
    else:
        id_obj_map = None
    dataframe_list = []
    for frame_id, frame_obj in enumerate(trajectoryList):
        for agent_id in frame_obj:
            x, y, vx, vy, ax, ay = frame_obj[agent_id]
            abs_v = (vx ** 2 + vy ** 2) ** 0.5
            abs_a = (ax ** 2 + ay ** 2) ** 0.5
            lane_id = _get_lane_id(x, y, id_obj_map)
            dataframe_list.append(
                [frame_id, agent_id, x, y, vx, vy, ax, ay, abs_v, abs_a,lane_id]
            )

    dataframe = pd.DataFrame(dataframe_list, columns=dataframe_columns)
    return dataframe





def smoothTrajectories(trajectories: list, interpolate_factor=5,sampling_factor=5,span_perc=0.1,spline_interpolation=False):
    """
    Smoothens the trajectory

    Our trajectory files are stored as a list of dictionaries. Each dictionary corresponds to a given frame.
    We have the [x,y,vx,vy,ax,ay] for each agent in a current frame. The agent id is the dictionary key.
    Example:
    Consider a scenario for 3 frames. Agent 1 is present in all frames. Agent 2 is only present in frame 1. Agent 3 is only present in Frame 2
    The data will look like this,
    ```
    [
        {id1: [x1, y1, vx1, vy1, ax1, ay1], id2: [x2, y2, vx2, vy2, ax2, ay2]},
        {id1: [..], id3: [..]},
        {id1: [..]},
    ]

    Args:
        trajectory (npy): [numpy trajectories]
        interpolate_factor = Takes every n points in an agent trajectory to fit a spline. Default 5
        sampling_factor= Up sampling the trajectories n times. (default 5)
        span_perc= Exponential moving average span as a fraction of total length of an agent's trajectory (default 0.3)
        spline_interpolation (Boolean )= Whether to perform spline interpolation (default True)

    Returns:
        dataFrame (pandas.df): dataFrame of input trajectories
        
        postProcess-
        1. small trajectories join
        2. smooth trajectories
        3. using mapinfo - trajectories to lanecentre
    """
    INTERPOLATE_FACTOR = 5
    SAMPLING_FACTOR = 5
    SPAN_PERC = 0.3

    trajectoryDataFrame = convertTrajectoryToDataFrame(trajectories)
    numOrigFrames = len(trajectoryDataFrame)

    

    newFrameIds = np.arange(0,numOrigFrames,1.0/INTERPOLATE_FACTOR)
    
    
    interpolatedDF=pd.DataFrame(columns=trajectoryDataFrame.columns)
    
    

    agentList = trajectoryDataFrame['agent_id'].unique()
    
    
    
    for agentID in agentList:
        singleAgentDataFrame = trajectoryDataFrame[trajectoryDataFrame['agent_id']==agentID]
        
        ewmSpan = int(SPAN_PERC * len(singleAgentDataFrame))
        singleAgentDataFrame[:]['x'] = singleAgentDataFrame['x'].ewm(span=ewmSpan).mean()
        singleAgentDataFrame[:]['y'] = singleAgentDataFrame['y'].ewm(span=ewmSpan).mean()
        singleAgentDataFrame[:]['vx'] = singleAgentDataFrame['vx'].ewm(span=ewmSpan).mean()
        singleAgentDataFrame[:]['vy'] = singleAgentDataFrame['vy'].ewm(span=ewmSpan).mean()
        singleAgentDataFrame[:]['ax'] = singleAgentDataFrame['ax'].ewm(span=ewmSpan).mean()
        singleAgentDataFrame[:]['ay'] = singleAgentDataFrame['ay'].ewm(span=ewmSpan).mean()
        
        agentTrajectory = singleAgentDataFrame[['x','y']].values
        agentVel = singleAgentDataFrame[['vx','vy']].values
        agentAcc = singleAgentDataFrame[['ax','ay']].values
        
        
        origAgentFrameId = singleAgentDataFrame[['frame_id']].values[::2]
        
        

        x,y= agentTrajectory[::2,0],agentTrajectory[::2,1]
        vx,vy= agentVel[::2,0],agentVel[::2,1]
        ax,ay= agentAcc[::2,0],agentAcc[::2,1]

        if(spline_interpolation):
            splX,splY=splrep(origAgentFrameId,x,k=5),splrep(origAgentFrameId,y,k=5)
            splVx,splVy=splrep(origAgentFrameId,vx,k=5),splrep(origAgentFrameId,vy,k=5)
            splAx,splAy=splrep(origAgentFrameId,ax,k=5),splrep(origAgentFrameId,ay,k=5)
            
            newAgentFrameIds = np.arange(origAgentFrameId.min(),origAgentFrameId.max(),1.0/INTERPOLATE_FACTOR)


            newX , newY = splev(newAgentFrameIds,splX), splev(newAgentFrameIds,splY)
            newVx , newVy = splev(newAgentFrameIds,splVx), splev(newAgentFrameIds,splVy)
            newAx , newAy = splev(newAgentFrameIds,splAx), splev(newAgentFrameIds,splAy)
            
            
            #Make sure frameIds are same as expected -> put all frameIds values in the nearest bins
            for idx, agentFrameId in enumerate(newAgentFrameIds):
                closestIdx=np.argmin(abs(newFrameIds - agentFrameId))    
                newAgentFrameIds[idx] = newFrameIds[closestIdx]

        else: 
            newX,newY,newVx,newVy,newAx,newAy,newAgentFrameIds = x,y,vx,vy,ax,ay,origAgentFrameId.flatten()


        tempdf=pd.DataFrame(data= {'x':newX,'y':newY,
                                'vx':newVx,'vy':newVy,
                                'ax':newAx,'ay':newAy,
                                'agent_id':agentID,
                                'frame_id':newAgentFrameIds,
                                'abs_v':np.sqrt(newVx**2+newVy**2),
                                'abs_a':np.sqrt(newAx**2+newAy**2)
                                })
        
        tempdf = tempdf.drop_duplicates(subset='frame_id', keep="first") #Drop duplicate frame_ids
        # trajectoryDataFrame = pd.merge(trajectoryDataFrame,tempdf,how='inner')
        interpolatedDF = interpolatedDF.append(tempdf)
        tempdf = tempdf.iloc[0:0]

        #plt.plot(newX,newY);
    interpolatedDF = interpolatedDF.sort_values(by=['frame_id'])
    interpolatedDF.reset_index(inplace=True)
    interpolatedTrajectories = convertDataFrametoTraj(interpolatedDF)
    return interpolatedTrajectories


def convertDataFrametoTraj(trajectoryDataFrame: pd.DataFrame):
    """Convert the input dataFrame to numpy trajectories

    Args:
        trajectoryDataFrame (pd.DataFrame): dataframe_columns = [
        "frame_id",
        "agent_id",
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "abs_v",
        "abs_a",
    ]

    Returns:
        [list]: [trajectories]
        [
        {id1: [x1, y1, vx1, vy1, ax1, ay1], id2: [x2, y2, vx2, vy2, ax2, ay2]},
        {id1: [..], id3: [..]},
        {id1: [..]},
    ]
    """
    uniqueId = trajectoryDataFrame["agent_id"].unique()
    frameIdList = trajectoryDataFrame['frame_id'].unique()
    keys = ['x','y','vx','vy','ax','ay']
    trajectoryList=[]
    ## Loop through all the frames in the dataframe
    for frameId in frameIdList:
        dfSingleFrame = trajectoryDataFrame[trajectoryDataFrame['frame_id']==frameId]
        agentDictionary={}
        ## Loop through all the agents in a given frame
        for singleAgentId in dfSingleFrame.agent_id.values:
            agentDictionary[singleAgentId] = dfSingleFrame[dfSingleFrame.agent_id==singleAgentId][keys].values.reshape(6,)
        trajectoryList.append(agentDictionary)

    return trajectoryList


def getTrajectoriesOnImage(trajectories: list,bevImg: np.ndarray,bevBoundary: list):
    """Plot the given trajectories of the vehicles on the given bird's eye view image bev

    Args:
        trajectoryDataFrame (pandas.DataFrame): [dataFrame of the stored trajectories]
        bevImg (np.ndarray): [Bird's eye view image]
        plot (bool, optional): [Whether to plot the image in this function]

    Returns:
        [np.ndarray]: [Final Image with trajectories plotted]
    """
    
    for wayPointsPerFrame in trajectories:
        if(len(wayPointsPerFrame)):
            bevImg = dp_utils.renderhomography(
                    wayPointsPerFrame, None, wayPointsPerFrame, bevImg, bevBoundary, False)
    return bevImg


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def extract_gtdata_from_bbox():

    scenario = 'tfl_65_dense'
    gt_folder = './../raw_data/ground_truth_data/'
    scenario_folder = gt_folder + scenario + "/"
    gt_data_file = gt_folder + scenario + '.json'

    gt_folder_entries = sorted(glob.glob(scenario_folder + "*.json"))

    frames = []
    for file in gt_folder_entries:
        if(file == scenario_folder + 'capture_data.json'):
            continue

        print(file)
        frame_data = json.load(open(file))
        extracted_frame_data = []
        for agent_data in frame_data:

            bbox = agent_data['bounding_box2d']
            obj_id = agent_data['agentid']
            agent_class = agent_data['classname']
            x = (bbox[0] + bbox[2])/2
            y = ((bbox[1] + bbox[3])/2 + bbox[3])/2

            # y = bbox[3]
            extracted_frame_data.append([x, y, obj_id, agent_class])

        #[[x,y,id1], [x,y,id2] ...}
        frames.append(extracted_frame_data)

    json.dump(frames, open(f"{gt_data_file}", "w"), cls=NumpyEncoder)

def extract_gtdata_from_worldlocs():

    scenario = 'tfl_55'
    gt_folder = './../raw_data/ground_truth_data/'
    scenario_folder = gt_folder + scenario + "/"
    gt_data_file = gt_folder + scenario + '.json'
    params_file = scenario_folder + "capture_data.json"
    params_data = json.load(open(params_file))

    gt_folder_entries = sorted(glob.glob(scenario_folder + "*.json"))

    scenario_data = {}
    frames = []
    for file in gt_folder_entries:
        if(file == scenario_folder + 'capture_data.json'):
            continue

        print(file)
        frame_data = json.load(open(file))
        extracted_frame_data = []
        for agent_data in frame_data:

            bbox = agent_data['bounding_box2d']
            obj_id = agent_data['agentid']
            agent_class = agent_data['classname']
            x = (bbox[0] + bbox[2])/2
            y = ((bbox[1] + bbox[3])/2 + bbox[3])/2

            loc = agent_data['location']
            obj_id = agent_data['agentid']
            agent_class = agent_data['classname']
            x_world = loc['x']
            y_world = loc['y']
            z_world = loc['z']
            extracted_frame_data.append([x_world, y_world, z_world, x, y, obj_id, agent_class])

        #[[x,y,id1], [x,y,id2] ...}
        frames.append(extracted_frame_data)

    scenario_data["frames"] = frames
    scenario_data["T"] = np.array(params_data["sensor_world_matrix"])
    scenario_data["K"] = np.array(params_data["intrinsics"])
    json.dump(scenario_data, open(f"{gt_data_file}", "w"), cls=NumpyEncoder)

def extract_bbox():
    scenario = 'round_3'
    gt_folder = './../raw_data/ground_truth_data/'
    scenario_folder = gt_folder + scenario + "/"
    gt_data_file = gt_folder + scenario + '_bbox.json'

    gt_folder_entries = sorted(glob.glob(scenario_folder + "*.json"))

    frames = []
    for file in gt_folder_entries:
        if (file == scenario_folder + 'capture_data.json'):
            continue
        frame_data = json.load(open(file))
        extracted_frame_data = {}
        for agent_data in frame_data:

            bbox = agent_data['bounding_box2d']
            extracted_frame_data[agent_data['agentid']] = bbox + [agent_data['classname']]

        # {id1: [x,y], id2: [x,y] ...}
        frames.append(extracted_frame_data)

    # [{id1: [x,y,vx,vy,ax,ay], id2:[x,y...]}]
    json.dump(frames, open(f"{gt_data_file}", "w"), cls=NumpyEncoder)

def verify_transform():

    scenario = 'round_2_55'
    gt_folder = './../raw_data/ground_truth_data/'
    scenario_folder = gt_folder + scenario + "/"
    gt_data_file = gt_folder + scenario + '.json'

    matrix_file = "./../raw_data/ground_truth_data/" + scenario + "/capture_data.json"
    matrix_data = json.load(open(matrix_file))
    T = matrix_data["sensor_world_matrix"]
    K = matrix_data["instrinsics"]

    gt_folder_entries = sorted(glob.glob(scenario_folder + "*.json"))

    frames = []
    for file in gt_folder_entries:
        if(file == scenario_folder + 'capture_data.json'):
            continue

        print(file)
        frame_data = json.load(open(file))
        extracted_frame_data = []
        for agent_data in frame_data:

            bbox = agent_data['bounding_box2d']
            obj_id = agent_data['agentid']
            agent_class = agent_data['classname']
            x = (bbox[0] + bbox[2])/2
            y = bbox[3]
            X = get_worldXY_coords(np.array([[x],[y],[1]]), T, K)
            x = X[0]
            y = X[1]
            extracted_frame_data.append([x, y, obj_id, agent_class])

        #[[x,y,id1], [x,y,id2] ...}
        frames.append(extracted_frame_data)

    json.dump(frames, open(f"{gt_data_file}", "w"), cls=NumpyEncoder)

if __name__ == '__main__':
    ipdb.set_trace()


    inputTrajectoryList = np.load("../../results/tracking_trajectories/fifth_craig_1.npy", allow_pickle=True)
    bevFile = "/home/stars/Code/detector/data/raw_data/images/bev_fifth_craig.png"
    bevImg = cv2.imread(bevFile)
    
    tf_id="fifth_craig_1"
    
    import json
    configFile= json.load(open("../../data/config.json"))
    bev_boundary=configFile[tf_id]["bev_boundary"]
    
    bevImg = getTrajectoriesOnImage(inputTrajectoryList,bevImg, bev_boundary)
    plt.figure()
    plt.imshow(bevImg)
    plt.savefig("before_smoothing_{}.png".format(tf_id)) 
    exit()
    interpolatedTrajectory = smoothTrajectories(inputTrajectoryList,spline_interpolation=False,interpolate_factor=1,span_perc=0.1)
    
    #interpolatedTrajectory=np.load("interpolated_trajectory.npy",allow_pickle=True)
    bevImg = cv2.imread(bevFile)
    bevImg = getTrajectoriesOnImage(interpolatedTrajectory,bevImg, bev_boundary)
    plt.figure()
    plt.imshow(bevImg)
    plt.savefig("after_smoothing_{}.png".format(0.1))
    
    #bevImg = cv2.imread(bevFile)
    #bevImg = getTrajectoriesOnImage(inputTrajectoryList,bevImg,[[351, 378], [878, 285], [943, 726], [430, 768], [351, 378]])
    #plt.figure()
    #plt.imshow(bevImg)
    #plt.show()
    
