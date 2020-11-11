import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import norm
from collections import OrderedDict
from IPython import embed
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
import ipdb
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from utils.tracker_utils import *
from utils.dp_utils import *
from numpy import arctan2,cos,sin
# TODO: from demo_pr3 import bev_pos_bound,BIRDVIEW_OFFSET 
# bev_pos_bound=[300,930,50,535] # xmin, xmax, ymin, ymax
BIRDVIEW_OFFSET=100

states={} #dictionary keeping track of all ids and states 6x1
gatingThreshold=70

no_see_counter={}
see_counter={}
touched_ids=set() #ids seen in current frame
untouched_ids=set() #ids not seen in current frame
tracked_ids=set() #unique set of all tracked ids 
debug=False
FPS=60
no_see_thresh=1.5*FPS  #stop tracking an id not seen for past these many frames numSec * FPS
min_see_thresh=2  #a car is added as new only if it is seen atleast for three frames
vel_weight=0.5; pos_weight=1.0 #weights when computing euclidean distance
max_vel=1.3
max_acc=0.3
theta_thresh=0.1
global bev_boundary_polygon
global USE_MAP_PRIOR
global MAP_DATA
# TODO: Remove from below and include via a file
filename = "/home/stars/Code/detector/data/map/bev_fifth_craig.starsjson"
map_data = json.load(open(filename))


def get_cv_motion_model(dt=1):
    '''
    Constant Velocity model
    returns numpy array of motion dynamics  6x6 np array - x,y,vx,vy,ax,ay
    '''
    model=np.array([[1,0,dt,0,0,0],
                    [0,1,0,dt,0,0],
                    [0,0,1,0,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0]])
    return model

def get_ca_motion_model(dt=1):
    '''
    Constant Acceleration model
    returns numpy array of motion dynamics  6x6 np array - x,y,vx,vy,ax,ay
    '''
    model=np.array([[1,0,dt,0,1/2*dt**2,0],
                    [0,1,0,dt,0,1/2*dt**2],
                    [0,0,1,0,dt,0],
                    [0,0,0,1,0,dt],
                    [0,0,0,0,1,0],
                    [0,0,0,0,0,1]])

    return model

def clear_old_ids():
    '''
     remove cars not seen for lonng time
    '''
    temp=deepcopy(no_see_counter)
    for item in temp:
        if(no_see_counter[item]>=no_see_thresh):
            try:
                states.pop(item)
                no_see_counter.pop(item)
                see_counter.pop(item)
            except:
                pass

    # no_see_counter[untouched]>=no_see_threshold

def clear_out_ids():
    '''
    removes cars outside of the boundary
    '''
    tracker_ids = list(states.keys()) 
    for item in tracker_ids:
        # ipdb.set_trace()
        position=states[item][:2]
        x,y=position
        if(not bev_boundary_polygon.contains(Point(x,y))):
            try:
                if(debug):
                    print("item removed: {0} with position : {1}".format(item,position))
                states.pop(item)
                no_see_counter.pop(item)
                see_counter.pop(item)
            except:
                ipdb.set_trace()

def limit_theta(state,old_state,theta_thresh=0.1):
    
    old_x,old_y,old_vx,old_vy,old_ax,old_ay=old_state
    new_x,new_y,vx,vy,ax,ay=state
    dx,dy = new_x-old_x, new_y - old_y
    
    old_theta = arctan2(old_vy,old_vx)
    theta = arctan2(dy,dx)
    d_theta = theta - old_theta

    if(abs(d_theta)>=theta_thresh):
        new_x=old_x+old_vx*cos(theta_thresh)
        new_y=old_y+old_vy*sin(theta_thresh)
        new_vx,new_vy = new_x-old_x,new_y-old_y
        new_ax,new_ay = new_vx-old_vx,new_vy-old_vy
        state = np.array([new_x,new_y,new_vx,new_vy,new_ax,new_ay])
    return state


def predict_state(input_state,dt=1):
    '''
    Predicted state from the motion model
    input state- (6,) np.array
    predicted state- (6,) np.array
    '''
    old_state=input_state.copy()
    predicted_state=get_ca_motion_model(dt)@input_state.reshape(6,)
    predicted_state = limit_theta(predicted_state,old_state)
    return predicted_state

def remove_min(states):
    '''
        Function to remove vehicles not seen for min_threshold
    '''
    tracker_states=deepcopy(states)
    
    for id_ in states.keys():
        try: see_counter[id_]+=1
        except: see_counter[id_]=1
        if(see_counter[id_]<min_see_thresh):
            tracker_states.pop(id_)
    return tracker_states

def limit_state(id_):
    '''
    Makes velocity and acceleration within a threshold
    Input 6x1 state vector as nparray
    Output 6x1 limited state vector as nparray
    '''
    state=states[id_]
    # print("Velocity  of id {} = {}".format(id_,state[2:4]))
    # print("Acceleration  of id {} = {}".format(id_,state[4:]))
    for idx in range(2,4):
        if(abs(state[idx])>max_vel):
            if(debug):
                print("Velocity  of id {} greater than threshold. Limited from {} to max_vel {}".format(id_,state[idx],np.sign(state[idx])*max_vel))
            state[idx]=np.sign(state[idx])*max_vel
    for idx in range(4,6):
        if(state[idx]>max_acc):
            if(debug):
                print("Acc  of id {} greater than threshold. Limited from {} to max_acc {}".format(id_,state[idx],np.sign(state[idx])*max_acc))
            state[idx]=np.sign(state[idx])*max_acc

def tracker(states_sort,bev_bound_polygon,use_map_prior=False):
    ''' 
    Function to associate any wrong id.
    Consists of state of all actors in a world snapshot
    
    Associating via linear sum
    
    1.* input-data- {[id: state],[id:state],[]} - sort's prediction at current time t    
    
    2.* History maintained by tracker- 1. States- all states of ids states- {[id: state],[id:state],[]} - tracker prediction current time t-1
                          2. no_see_counter - count of ids not seen    - no_see_counter[untouched_ids]+=1 <<no_see_thresh-200 
    
    auxilliary data structures-    a. touched_ids - set of ids seen in the current frame
                                   b. untouched_ids- set of ids not currently seen 
        
    3.* states- states[predict for time t] for all states
        
    4. get a distance[weighted euclidean] matrix of each detection in sort against each tracked vehicle- MxN matrix where
        M is number of detections from sort
        N is number of vehicles tracked
        old_ids=[]
        new_ids=[]
        dist_mat=[old_ids X new_ids]

    5. associate_id- corrects sort input data - > {[id: state],[id:state],[]}-> {[new_id:new_state],[new_id: new_state],[]} at time t
        a. we use linear_sum from scipy-
           associated_tuple=linear_sum(dist_mat)
        b. post_check- 
            1. for all associations if dist>thresh, new_id
            2. if no association in new -> new_id
        
    6.* get touched -> untouched-> no_see_counter ->remove too old cars.
    
    7. return tracker_states

    *: same as before associating via linear_sum
    '''
    global bev_boundary_polygon
    global USE_MAP_PRIOR
    bev_boundary_polygon=bev_bound_polygon
    USE_MAP_PRIOR = use_map_prior
    states_sort=deepcopy(states_sort)
    
    
    global states,untouched_ids #TODO in a class
    # if(len(states.items())):  #first seen car frame
        # ipdb.set_trace()
    if(not len(states.items())):  #first seen car frame
        states=deepcopy(states_sort)
        for id_ in states.keys():
            no_see_counter[id_]=0
        # tracker_states=states.copy() # 
        tracker_states=remove_min(states)
        return tracker_states
        
    old_states = states.copy()

    ##3. predict for current time step t
    for id_ in states.keys():
        states[id_]=get_predicted_state(id_)
    
    untouched_ids=deepcopy(set(states.keys())) #no cars seen yet.
    
    #4,5 Association
    if(len(states_sort)):   #If there are no input vehicles, then nothing to associate
        data_association(states_sort,old_states)
    

    ###6. Stop tracking cars not seen for a long time    
    clear_old_ids() #remove from tracker those cars which have not been seen for a long time
    clear_out_ids()
    ##Update 'to be tracked' ids not seen in this frame
    for id_ in untouched_ids:
        try: no_see_counter[id_]+=1 #no_see_counter
        except:  pass
    
    #Limit velocities in pixel space and remove the vehicles not seen for atleast threshold frames
    _=[limit_state(id_) for id_ in states]

    tracker_states=states.copy()
    #tracker_states=remove_min(states)

    return tracker_states

def fuse_state(state1,state2,old_state):
    global USE_MAP_PRIOR
    fused_state = (state1+state2)/2
    # fused_state=state2
    x,y=fused_state[:2]
    vx,vy = fused_state[2:4]
    ax,ay=fused_state[4:6]
    
    if(USE_MAP_PRIOR):
        lane_id = get_lane_id_from_point((x,y),map_data)
        if(lane_id is not None and lane_id is not -1):
            for entry in map_data:
                if(int(entry["id"])==lane_id):
                    center_line=entry['center_line']
                    break
            new_x,new_y = get_closest_point_on_line((x,y),center_line)[0]
            new_vx,new_vy = (vx-x+new_x,vy-y+new_y)  #dt is assumed to be 1
            new_ax,new_ay = (ax-vx+new_vx,ay-vy+new_vy)  #dt is assumed to be 1
            map_fused_state=np.array([new_x,new_y,new_vx,new_vy,new_ax,new_ay])
            return map_fused_state
        
    return fused_state


def data_association(states_sort, old_states, gating_threshold=gatingThreshold):
    # Extract pose from states
    '''
        Input: States_sort- SORT dictionary of states - {id:[x,y,vx,vy],id:[x,y,vx,vy],....}
        Output:States_tr- dictionary of states containing tracker predictions- {id:[x,y,vx,vy],[x,y,vx,vy],....]
    '''
    
    sort_states =  np.asarray(list(states_sort.values()))#np.asarray([state for state in states_sort.values()]) #2x6 np array
    tracker_states = np.asarray(list(states.values()))#np.asarray([state for state in states_tr.values()]) #2x6 np array
    tracker_ids=list(states.keys());sort_ids=list(states_sort.keys()); #list of ids
    
    # Formulate cost matrix - weighted euclidean distance between 
    cost_pos = cdist(tracker_states[:,:2].reshape(-1,2), sort_states[:,:2].reshape(-1,2), 'euclidean') #Confirmed
    cost_vel = cdist(tracker_states[:,2:4].reshape(-1,2), sort_states[:,2:4].reshape(-1,2), 'euclidean')

    cost=pos_weight*cost_pos+vel_weight*cost_vel #rows are trackers and columns are sort
    
    ##Setting distance between same IDs from SORT&tracker as 0
    same_states=set(states_sort.keys()).intersection(set(states.keys()))
    same_idx=tuple([(tracker_ids.index(item),sort_ids.index(item) ) for item in same_states])
    for item in same_idx:
        cost[item]=0
    

    #Computing associations
    row_ind, col_ind = linear_sum_assignment(cost) 
    gate_ind = np.where(cost[row_ind, col_ind] < gating_threshold)[0] 

    #Get associated poses/ids
    associated_ids=np.asarray([(tracker_ids[r],sort_ids[c]) for r,c in zip(row_ind,col_ind)])[gate_ind]   # a list of tuples with associated tracker_ids,SORT_ids

    #Update tracker states dictionary
    #a. Pass associated states from above, and fuse prediction->add to states
    #b. For all remaining states in tracker-> update no_see_counter
    #Fuse same vehicles
    for tr_id,sort_id in associated_ids:
        states[tr_id]=fuse_state(states[tr_id],states_sort[sort_id],old_states[tr_id])
        untouched_ids.remove(tr_id)
        no_see_counter[tr_id]=0
        states_sort.pop(sort_id) #removing cars seen from sort
    
    #All remaining cars in states_sort are new. Add these new cars
    states.update(states_sort)
    no_see_counter.update(dict.fromkeys(states_sort.keys(), 0))


def get_predicted_state(id_):
    '''
    Returns predicted state for the input id_
    id_ [int]: vehicle id
    Output [np.ndarray]: 6x1 predicted state or None if id_ not available
    '''
    if(states.get(id_) is not None):
        return predict_state(states[id_]) #predicting the next state using the motion model
    else: return None


if __name__=="__main__":
    
    # input_data_list=np.load('../realWorldAnalysis/data/fifth_craig_orig_pr7.npy',allow_pickle=True)
    input_data_list=np.load('../realWorldAnalysis/data/fifth_craig_orig_pr7.npy',allow_pickle=True)
    ''' 
    consists of state of all actors in a world snapshot
    input_data= [
                   {id:[x,y,vx,vy],id:[x,y,vx,vy],....},
                    {id:[x,y,vx,vy],id:[x,y,vx,vy],....},
                   {...},
                   .,
                   ]
    '''
    ipdb.set_trace()
    import mapping.convert_labelme_to_stars as convert
    filename = "../../data/bev_62_.starsjson"
    map_data = json.load(open(filename))
    import cv2
    bev_image= cv2.imread("/home/stars/Code/detector/data/raw_data/images/bev_62_.png")
    bev_pos_bound = [800, 1370, 290, 790]
    traj_without_tracking=np.load("/home/stars/Code/detector/results/tracking_trajectories/carla_62_only_sort.npy",allow_pickle=True)
    traj_with_tracking=np.load("/home/stars/Code/detector/results/tracking_trajectories/carla_62_without_map.npy",allow_pickle=True)
    traj_with_tracking_with_map=np.load("/home/stars/Code/detector/results/tracking_trajectories/carla_62.npy",allow_pickle=True)
    
    bev_boundary=[[880, 820], [880, 330], [1410, 330], [1410, 568], [1262, 820], [880, 820]]
    
    # plotted_onlysort_img = getTrajectoriesOnImage(traj_without_tracking,bev_image.copy(),bev_boundary)
    # plotted_tracking_img = getTrajectoriesOnImage(traj_with_tracking,bev_image.copy(),bev_boundary)
    plotted_tracking_img_with_map = getTrajectoriesOnImage(traj_with_tracking_with_map,bev_image.copy(),bev_boundary)
    # cv2.imwrite("plotted_onlysort_img.png",plotted_onlysort_img)
    # cv2.imwrite("plotted_tracking_img.png",plotted_tracking_img)
    cv2.imwrite("plotted_tracking_img_with_map.png",plotted_tracking_img_with_map)
    # plotted_tracking_map_img = getTrajectoriesOnImage(traj_without_tracking,bev_image,bev_pos_bound)

    center_line = map_data['center_line']
    filtered_data_list=[]
    bev_pos_bound= [580, 1380, 310, 790]
    
    # embed()
    IDX_THRESH=545
    ipdb.set_trace()
    for idx,data in enumerate(input_data_list):
        if(debug):
            if(idx>=IDX_THRESH):
                print("Started frame idx {}".format(idx))
                print("INPUT sort States: {}".format(input_data_list[idx]))
                print("Tracker States: {}".format(states))
                print(no_see_counter)
                # ipdb.set_trace()
        output_data=deepcopy(tracker(data,bev_pos_bound)   ) 
        filtered_data_list.append(output_data)
    plotted_img = getTrajectoriesOnImage(output_data,bev_image,Polygon(bev_boundary))
    np.save('tracking_filtering/tfl_65_dense_with_tracker.npy',filtered_data_list)
