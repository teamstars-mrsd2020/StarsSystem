import json
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

class BevTracker:
    
    def __init__(self,params,bev_boundary_polygon):
        self._params=params
        self._bev_boundary_polygon=bev_boundary_polygon
        self._no_see_counter={}
        self._see_counter={}
        self._lane_history={}
        self._touched_ids=set() #ids seen in current frame
        self._untouched_ids=set() #ids not seen in current frame
        self._tracked_ids=set() #unique set of all tracked ids 
        self._debug=False
        self._states={}
        self._states_sort={}
        self._trajectories=[]
        
        self._old_states={}
        self._FPS=params["FPS"]
        self._no_see_thresh=params["FPS"]*1.5  #stop tracking an id not seen for past these many frames numSec * FPS
        self._min_see_thresh=params["min_see_thresh"]  #a car is added as new only if it is seen atleast for three frames
        self._vel_weight=params["vel_weight"]
        self._pos_weight=params["pos_weight"] #weights when computing euclidean distance
        self._max_vel=params["max_vel"]
        self._max_acc=params["max_acc"]
        self._use_map_prior=params["use_map_prior"]
        self._theta_thresh=params["theta_thresh"]
        self._gating_threshold=params["gating_threshold"]
        self._dt=params["dt"]
        self._map_data= json.load(open(params["map_file"]))
        self._agent_lanes_history={}
        self._tracked_agents_trajectory_history={}
        if(self._use_map_prior):
            self._point_on_intersection = self._get_point_on_intersection()

    def _get_point_on_intersection(self):
        for entry in self._map_data:
            if(entry["id"]==-1):
                return entry["polygon"][0]
        
        return np.array([834,330]) #intersection -1 is not added to fifth craig map; waiting for the fix

    def _get_cv_motion_model(self):
        '''
        Constant Velocity model
        returns numpy array of motion dynamics  6x6 np array - x,y,vx,vy,ax,ay
        '''
        dt=self._dt
        model=np.array([[1,0,dt,0,0,0],
                        [0,1,0,dt,0,0],
                        [0,0,1,0,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]])
        return model

    def _get_ca_motion_model(self,dt):
        '''
        Constant Acceleration model
        returns numpy array of motion dynamics  6x6 np array - x,y,vx,vy,ax,ay
        '''
        dt=self._dt
        model=np.array([[1,0,dt,0,1/2*dt**2,0],
                        [0,1,0,dt,0,1/2*dt**2],
                        [0,0,1,0,dt,0],
                        [0,0,0,1,0,dt],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,1]])

        return model

    def _clear_old_ids(self):
        '''
        remove cars not seen for lonng time
        '''
        temp=deepcopy(self._no_see_counter)
        for item in temp:
            if(self._no_see_counter[item]>=self._no_see_thresh):
                try:
                    self._states.pop(item)
                except: pass
                try:    
                    self._no_see_counter.pop(item)
                except:
                    pass
                    self._see_counter.pop(item)
                try:
                    self._tracked_agents_trajectory_history.pop(item)
                except:
                    pass
                try:
                    self._agent_lanes_history.pop(item)
                except:
                    pass
                # try:
                #     self._states.pop(item)
                #     self._no_see_counter.pop(item)
                #     self._see_counter.pop(item)
                #     self._tracked_agents_trajectory_history.pop(item)
                # except:
                #     pass

        # no_see_counter[untouched]>=no_see_threshold

    def _clear_out_ids(self):
        '''
        removes cars outside of the boundary
        '''
        tracker_ids = list(self._states.keys()) 
        for item in tracker_ids:
            
            position=self._states[item][:2]
            x,y=position
            if(not self._bev_boundary_polygon.contains(Point(x,y))):
            
                if(self._debug):
                    print("item removed: {0} with position : {1}".format(item,position))
                try:
                    self._states.pop(item)
                except: pass
                try:    
                    self._no_see_counter.pop(item)
                except:
                    pass
                    self._see_counter.pop(item)
                try:
                    self._tracked_agents_trajectory_history.pop(item)
                except:
                    pass
                try:
                    self._agent_lanes_history.pop(item)
                except:
                    pass
                    

    def _limit_theta(self,state,old_state):
        
        old_x,old_y,old_vx,old_vy,old_ax,old_ay=old_state
        new_x,new_y,vx,vy,ax,ay=state
        dx,dy = new_x-old_x, new_y - old_y
        
        old_theta = arctan2(old_vy,old_vx)
        theta = arctan2(vy,vx)
        d_theta = theta - old_theta

        if(abs(d_theta)>=self._theta_thresh):
            new_x=old_x+old_vx*cos(self._theta_thresh)
            new_y=old_y+old_vy*sin(self._theta_thresh)
            new_vx,new_vy = new_x-old_x,new_y-old_y
            new_ax,new_ay = new_vx-old_vx,new_vy-old_vy
            state = np.array([new_x,new_y,new_vx,new_vy,new_ax,new_ay])
        return state

    
    def _predict_state(self,input_state):
        '''
        Predicted state from the motion model
        input state- (6,) np.array
        predicted state- (6,) np.array
        '''
        old_state=input_state.copy()
        predicted_state=self._get_cv_motion_model()@input_state.reshape(6,)
        # predicted_state = self._limit_theta(predicted_state,old_state)
        return predicted_state



    def _post_process(self):
        
        ###Stop tracking cars not seen for a long time    
        self._clear_old_ids() #remove from tracker those cars which have not been seen for a long time
        self._clear_out_ids()
        
        ##Update 'to be tracked' ids not seen in this frame
        for id_ in self._untouched_ids:
            try: self._no_see_counter[id_]+=1 #no_see_counter
            except:  pass
        
        
        

        #remove the vehicles not seen for atleast threshold frames
        
        #Limit velocities in pixel space 
        _=[self._limit_state(id_) for id_ in self._states]

        self._compute_avg_vel()
        #Put all vehicles to their map center lane and align them to lane direction
        if(self._use_map_prior):
            self._move_to_center_lane()
        
        
        
        #add the detected states to agents' history to keep track of all trajectories of agents
        self._append_to_agent_history()

    def _compute_avg_vel(self,n=5):
        for id_ in self._tracked_agents_trajectory_history:
            traj=self._tracked_agents_trajectory_history[id_][-n:]
            try:
                avg_vel=(np.array(traj[-1])-np.array(traj[0]))/n
                self._states[id_][2:4]=avg_vel
            except: pass
            try:
                vel_hist=np.diff(traj,axis=0)
                avg_acc=(np.array(vel_hist[-1])-np.array(vel_hist[0]))/n
                self._states[id_][4:6]=avg_acc
                
            except: pass
            
            
            

    def _align_vel_with_lane(self,id_,center_line,direction):
        p1,p2=center_line
        p = self._point_on_intersection
        if(norm(p1-p)>norm(p2-p)):
            far_point=p1
            near_point=p2
        else:
            far_point=p2
            near_point=p1
        lane_direction_vector=np.array([far_point[0]-near_point[0],(far_point[1]-near_point[1])])
        if(direction=='INWARDS'):
            lane_direction_vector*=-1
        u=self._states[id_][2:4]
        vel_mag=norm(self._states[id_][2:4])
        unit_lane_vec=lane_direction_vector/norm(lane_direction_vector)
        # proj_of_u_on_v = (np.dot(u, v)/norm(v)**2)*v
        # self._states[id_][2:4] = (proj_of_u_on_v/norm(proj_of_u_on_v))*vel_mag
        
        if(vel_mag>1e-6):
            self._states[id_][2:4] = unit_lane_vec*vel_mag
        else: self._states[id_][2:4] = unit_lane_vec*1e-6
                    

        
    def _remove_min(self):
        '''
            Function to remove vehicles not seen for min_threshold
        '''
        tracker_states=deepcopy(self._states)
        
        for id_ in self._states.keys():
            try: self._see_counter[id_]+=1
            except: self._see_counter[id_]=1
            if(self._see_counter[id_]<self._min_see_thresh):
                tracker_states.pop(id_)
                try:
                    self._tracked_agents_trajectory_history.pop(id_)
                except: pass
        return tracker_states

    def _limit_state(self,id_):
        '''
        Makes velocity and acceleration within a threshold
        Input 6x1 state vector as nparray
        Output 6x1 limited state vector as nparray
        '''
        state=self._states[id_]
        # print("Velocity  of id {} = {}".format(id_,state[2:4]))
        # print("Acceleration  of id {} = {}".format(id_,state[4:]))

        #Limit velocity
        old_vel=state[2:4].copy()
        max_velxy = np.max(abs(state[2:4])) #minimum of absolute velocities in x and y
        if(max_velxy>self._max_vel):
            
            ratio = self._max_vel/max_velxy
            if(self._debug):
                print("Velocity  of id {} greater than threshold. Limited from {} to self._max_vel {}".format(id_,state[2:4],np.sign(state[idx])*self._max_vel))
            state[2:4]*=ratio        
        
        # #Limit positon
        # state[:2]+=state[2:4]-old_vel
        
        #Limit acceleration
        for idx in range(4,6):
            if(state[idx]>self._max_acc):
                if(self._debug):
                    print("Acc  of id {} greater than threshold. Limited from {} to self._max_acc {}".format(id_,state[idx],np.sign(state[idx])*self._max_acc))
                state[idx]=np.sign(state[idx])*self._max_acc

        #Limit angle
        if id_ in self._old_states:
            self._states[id_] = self._limit_theta(self._states[id_],self._old_states[id_])

    def tracker(self,states_sort):
        ''' 
        
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
        
        self._states_sort=deepcopy(states_sort)
        
        # if(len(states.items())):  #first seen car frame
            # ipdb.set_trace()
        if(not len(self._states.items())):  #first seen car frame
            self._states=deepcopy(states_sort)
            for id_ in self._states.keys():
                self._no_see_counter[id_]=0
            # tracker_states=states.copy() # 
            self._append_to_agent_history()
            tracker_states=self._remove_min()
            return tracker_states,self._tracked_agents_trajectory_history.copy()
            
        self._old_states = self._states.copy()

        ##3. predict for current time step t
        for id_ in self._states.keys():
            self._states[id_]=self._predict(id_)
        
        self._untouched_ids=deepcopy(set(self._states.keys())) #no cars seen yet.
        
        #4,5 Association
        if(len(self._states_sort)):   #If there are no input vehicles, then nothing to associate
            self._data_association()
        
        self._post_process()

        self._trajectories.append(self._states.copy())

        return self._states.copy(),self._tracked_agents_trajectory_history.copy()

    def _move_to_center_lane(self):
        
        for vehicle_id in self._states:
            x,y,vx,vy,ax,ay=self._states[vehicle_id]
            lane_id = get_lane_id_from_point((x,y),self._map_data)
            
            try:
                self._agent_lanes_history[vehicle_id].append(lane_id)
            except:
                self._agent_lanes_history[vehicle_id]=[lane_id]
            
            if(lane_id is not None and lane_id is not -1):
                for entry in self._map_data:
                    if(int(entry["id"])==lane_id):
                        center_line=entry['center_line']
                new_x,new_y = get_closest_point_on_line((x,y),center_line)[0]
                new_vx,new_vy = (vx-x+new_x,vy-y+new_y)  #dt is assumed to be 1
                new_ax,new_ay = (ax-vx+new_vx,ay-vy+new_vy)  #dt is assumed to be 1
                self._states[vehicle_id]=np.array([new_x,new_y,new_vx,new_vy,new_ax,new_ay])
            
                #Align velocity with lane
                if(self._use_map_prior):
                    self._align_vel_with_lane(vehicle_id,center_line,entry["direction"])
        
    def _append_to_agent_history(self):
        """
        Appends the agents' trajectory in current frame, to their history
        """
        for id in self._states:
            x,y=self._states[id][:2].astype(np.int32)
            # if id in self._tracked_agents_trajectory_history:
            try:
                self._tracked_agents_trajectory_history[id].append((x,y))
            except:
                self._tracked_agents_trajectory_history[id]= [(x,y)]

    def _update_prediction(self,state1,state2,old_state):
        fused_state = (state1+state2)/2
        # fused_state=state2
        x,y=fused_state[:2]
        vx,vy = fused_state[2:4]
        ax,ay=fused_state[4:6]
        
        if(self._use_map_prior):
            lane_id = get_lane_id_from_point((x,y),self._map_data)
            if(lane_id is not None and lane_id is not -1):
                for entry in self._map_data:
                    if(int(entry["id"])==lane_id):
                        center_line=entry['center_line']
                new_x,new_y = get_closest_point_on_line((x,y),center_line)[0]
                new_vx,new_vy = (vx-x+new_x,vy-y+new_y)  #dt is assumed to be 1
                new_ax,new_ay = (ax-vx+new_vx,ay-vy+new_vy)  #dt is assumed to be 1
                map_fused_state=np.array([new_x,new_y,new_vx,new_vy,new_ax,new_ay])
                return map_fused_state
            
        return fused_state

    def _check_feasibility(self, associated_ids):
        pass
    
    def _data_association(self):
        # Extract pose from states
        '''
            Input: States_sort- SORT dictionary of states - {id:[x,y,vx,vy],id:[x,y,vx,vy],....}
            Output:States_tr- dictionary of states containing tracker predictions- {id:[x,y,vx,vy],[x,y,vx,vy],....]
        '''
        
        sort_states =  np.asarray(list(self._states_sort.values()))#np.asarray([state for state in states_sort.values()]) #2x6 np array
        tracker_states = np.asarray(list(self._states.values()))#np.asarray([state for state in states_tr.values()]) #2x6 np array
        tracker_ids=list(self._states.keys());sort_ids=list(self._states_sort.keys()); #list of ids
        
        # Formulate cost matrix - weighted euclidean distance between 
        cost_pos = cdist(tracker_states[:,:2].reshape(-1,2), sort_states[:,:2].reshape(-1,2), 'euclidean') #Confirmed
        cost_vel = cdist(tracker_states[:,2:4].reshape(-1,2), sort_states[:,2:4].reshape(-1,2), 'euclidean')

        cost=self._pos_weight*cost_pos+self._vel_weight*cost_vel #rows are trackers and columns are sort
        
        ##Setting distance between same IDs from SORT&tracker as 0
        same_states=set(self._states_sort.keys()).intersection(set(self._states.keys()))
        same_idx=tuple([(tracker_ids.index(item),sort_ids.index(item) ) for item in same_states])
        for item in same_idx:
            cost[item]=0
        

        #Computing associations
        row_ind, col_ind = linear_sum_assignment(cost) 
        gate_ind = np.where(cost[row_ind, col_ind] < self._gating_threshold)[0] 

        #Get associated poses/ids
        associated_ids=np.asarray([(tracker_ids[r],sort_ids[c]) for r,c in zip(row_ind,col_ind)])[gate_ind]   # a list of tuples with associated tracker_ids,SORT_ids
        # self._check_feasibility(associated_ids)

        #Update tracker states dictionary
        #a. Pass associated states from above, and fuse prediction->add to states
        #b. For all remaining states in tracker-> update no_see_counter
        #Fuse same vehicles
        for tr_id,sort_id in associated_ids:
            self._states[tr_id]=self._update_prediction(self._states[tr_id],self._states_sort[sort_id],self._old_states[tr_id])
            self._untouched_ids.remove(tr_id)
            self._no_see_counter[tr_id]=0
            self._states_sort.pop(sort_id) #removing cars seen from sort
        
        #All remaining cars in states_sort are new. Add these new cars
        self._states.update(self._states_sort)
        self._no_see_counter.update(dict.fromkeys(self._states_sort.keys(), 0))


    def _predict(self,id_):
        '''
        Returns predicted state for the input id_
        id_ [int]: vehicle id
        Output [np.ndarray]: 6x1 predicted state or None if id_ not available
        '''
        if(self._states.get(id_) is not None):
            return self._predict_state(self._states[id_]) #predicting the next state using the motion model
        else: return None


