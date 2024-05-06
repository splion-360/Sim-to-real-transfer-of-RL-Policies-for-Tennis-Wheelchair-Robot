import yaml
import numpy as np
import os
from gymnasium.spaces import Box


## Helper Functions

def load_config(cfg_file):
    cfg_file = os.path.join("/home/splion360/Desktop/project/Sim2Real/mujoco/wam_model/scripts/cfg",cfg_file)
    with open(cfg_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def distance(pos1,pos2):
    return np.linalg.norm(pos1-pos2)

def differential_drive(config, action):
    axle_length = config["differential_drive"].get("axle_length") # Original length x scaling factor (0.92575*1.3)
    diameter = config["differential_drive"].get("wheel_diameter") # Original diameter x scaling factor (0.62*1.2)
    v_des, w_des = action
    # Linear velocities
    vr,vl = (2*v_des + w_des*axle_length)/2,(2*v_des - w_des*axle_length)/2 
    # Angular velocities
    w1,w2 = 2*vr/diameter, 2*vl/diameter
    return np.array([w1,w2])

def getConstraintMap(config):
    from pandas import DataFrame as DF

    constraintMap = DF()
    scale_factor = config["mujoco"].get("factorofsafety") # Operating with only scale_factor% of the max limits

    constraintMap["j0"] = {"j_limit":np.array(config["dynamics"]["wam_base"].get("j_limit")),\
                                "v_limit":np.array(config["dynamics"]["wam_base"].get("v_limit"))*scale_factor,\
                                "a_limit":np.array(config["dynamics"]["wam_base"].get("a_limit"))*scale_factor}
    
    constraintMap["j1"] = {"j_limit":np.array(config["dynamics"]["wam_shoulder_pitch"].get("j_limit")),\
                        "v_limit":np.array(config["dynamics"]["wam_shoulder_pitch"].get("v_limit"))*scale_factor,\
                        "a_limit":np.array(config["dynamics"]["wam_shoulder_pitch"].get("a_limit"))*scale_factor}
    
    constraintMap["j2"] = {"j_limit":np.array(config["dynamics"]["wam_shoulder_yaw"].get("j_limit")),\
                        "v_limit":np.array(config["dynamics"]["wam_shoulder_yaw"].get("v_limit"))*scale_factor,\
                        "a_limit":np.array(config["dynamics"]["wam_shoulder_yaw"].get("a_limit"))*scale_factor}
    
    constraintMap["j3"] = {"j_limit":np.array(config["dynamics"]["wam_upper_arm"].get("j_limit")),\
                        "v_limit":np.array(config["dynamics"]["wam_upper_arm"].get("v_limit"))*scale_factor,\
                        "a_limit":np.array(config["dynamics"]["wam_upper_arm"].get("a_limit"))*scale_factor}
    
    constraintMap["j4"] = {"j_limit":np.array(config["dynamics"]["wam_fore_arm"].get("j_limit")),\
                        "v_limit":np.array(config["dynamics"]["wam_fore_arm"].get("v_limit"))*scale_factor,\
                        "a_limit":np.array(config["dynamics"]["wam_fore_arm"].get("a_limit"))*scale_factor}
    
    constraintMap["j5"] = {"j_limit":np.array(config["dynamics"]["wam_wrist_yaw"].get("j_limit")),\
                        "v_limit":np.array(config["dynamics"]["wam_wrist_yaw"].get("v_limit"))*scale_factor,\
                        "a_limit":np.array(config["dynamics"]["wam_wrist_yaw"].get("a_limit"))*scale_factor}
    
    constraintMap["j6"] = {"j_limit":np.array(config["dynamics"]["wam_wrist_pitch"].get("j_limit")),\
                        "v_limit":np.array(config["dynamics"]["wam_wrist_pitch"].get("v_limit"))*scale_factor,\
                        "a_limit":np.array(config["dynamics"]["wam_wrist_pitch"].get("a_limit"))*scale_factor}

    return constraintMap,scale_factor

def normalise(theta):
    if theta < 0: return theta + 2 * np.pi
    elif theta >= 2 * np.pi: return theta - 2 * np.pi
    else: return theta

def torque_denomalise(torques,max_torque):
    return (max_torque * (torques + 1))/2

def differential_vels_denormalize(v,w,max_v,max_w):
    normalised_v = (0.5 * (v+1) * (2* max_v)) - max_v
    normalised_w = (0.5 * (w+1) * (2* max_w)) - max_w
    return normalised_v,normalised_w


class Reward:
    def __init__(self,wtrobj):
        self.reward = 0
        self.wtrobj = wtrobj
    
    def groundReward(self,data):
        '''
        Function for aggregating the rewards accumulated for ground traversal
        
        '''
        print(self.wtrobj)
        robot_pos = self.wtrobj.data.qpos[7:9].copy()
        curr_goal_distance = distance(robot_pos,self.wtrobj.goal[:2])
        distance_rate = self.wtrobj.wtr_prev_distance  - curr_goal_distance
        reward = self.wtrobj.config["reward_scales"].get("distance_incentive") * distance_rate # Distance incentive
        payoff = 0

        if self.wtrobj.failed: 
            payoff = min(self.wtrobj.config["reward_scales"].get("terminal_payoff")) # Truncation if the robot goes out of bounds. 
        if self.wtrobj.reached:
            payoff = max(self.wtrobj.config["reward_scales"].get("terminal_payoff"))

        return reward + payoff + sum(data)
    
    def flyReward(self):
        '''
        Function for aggregating the rewards accumulated for arm manipulation
        '''
        reward,payoff = 0,0
        ## Distance Rate
        curr_target_distance = distance(self.wtrobj.data.xpos[15],self.wtrobj.goal)
        distance_rate = self.wtrobj.ee_prev_distance - curr_target_distance 

        if self.wtrobj.failed: 
            payoff = min(self.wtrobj.config["reward_scales"].get("terminal_payoff"))
        if self.wtrobj.reached: 
            payoff = max(self.wtrobj.config["reward_scales"].get("terminal_payoff"))
        reward = self.wtrobj.config["reward_scales"].get("distance_incentive") * distance_rate + payoff  

        return reward
    
    def control_cost(self, action):
        '''
        Action cost associated with control
        '''
        control_cost = float(self.wtrobj.config["reward_scales"].get("control_cost")) * np.sum(np.square(action))
        return control_cost
    
    def checkFlags(self):
        '''
        Predicate for toggling Episode termination flags
        
        '''
        ### For arm manipulation
        ee_pos = self.wtrobj.data.xpos[15].copy()
        arm_goal_distance = distance(ee_pos,self.wtrobj.goal)
        
        ## For wheelchair navigation
        robot_pos = self.wtrobj.data.qpos[7:9].copy()
        wheel_goal_distance = distance(robot_pos,self.wtrobj.goal[:2])
        x,y = robot_pos
        conditions = [ (x < -self.wtrobj.config["mujoco"].get("env_length") or x > self.wtrobj.config["mujoco"].get("env_length")),
                       (y < -self.wtrobj.config["mujoco"].get("env_length") or y > self.wtrobj.config["mujoco"].get("env_length"))]
        if any(conditions):
            self.wtrobj.failed = True
        
        if arm_goal_distance <= self.wtrobj.radius and wheel_goal_distance <= self.wtrobj.radius: 
            self.wtrobj.reached = True
