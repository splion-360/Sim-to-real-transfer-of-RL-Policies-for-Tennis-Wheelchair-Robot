from typing import Dict
import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import logging
import yaml
from helper import *



class WTR3DReacherEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ]
    
    }
    config = load_config("wam7_3Dreacher.yaml")
    DEFAULT_CAMERA_CONFIG = {"lookat": np.array(config["mujoco"].get("camera"))}
    full_path = "/home/splion360/Desktop/project/Sim2Real/mujoco/wam_model"
    def __init__(
        self,
        xml_file: str = os.path.join(full_path,"robots/wam7_armreacher.xml"),
        frame_skip: int = config["mujoco"].get("frame_skip"),
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = config["mujoco"].get("reset_noise_scale"),
        error:float = config["mujoco"].get("error"),
        radius:float = config["mujoco"].get("radius"),
        render_mode: str = None,
        config:Dict[str,float] = config,
        **kwargs):
        
        utils.EzPickle.__init__(self,
        xml_file,
        frame_skip,
        default_camera_config,
        reset_noise_scale,
        error,
        radius,
        render_mode,
        config,
        **kwargs
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            render_mode=render_mode,
            **kwargs
            )
        


        ## Some commmon set properties for use
        self.config = config
        self.reset_noise_scale = float(config["mujoco"].get("reset_noise_scale"))
        self.error = float(config["mujoco"].get("error"))
        self.radius = radius
        self.failed,self.reached = False, False
        self.render_mode = render_mode
        self.goal = np.zeros(3)
        self.metadata = {"render_modes":["human","rgb_array","depth_array"],"render_fps":int(np.round(1.0/self.dt))}


        ## WAM arm specific parameters definition
        self.max_torque = config["dynamics"].get("joint_torque_limits")
        self.vf = config["dynamics"].get("goal_velocity")
        self.max_reachable_distance = config["dynamics"].get("max_reachable_distance") # Length of the upright Wam arm
        qpos = config["mujoco"].get("pos")
        self.init_pos = np.array(qpos.split(" "),dtype=np.float64)
        self.constraintMap,self.scale = getConstraintMap(config)
        self.ee_prev_distance = 0

        ## Wheelchair specific parameters definition
        self.t_accepted = 0
        self.a_lin_max = config["dynamics"].get("max_linear_acceleration")
        self.a_ang_max = config["dynamics"].get("max_angular_acceleration")
        self.max_v,self.max_w = config["dynamics"].get("max_linear_velocity"), config["dynamics"].get("max_angular_velocity")
        self.diagnoal_length = self.config["mujoco"].get("env_length") * 2 * np.sqrt(2)
        self.wtr_prev_distance = 0

        ## Simulation specific common variables
        action_size = 9  # Action Space -> 7 Joint Torques, Observation Space -> (7 Joint angles + 7 Joint velocities + 1 distance to goal)
        self.observation_structure = {
            "skipped_qpos": 0,
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,}
        self.reward = Reward(self)   
        self.observation_space = self._get_observation_space()
        self.action_space = Box(low = -1, high = 1, shape=(action_size,), dtype=np.float32)
        self.datatoplot = self._get_info()     

        logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
        self.env_logger = logging.getLogger()

    
    def _get_observation_space(self):
        
        ## Observations specific to wheelchair 
        min_gd = np.array([-np.inf])
        max_gd = np.array([np.inf])
        min_rel = np.array([-np.inf])
        max_rel = np.array([np.inf])
        min_yaw = np.array([-np.inf])
        max_yaw = np.array([np.inf])
        
        min_lin_vel = np.array([-self.max_v]*3)
        max_lin_vel = np.array([self.max_v]*3)

        min_ang_vel = np.array([-self.max_w]*3)
        max_ang_vel = np.array([self.max_w]*3)

        min_vel = np.concatenate((min_lin_vel,min_ang_vel))
        max_vel = np.concatenate((max_lin_vel,max_ang_vel))

        ## Observations specific to wam arm 

        size = len(self.constraintMap.T)
        max_ee_pos,min_ee_pos = np.array([np.inf]*3),np.array([-np.inf]*3)
        max_goal,min_goal = np.array([np.inf]*3),np.array([-np.inf]*3)
        
        max_joint_lim, min_joint_lim = np.zeros(size), np.zeros(size)
        for i in range(size):
            max_joint_lim[i] = self.constraintMap[f"j{i}"].loc["j_limit"].max() 
            min_joint_lim[i] = self.constraintMap[f"j{i}"].loc["j_limit"].min()

        max_vel_lim, min_vel_lim = np.zeros(size), np.zeros(size)
        max_acc_lim, min_acc_lim = np.zeros(size), np.zeros(size)
        for i in range(size):
            max_vel_lim[i] = self.constraintMap[f"j{i}"].loc["v_limit"].max() +  self.constraintMap[f"j{i}"].loc["a_limit"].max() * self.dt
            min_vel_lim[i] = self.constraintMap[f"j{i}"].loc["v_limit"].min() +  self.constraintMap[f"j{i}"].loc["a_limit"].min() * self.dt
            min_acc_lim[i] = self.constraintMap[f"j{i}"].loc["a_limit"].min()
            max_acc_lim[i] = self.constraintMap[f"j{i}"].loc["a_limit"].max()

        ## Combine the observations
        
        min_obs = np.hstack((min_gd,min_rel,min_yaw,min_vel,min_joint_lim, min_vel_lim, min_acc_lim, *min_ee_pos, *min_goal))
        max_obs = np.hstack((max_gd,max_rel, max_yaw, max_vel,max_joint_lim, max_vel_lim, max_acc_lim, *max_ee_pos, *max_goal))

        return Box(low = min_obs, high = max_obs, dtype = np.float64)
    
    def _get_info(self):
        info = {}
        info['total_reward'] = float('-inf')
        info['linear_velocity_error'] = float('-inf')
        info['angular_velocity_error'] = float('-inf')
        info['wheelchair_linear_acceleration'] = float('-inf')
        info['wheelchair_angular_acceleration'] = float('-inf')
        info['timetoreach'] = float('-inf')
        info['wheel_1_velocity'] = float('-inf')
        info['wheel_2_velocity'] = float('-inf')
        info['ground_positional_error'] = float('-inf')

        for i in range(len(self.constraintMap.T)):
            info[f'joint_positions_{i}'] = float('-inf')
            info[f'arm_angular_velocity_{i}'] = float('-inf')
            info[f'arm_angular_acceleration_{i}'] = float('-inf')
        info['arm_positional_error'] = float('-inf')
        info['is_success'] = False
        return info
    
    def reset(self,seed = None, options = None):
        np.random.seed(seed)
        # Random initialization of the marker coordinates
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        # Setting the flags to false
        self.reached,self.failed = False, False

        # Resetting the simulation data
        self.data.qpos[:] = self.init_pos.copy()
        self.data.qvel[:] = np.zeros(self.model.nv)
        self.data.ctrl[:] = np.zeros(self.model.nu)
        self.data.time = 0.0

        ## Generate x,y coordinates based on the goal coordinate generation mechanism from the wheelchair navigation
        tx,ty = np.random.randint(low=-self.config["mujoco"].get("coordinate_limits"),high=self.config["mujoco"].get("coordinate_limits")) + \
        np.random.uniform(low=noise_low, high=noise_high),\
        np.random.randint(low=-self.config["mujoco"].get("coordinate_limits"),high=self.config["mujoco"].get("coordinate_limits")) +\
        np.random.uniform(low=noise_low, high=noise_high)
        tz = Box(low = 0, high = self.max_reachable_distance , shape=(1,), dtype = np.float64).sample()

        self.goal = np.array((tx,ty,*tz))
        self.env_logger.info(f"Target coordinates: {self.goal}")
        self.model.site('ball_coord').pos = self.goal
        
        self.set_state(self.data.qpos,self.data.qvel)
        self.wtr_prev_distance = distance(self.goal[:2],self.data.qpos[7:9].copy())
        self.ee_prev_distance  = distance(self.goal,self.data.xpos[15].copy())
        
        observation = self._get_obs()
        info = self._get_reset_info()

        self.t_accepted = (distance(self.goal[:2],self.data.qpos[7:9].copy()) / self.max_v) + (self.sweep/self.max_w) + \
                           self.config["reward_scales"].get("grace_time_period")

        return observation, info
    
    def _get_obs(self):

        ## Wheelchair specific observations 
        robot_pos = self.data.qpos[7:9].copy()
        qx,qy,qz,qw =  self.data.qpos[10:14].copy() # Quaternions 

        robot_vel = self.data.qvel[6:12].copy() # Robot Velocities

        ## Normalise the robot velocities between -1 and 1
        robot_vel[:3] /= self.max_v
        robot_vel[3:] /= self.max_w

        yaw = np.arctan2(2*(qx*qy + qw*qz),1-2*(qy**2 + qz**2)) # Euler z
        if yaw < 0: yaw += 2*np.pi # Wrapping angles between [0,2*pi)

        rel_x,rel_y = self.goal[0]-robot_pos[0], self.goal[1]-robot_pos[1]
        ## Compute the relative angle between the robot and the target [0,2*pi)
        uncalibrated_angle = np.arctan(rel_y/rel_x + self.error)
        if rel_x > 0 and rel_y > 0: 
            theta = uncalibrated_angle
        elif rel_x > 0 and rel_y < 0: 
            theta = 2*np.pi + uncalibrated_angle
        elif rel_x < 0 and rel_y < 0 or rel_x < 0 and rel_y > 0:
            theta = np.pi + uncalibrated_angle
        
        elif rel_x == 0 and rel_y > 0: 
            theta = 0.5 * np.pi
        elif rel_x == 0 and rel_y < 0:
            theta = 1.5 * np.pi
        elif rel_x > 0  and rel_y == 0:
            theta = 0
        else: 
            theta = np.pi
        
        ## Relative orientation between the robot and the goal
        rel_theta = abs(theta - yaw)

        ## Angle wrapping [0,180) 
        if rel_theta > np.pi: 
            rel_theta = 2 * np.pi - rel_theta
        self.sweep = rel_theta
        ## Get goal distance
        goal_distance = distance(robot_pos,self.goal[:2])
        self.wtr_prev_distance = goal_distance

        ## Normalise all the attributes before stacking as an observation vector
        goal_distance_normalised = goal_distance/self.diagnoal_length
        yaw_normalised = yaw/2*np.pi
        rel_theta_normalised = rel_theta/np.pi


        ## WAM specific observations
                ## Get the generalised joint angles
        joint_angles = self.data.qpos[14:21]
        ## Get the generalised joint velocities
        joint_vels = self.data.qvel[12:19]

        joint_acc = self.data.qacc[12:19]
        ## End effector position
        ee_pos = self.data.xpos[15]

        normalised_joint_angles = np.zeros_like(joint_angles)
        normalised_joint_vels = np.zeros_like(joint_vels)
        normalised_joint_accs = np.zeros_like(joint_acc)

        for i in range(len(joint_angles)):
            angle = normalise(joint_angles[i]) # [0,360)
            normalised_joint_angles[i] = ((angle + np.pi) % (2 * np.pi) - np.pi)/np.pi  #[-1,1]

        # Normalise the joint vels between -1 and 1
        for i in range(len(joint_vels)):
            normalised_joint_vels[i] = joint_vels[i]/self.constraintMap[f"j{i}"].loc["v_limit"][0]
            normalised_joint_accs[i] = joint_acc[i]/self.constraintMap[f"j{i}"].loc["a_limit"][0]
        self.ee_prev_distance = distance(ee_pos,self.goal)


        observation = np.hstack((goal_distance_normalised, rel_theta_normalised, yaw_normalised, robot_vel,\
                                 normalised_joint_angles, normalised_joint_vels,normalised_joint_accs,
                                 *ee_pos, *self.goal))

        return observation
            
    def _get_reset_info(self):
        return self._get_info() 
    
    def close(self):
        super().close()

    @property
    def truncated(self):
        return self.failed
    @property
    def terminated(self):
        return self.reached

    def step(self,actions):
        '''
        Input:
        actions - np.array of shape 9 X 1, where the first 7 actions correspond to arm manipulation and last 2 actions describe wheelchair navigation
        '''
        ctrl = self.data.ctrl.copy()
        ## Pre simulation variable usage
        ### Arm navigation
        ctrl[:7] = torque_denomalise(actions[:7],self.max_torque)
        x_prev = self.data.xpos[15]
        
        ## Robot wheelchair navigation
        v,w = differential_vels_denormalize(*actions[-2:],self.max_v,self.max_w)
        ctrlvels = differential_drive(self.config,np.array([v,w]))
        vb_prev, wb_prev = self.data.qvel[6:9], self.data.qvel[9:12]
        ctrl[7:] = ctrlvels

        ## Perform simulation
        self.do_simulation(ctrl,self.frame_skip)

        ## Post simulation variable usage
        ## Arm navigation
        x_curr = self.data.xpos[15]
        velocity_curr = (x_curr - x_prev)/self.dt
        ee_finalVelocity_error = 1/(1 + np.linalg.norm(np.linalg.norm(velocity_curr) - self.vf))

        ## Robot wheechair navigation
        vb_curr,wb_curr = self.data.qvel[6:9], self.data.qvel[9:12]
        a_linear = np.linalg.norm((vb_curr - vb_prev) / self.dt)
        a_angular = np.linalg.norm((wb_curr - wb_prev) / self.dt)
        ## Check for linear acceleration limits
        acc_linear_error,acc_angular_error = 0,0
        if abs(a_linear) > self.a_lin_max: 
            acc_linear_error = (self.a_lin_max - abs(a_linear))/(2 * self.max_v/self.dt)
        if abs(a_angular) > self.a_ang_max: 
            acc_angular_error = (self.a_ang_max - abs(a_angular))/(2 * self.max_w/self.dt)
        ## Check for ctrlvels
        wtr_finalVelocity_error = abs(self.data.qvel[6:9] - self.vf) if self.reached else 0
        ## Check for the timely completion of the trajectory
        inTimeCost = float(self.config["reward_scales"].get("inTimeCost")) * (self.t_accepted - self.data.time)
        ## Check for episode termination
        self.reward.checkFlags()
        if self.reached:
            self.env_logger.info(f"Final Coordinate: {self.data.xpos[15]}")
            print()
            print()

        ## Rewards 
        navigation_reward   = self.reward.groundReward([acc_angular_error,acc_linear_error,- wtr_finalVelocity_error,inTimeCost])  
        manipulation_reward = self.reward.flyReward()
        total_reward = navigation_reward + manipulation_reward - self.reward.control_cost(actions)

        ## Final Simulation step parameters to return 
        terminated,truncated = self.terminated, self.truncated
        observation = self._get_obs()

        info = self._get_info()
        info['total_reward'] = total_reward
        info['linear_velocity'] = np.linalg.norm(self.data.qvel[6:9]) 
        info['wheel_1_velocity'] = self.data.qvel[31]
        info['wheel_2_velocity'] = self.data.qvel[32]
        info['timetoreach'] = self.data.time
        info['wheelchair_angular_velocity'] = np.linalg.norm(self.data.qvel[9:12])
        info['wheelchair_linear_acceleration'] = np.linalg.norm(self.data.qacc[6:9])
        info['angular_acceleration'] = np.linalg.norm(self.data.qacc[9:12])
        info['ground_positional_error'] = distance(self.goal[:2],self.data.qpos[7:9])
        for i in range(len(self.constraintMap.T)):
            info[f'joint_positions_{i}'] = self.data.qpos[14 + i]
            info[f'arm_angular_velocity_{i}'] = self.data.qvel[12 + i]
            info[f'arm_angular_acceleration_{i}'] = self.data.qacc[12 + i]
        info['arm_positional_error'] = distance(self.goal,self.data.xpos[15])
        info['is_success'] = terminated
        self.datatoplot = info

        if self.render_mode == "human":
            self.render()
        
        return observation,total_reward,terminated,truncated,info
        



if __name__ == "__main__":
    env = WTR3DReacherEnv()
    breakpoint()


        
