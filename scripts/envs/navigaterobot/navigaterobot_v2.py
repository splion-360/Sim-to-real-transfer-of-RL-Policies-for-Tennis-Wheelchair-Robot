
from typing import Dict, Tuple
import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box,Tuple
import yaml



def load_config(cfg_file):
    cfg_file = os.path.join('../../cfg',cfg_file)
    with open(cfg_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

class WTRBlockReacherEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ]
    }
    full_path = "/home/splion360/Desktop/project/Sim2Real/mujoco/wam_model"
    config = load_config("wam7_blockreacher.yaml")
    DEFAULT_CAMERA_CONFIG = {
    "lookat": np.array(config["mujoco"].get("camera"))}

    def __init__(
        self,
        xml_file: str = os.path.join(full_path,"robots/wam7_blockreacher.xml"),
        frame_skip: int = config["mujoco"].get("frame_skip"),
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = config["mujoco"].get("reset_noise_scale"),
        error:float = config["mujoco"].get("error"),
        radius:float = config["mujoco"].get("radius"),
        maxv:float = config["dynamics"].get("max_linear_velocity"), # from the drive_controller.yaml file
        maxw:float = config["dynamics"].get("max_angular_velocity"),
        camera_name="view",
        config:Dict[str,float] = config,
        render_mode: str = None,
        **kwargs):

        utils.EzPickle.__init__(self,
        xml_file,
        frame_skip,
        default_camera_config,
        reset_noise_scale,
        error,
        radius,
        maxv,
        maxw,
        render_mode,
        camera_name,
        config,
        **kwargs
        )


        self.config = config
        self.reset_noise_scale = float(config["mujoco"].get("reset_noise_scale"))
     
        self.max_v,self.max_w = config["dynamics"].get("max_linear_velocity"), config["dynamics"].get("max_angular_velocity")

        self.error = float(config["mujoco"].get("error"))
        self.radius = config["mujoco"].get("radius")
        self.failed,self.reached = False, False
        
        self.render_mode = render_mode
        self.goal = np.zeros(2)
        self.t_accepted = 0
        self.prev_distance = 0
        self.vf = config["dynamics"].get("goal_velocity")
        self.a_lin_max = config["dynamics"].get("max_linear_acceleration")
        self.a_ang_max = config["dynamics"].get("max_angular_acceleration")

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            render_mode=render_mode,
            **kwargs
            )
        self.metadata = {"render_modes":["human","rgb_array","depth_array"],"render_fps":int(np.round(1.0/self.dt))}
        
        action_size = 2

        self.observation_structure = {
            "skipped_qpos": 0,
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }
        self.observation_space = self._get_observation_space()
        ## Set linear and angular velocity limits based on the output from the differential controller
        ## Linear Velocity limit is set as 4 m/s and Angular velocity limit is set as 8 rad/s (based on the Athletic paper following 1:2 ratio)
        ## Corresponding angular velocity limits for the wheels are -10 rad/s and 10 rad/s
        ## Corresponding linear velocity limits for the wheels are -11 m/s and 11 m/s
        ## Acceleration limit is still not set. |v| <= amax * dt and |w| <= alpha(tan) * dt. Centrifugal force limitation: v*w <= a(rad) * dt
        self.wheel_torque_limits = self.config["dynamics"].get("wheel_torque_limit") # Operating at 60% of max wheel torque limits

        self.action_space = Box(low = -1, high = 1, shape=(action_size,), dtype=np.float64)
        self.diagnoal_length = self.config["mujoco"].get("env_length") * 2 * np.sqrt(2)
        self.init_pos = self.data.qpos.copy()
        self.datatoplot = self._get_info()

    
        
    def distance(self,pos1,pos2):
        return np.linalg.norm(pos1-pos2)

    def _get_observation_space(self):
        min_gd = np.array([-np.inf])
        max_gd = np.array([np.inf])
        min_the = np.array([-np.inf])
        max_the = np.array([np.inf])
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

        min_obs = np.hstack((min_gd,min_the,min_rel,min_yaw,min_vel))
        max_obs = np.hstack((max_gd,max_the, max_rel, max_yaw, max_vel))

        return Box(low = min_obs, high = max_obs, dtype = np.float64)

    def differential_drive(self,action):
        axle_length = self.config["differential_drive"].get("axle_length") # Original length x scaling factor (0.92575*1.3)
        diameter = self.config["differential_drive"].get("wheel_diameter") # Original diameter x scaling factor (0.62*1.2)
        v_des, w_des = action
        # Linear velocities
        vr,vl = (2*v_des + w_des*axle_length)/2,(2*v_des - w_des*axle_length)/2 
        # Angular velocities
        w1,w2 = 2*vr/diameter, 2*vl/diameter
        return np.array([w1,w2])

    def denormalize(self,t1,t2):
        normalised_v = (0.5 * (t1+1) * (2* self.wheel_torque_limits)) - self.wheel_torque_limits
        normalised_w = (0.5 * (t2+1) * (2* self.wheel_torque_limits)) - self.wheel_torque_limits
        return np.array([normalised_v,normalised_w])
    
    def _get_obs(self):
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
        goal_distance = self.distance(robot_pos,self.goal)
        self.prev_distance = goal_distance

        ## Normalise all the attributes before stacking as an observation vector
        goal_distance_normalised = goal_distance/self.diagnoal_length
        yaw_normalised = yaw/2*np.pi
        theta_normalised = theta/2*np.pi
        rel_theta_normalised = rel_theta/np.pi

        observation = np.hstack((goal_distance_normalised, theta_normalised, rel_theta_normalised, yaw_normalised, robot_vel))
        return observation
    
    
    def control_cost(self, action):
        control_cost = float(self.config["reward_scales"].get("control_cost")) * np.sum(np.square(action))
        return control_cost
    
    @property
    def reward(self):
        robot_pos = self.data.qpos[7:9].copy()
        curr_goal_distance = self.distance(robot_pos,self.goal)
        distance_rate = self.prev_distance - curr_goal_distance
        reward = self.config["reward_scales"].get("distance_incentive") * distance_rate # Distance incentive
        payoff = 0
        if self.failed: 
            payoff = min(self.config["reward_scales"].get("terminal_payoff")) # Truncation if the robot goes out of bounds. 
        if self.reached:
            payoff = max(self.config["reward_scales"].get("terminal_payoff"))
        return reward + payoff 

    def checkFlags(self):
        ## Check for target reach
        robot_pos = self.data.qpos[7:9].copy()
        curr_goal_distance = self.distance(robot_pos,self.goal)
        if curr_goal_distance <= self.radius: 
            self.reached = True

        ## Check for target out of bounds error
        x,y = robot_pos
        conditions = [(x < -8 or x > 8),
                       (y < -8 or y > 8),]
        if any(conditions):
            self.failed = True

    def _get_info(self):
        info = {}
        info['total_reward'] = float('-inf')
        info['linear_velocity_error'] = float('-inf')
        info['angular_velocity_error'] = float('-inf')
        info['linear_acceleration'] = float('-inf')
        info['angular_acceleration'] = float('-inf')
        info['timetoreach'] = float('-inf')
        info['wheel_1_velocity'] = float('-inf')
        info['wheel_2_velocity'] = float('-inf')
        info['positional_error'] = float('-inf')
        info['is_success'] = False
        return info

    def reset(self,seed=None,options=None):
        np.random.seed(seed)
        # Random initialization of the ball coordinates
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        # Setting the flags to false
        self.reached,self.failed = False, False

        # Resetting the data
        self.data.qpos[:] = self.init_pos.copy()
        self.data.qvel[:] = np.zeros(self.model.nv)
        self.data.time = 0.0

        ## Ball coordinates
        cx,cy = np.random.randint(low=-self.config["mujoco"].get("coordinate_limits"),high=self.config["mujoco"].get("coordinate_limits")) + np.random.uniform(low=noise_low, high=noise_high),\
                np.random.randint(low=-self.config["mujoco"].get("coordinate_limits"),high=self.config["mujoco"].get("coordinate_limits")) + np.random.uniform(low=noise_low, high=noise_high)
        self.data.qpos[0] = cx
        self.data.qpos[1] = cy
        self.set_state(self.data.qpos,self.data.qvel)
        ## Setting the original goal position
        self.goal = self.data.qpos[:2].copy()
        self.prev_distance = self.distance(self.goal,self.data.qpos[7:9].copy())
        
        ## Estimating the minimum time required to reach the goal from the starting position and adding grace time period of 2 seconds
        observation = self._get_obs()
        info = self._get_reset_info()
        self.t_accepted = (self.distance(self.goal,self.data.qpos[7:9].copy()) / self.max_v) + (self.sweep/self.max_w) + \
                           self.config["reward_scales"].get("grace_time_period")
        return observation, info

    @property
    def truncated(self):
        return self.failed
    @property
    def terminated(self):
        return self.reached

    def step(self,action):
        ## Action is an array of size 2 containing the angular velocities of the wheels
        vb_prev, wb_prev = self.data.qvel[6:9], self.data.qvel[9:12]
        ctrl = self.data.ctrl.copy()
        ctrl[-2:] = self.denormalize(*action)
        self.do_simulation(ctrl,self.frame_skip)
        vb_curr,wb_curr = self.data.qvel[6:9], self.data.qvel[9:12]
        ## Check for accleration limits
        a_linear =  np.linalg.norm(self.data.qacc[6:9])                            #np.linalg.norm((vb_curr - vb_prev) / self.dt)
        a_angular = np.linalg.norm(self.data.qacc[9:12])                            #np.linalg.norm((wb_curr - wb_prev) / self.dt)
        ## Check for linear acceleration limits
        acc_linear_error,acc_angular_error = 0,0
        # print(self.data.qacc[6:9])
        # print((vb_curr - vb_prev) / self.dt)

        if abs(a_linear) > self.a_lin_max: 
            acc_linear_error = (self.a_lin_max - abs(a_linear))/(2 * self.max_v/self.dt)
        if abs(a_angular) > self.a_ang_max: 
            acc_angular_error = (self.a_ang_max - abs(a_angular))/(2 * self.max_w/self.dt)
      
        ## Check for vehicle final velocity when reaching the target
        finalVelocity = 1/(1 + np.linalg.norm(np.linalg.norm(self.data.qvel[6:9]) - self.vf)) if self.reached else 0

        ## Check for the timely completion of the trajectory
        if self.reached: 
            inTimeCost = float(self.config["reward_scales"].get("inTimeCost")) * (self.t_accepted - self.data.time)
        else: inTimeCost = 0

        ctrl_cost = self.control_cost(action)  
        self.checkFlags()
        reward = self.reward - ctrl_cost + acc_angular_error + acc_linear_error + finalVelocity + inTimeCost
        terminated,truncated = self.terminated, self.truncated
        observation = self._get_obs()

        info = self._get_info()
        info['total_reward'] = reward
        info['linear_velocity'] = np.linalg.norm(self.data.qvel[6:9]) 
        info['wheel_1_velocity'] = self.data.qvel[31]
        info['wheel_2_velocity'] = self.data.qvel[32]
        info['timetoreach'] = self.data.time
        info['angular_velocity'] = np.linalg.norm(self.data.qvel[9:12])
        info['linear_acceleration'] = np.linalg.norm(self.data.qacc[6:9])
        info['angular_acceleration'] = np.linalg.norm(self.data.qacc[9:12])
        info['positional_error'] = self.distance(self.goal,self.data.qpos[7:9])
        info['is_success'] = terminated
        self.datatoplot = info

        if self.render_mode == "human":
            self.render()
        
        return observation, reward,terminated,truncated,info
    
    def _get_reset_info(self):
        return self._get_info() 
    
    def close(self):
        super().close()


if __name__ == "__main__":
    env = WTRBlockReacherEnv()
    breakpoint()
