from typing import Dict
import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from pandas import DataFrame as DF
import logging
import yaml

def load_config(cfg_file):
    cfg_file = os.path.join("/home/splion360/Desktop/project/Sim2Real/mujoco/wam_model/scripts/cfg",cfg_file)
    with open(cfg_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


class WTRArmReacherEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ]
    }
    config = load_config("wam7_armreacher.yaml")
    DEFAULT_CAMERA_CONFIG = {
    "lookat": np.array(config["mujoco"].get("camera"))}
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

        self.config = config
        self.reset_noise_scale = float(config["mujoco"].get("reset_noise_scale"))
        self.error = float(config["mujoco"].get("error"))
        self.radius = radius
        self.failed,self.reached = False, False
        
        self.render_mode = render_mode
        self.goal = np.zeros(3)

        self.prev_distance = 0


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
        
        action_size = 7  # Action Space -> 7 Joint Torques, Observation Space -> (7 Joint angles + 7 Joint velocities + 1 distance to goal)
        self.observation_structure = {
            "skipped_qpos": 0,
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }
        torque_limits = config["dynamics"].get("joint_torque_limits")
        self.vf = config["dynamics"].get("goal_velocity")
        self.max_reachable_distance = config["dynamics"].get("max_reachable_distance") # Length of the upright Wam arm


        qpos = config["mujoco"].get("pos")
        ## Joint self.self.constraintMap
        self.constraintMap = DF()

        scale_factor = config["mujoco"].get("factorofsafety") # Operating with only scale_factor% of the max limits

        self.constraintMap["j0"] = {"j_limit":np.array(config["dynamics"]["wam_base"].get("j_limit")),\
                                    "v_limit":np.array(config["dynamics"]["wam_base"].get("v_limit"))*scale_factor,\
                                    "a_limit":np.array(config["dynamics"]["wam_base"].get("a_limit"))*scale_factor}
        
        self.constraintMap["j1"] = {"j_limit":np.array(config["dynamics"]["wam_shoulder_pitch"].get("j_limit")),\
                            "v_limit":np.array(config["dynamics"]["wam_shoulder_pitch"].get("v_limit"))*scale_factor,\
                            "a_limit":np.array(config["dynamics"]["wam_shoulder_pitch"].get("a_limit"))*scale_factor}
        
        self.constraintMap["j2"] = {"j_limit":np.array(config["dynamics"]["wam_shoulder_yaw"].get("j_limit")),\
                            "v_limit":np.array(config["dynamics"]["wam_shoulder_yaw"].get("v_limit"))*scale_factor,\
                            "a_limit":np.array(config["dynamics"]["wam_shoulder_yaw"].get("a_limit"))*scale_factor}
        
        self.constraintMap["j3"] = {"j_limit":np.array(config["dynamics"]["wam_upper_arm"].get("j_limit")),\
                            "v_limit":np.array(config["dynamics"]["wam_upper_arm"].get("v_limit"))*scale_factor,\
                            "a_limit":np.array(config["dynamics"]["wam_upper_arm"].get("a_limit"))*scale_factor}
        
        self.constraintMap["j4"] = {"j_limit":np.array(config["dynamics"]["wam_fore_arm"].get("j_limit")),\
                            "v_limit":np.array(config["dynamics"]["wam_fore_arm"].get("v_limit"))*scale_factor,\
                            "a_limit":np.array(config["dynamics"]["wam_fore_arm"].get("a_limit"))*scale_factor}
        
        self.constraintMap["j5"] = {"j_limit":np.array(config["dynamics"]["wam_wrist_yaw"].get("j_limit")),\
                            "v_limit":np.array(config["dynamics"]["wam_wrist_yaw"].get("v_limit"))*scale_factor,\
                            "a_limit":np.array(config["dynamics"]["wam_wrist_yaw"].get("a_limit"))*scale_factor}
        
        self.constraintMap["j6"] = {"j_limit":np.array(config["dynamics"]["wam_wrist_pitch"].get("j_limit")),\
                            "v_limit":np.array(config["dynamics"]["wam_wrist_pitch"].get("v_limit"))*scale_factor,\
                            "a_limit":np.array(config["dynamics"]["wam_wrist_pitch"].get("a_limit"))*scale_factor}

        self.init_pos = np.array(qpos.split(" "),dtype=np.float64)
        self.datatoplot = self._get_info()

        logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
        self.env_logger = logging.getLogger()

        self.observation_space = self._get_observation_space()
        self.action_space = Box(low = -torque_limits, high = torque_limits, shape=(action_size,), dtype=np.float32)


    ## Define Custom Observation Space instead of imposing constraint cost 
    def _get_observation_space(self):
        size = len(self.constraintMap.T)
        max_ee_pos,min_ee_pos = np.array([np.inf]*3),np.array([-np.inf]*3)
        max_target_pos,min_target_pos = np.array([np.inf]*3),np.array([-np.inf]*3)
        
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
        
        
        
        min_obs = np.hstack((min_joint_lim, min_vel_lim, min_acc_lim, *min_ee_pos, *min_target_pos))
        max_obs = np.hstack((max_joint_lim, max_vel_lim, max_acc_lim, *max_ee_pos, *max_target_pos))

        return Box(low = min_obs, high = max_obs, dtype = np.float64)

    ## Helper Functions
    def distance(self,pos1,pos2):
        return np.linalg.norm(pos1-pos2)

    def reset(self,seed = None, options = None):
        np.random.seed(seed)
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        # Setting the flags to false
        self.reached,self.failed = False, False

        # Resetting the data
        self.data.qpos[:] = self.init_pos.copy()
        self.data.qvel[:] = np.zeros(self.model.nv)
        self.data.ctrl[:] = np.zeros(self.model.nu)
        self.data.time = 0.0
        
        ## Target coordinates, Randomisation will be done based on the polar coordinates (r,theta)
        # r,theta = np.random.randint(low = 45 ,high = 75)/100 + np.random.uniform(low=noise_low, high=noise_high),\
        #           np.random.randint(low = 0, high = np.pi) + np.random.uniform(low=noise_low, high=noise_high)

        # tx,ty = self.data.qpos[7] + r*np.cos(theta), self.data.qpos[8] + r*np.sin(theta)
        # tz = self.data.qpos[9] + np.random.randint(low = 1 ,high = 75)/100 + np.random.uniform(low=noise_low, high=noise_high)

        # r,theta,phi = np.random.randint(low = 45 ,high = 75)/100 + np.random.uniform(low=noise_low, high=noise_high),\
        #               np.random.randint(low = 0, high = np.pi) + np.random.uniform(low=noise_low, high=noise_high),\
        #               np.random.randint(low = 0, high = 2*np.pi) + np.random.uniform(low=noise_low, high=noise_high)
        
        # tx = self.data.qpos[7] + r*np.cos(phi) * np.cos(theta)
        # ty = self.data.qpos[8] + r*np.sin(phi) * np.cos(theta)
        # tz = self.data.qpos[9] + r*np.sin(theta)

        x_offset = self.data.qpos[7] + self.config["mujoco"].get("baseoffset")
        y_offset = self.data.qpos[8] + self.config["mujoco"].get("baseoffset")
        z_offset = self.data.qpos[9] + self.config["mujoco"].get("baseoffset")

        ## Randomly sample values of x,y,z from the cube of size 0.3
        mid = self.config["mujoco"].get("cubesize")//2
        tx = Box(low = x_offset - mid, high = x_offset + mid , shape=(1,), dtype = np.float64).sample()
        ty = Box(low = y_offset - mid, high = y_offset + mid , shape=(1,), dtype = np.float64).sample()        
        tz = Box(low = z_offset - mid, high = z_offset + mid , shape=(1,), dtype = np.float64).sample()

        self.target_pos = np.array((*tx,*ty,*tz))
        self.env_logger.info(f"Target coordinates: {self.target_pos}")

        self.model.site('ball_coord').pos = self.target_pos
        self.set_state(self.data.qpos,self.data.qvel)        

        self.prev_distance = self.distance(self.target_pos,self.data.xpos[15].copy())
        observation = self._get_obs()
        print(observation)
        info = self._get_reset_info()
        return observation, info
    
    def normalise(self,theta):
        if theta < 0: return theta + 2 * np.pi
        elif theta >= 2 * np.pi: return theta - 2 * np.pi
        else: return theta

    def _get_obs(self):
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
            angle = self.normalise(joint_angles[i]) # [0,360)
            normalised_joint_angles[i] = ((angle + np.pi) % (2 * np.pi) - np.pi)/np.pi  #[-1,1]

        # Normalise the joint vels between -1 and 1
        for i in range(len(joint_vels)):
            normalised_joint_vels[i] = joint_vels[i]/self.constraintMap[f"j{i}"].loc["v_limit"][0]
            normalised_joint_accs[i] = joint_acc[i]/self.constraintMap[f"j{i}"].loc["a_limit"][0]
        # normalised_distance =  self.distance(ee_pos,self.target_pos)/self.max_reachable_distance
        self.prev_distance = self.distance(ee_pos,self.target_pos)
        observation = np.hstack((normalised_joint_angles, normalised_joint_vels,normalised_joint_accs,
                                 *ee_pos, *self.target_pos))
    
        return observation
    

    def control_cost(self, action):
        control_cost = float(self.config["reward_scales"].get("control_cost")) * np.sum(np.square(action))
        return control_cost
    
    @property
    def reward(self):
        reward,payoff = 0,0
        ## Distance Rate
        curr_target_distance = self.distance(self.data.xpos[15],self.target_pos)
        distance_rate = self.prev_distance - curr_target_distance 

        if self.failed: 
            payoff = min(self.config["reward_scales"].get("terminal_payoff"))
        if self.reached: 
            payoff = max(self.config["reward_scales"].get("terminal_payoff"))
        reward = self.config["reward_scales"].get("distance_incentive") * distance_rate + payoff #+ penalty 

        return reward
    
    def checkFlags(self):
        ## Check for target reach
        ee_pos = self.data.xpos[15].copy()
        curr_goal_distance = self.distance(ee_pos,self.target_pos)
        if curr_goal_distance <= self.radius: 
            self.reached = True
    @property
    def truncated(self):
        return self.failed
    @property
    def terminated(self):
        return self.reached

    def step(self,action):
        ctrl = self.data.ctrl.copy()
        ctrl[:7] = action
        x_prev = self.data.xpos[15]
        
        self.do_simulation(ctrl,self.frame_skip)
        x_curr = self.data.xpos[15]
        velocity_curr = (x_curr - x_prev)/self.dt
        finalVelocity = 1/(1 + np.linalg.norm(np.linalg.norm(velocity_curr) - self.vf))
        self.checkFlags()
        if self.reached:
            self.env_logger.info(f"Final Coordinate: {self.data.xpos[15]}")
            print()
            print()
        ctrl_cost = self.control_cost(action)
        total_reward = self.reward - ctrl_cost #+ 10 * acc_error #+ finalVelocity
        terminated,truncated = self.terminated, self.truncated
        observation = self._get_obs()


        info = self._get_info()
        info['total_reward'] = total_reward
        for i in range(len(self.constraintMap.T)):
            info[f'joint_positions_{i}'] = self.data.qpos[14 + i]
            info[f'angular_velocity_{i}'] = self.data.qvel[12 + i]
            info[f'angular_acceleration_{i}'] = self.data.qacc[12 + i]

        info['positional_error'] = self.distance(self.target_pos,self.data.xpos[15])
        # info['velocity_differential'] = np.linalg.norm(velocity_curr) - self.vf
        info['is_success'] = terminated
        self.datatoplot = info
        if self.render_mode == "human":
            self.render()
        
        return observation, total_reward,terminated,truncated,info
    
    def _get_reset_info(self):
        return self._get_info() 
    def _get_info(self):
        info = {}
        info['total_reward'] = float('-inf')
        for i in range(len(self.constraintMap.T)):
            info[f'joint_positions_{i}'] = float('-inf')
            info[f'angular_velocity_{i}'] = float('-inf')
            info[f'angular_acceleration_{i}'] = float('-inf')
        info['positional_error'] = float('-inf')
        # info['velocity_differential'] = float('-inf')
        info['is_success'] = False
        return info

    def close(self):
        super().close()


if __name__ == "__main__":
    env = WTRArmReacherEnv()
    env.reset()
    breakpoint()
