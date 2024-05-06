from typing import Dict
import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from pandas import DataFrame as DF
import logging

DEFAULT_CAMERA_CONFIG = {
    "lookat": np.array([-27.212, -0.006, 16.410]),
}

class WTRArmReacherEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ]
    }
    full_path = "/home/splion360/Desktop/project/Sim2Real/mujoco/wam_model"
    def __init__(
        self,
        xml_file: str = os.path.join(full_path,"robots/wam7_armreacher.xml"),
        frame_skip: int = 4,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 5e-3,
        error:float = 1e-5,
        radius:float = 0.05,
        render_mode: str = None,
        **kwargs):

        utils.EzPickle.__init__(self,
        xml_file,
        frame_skip,
        default_camera_config,
        reset_noise_scale,
        error,
        radius,
        render_mode,
        **kwargs
        )

        self.reset_noise_scale = reset_noise_scale
        self.error = error
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
        
        obs_size,action_size = 15,7 # Action Space -> 7 Joint Torques, Observation Space -> (7 Joint angles + 7 Joint velocities + 1 distance to goal)
        self.observation_structure = {
            "skipped_qpos": 0,
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }
        torque_limits = 2
        self.vf = 0
        self.max_reachable_distance = 0.850 # Length of the upright Wam arm

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self.action_space = Box(low = -torque_limits, high = torque_limits, shape=(action_size,), dtype=np.float32)
        qpos='12 2.5773e-20 -0.400367 1 1.26736e-18 0 0 -0.000195325 0.0474326 -0.0352322 0.70946 0.00587794 -0.00591294 0.704696 -0.00203498 1.22339 -2.88888 3.04008 -5.01718 -0.300392 0.622417 0.00593441 -0.868072 0.0155506 -0.457605 0.191851 0.00593441 -0.895244 -0.0746412 -0.424762 0.112001 0.00593441 0.687093 0.453329 0.500147 0.268791 -0.191714 0.168391'
        ## Joint self.self.constraintMap
        self.constraintMap = DF()
        '''
        max_velocities:    [3.9, 6.3, 10.0, 10.0, 24.0, 19.0, 27.0]
        max_accelerations: [6.82, 11.5, 17.0, 21.5, 84.0, 110.0, 100.0]

        min_positions: [-2.6, -1.98, -2.8, -0.9, -4.65, -1.57, -2.95]
        max_positions: [ 2.6,  1.98,  2.8,  3.1,  1.35,  1.57,  2.95]

        '''
        scale_factor = 0.2 # Operating with only scale_factor% of the max limits

        self.constraintMap["j0"] = {"j_limit":np.array((-2.6,2.6)),"v_limit":np.array((-3.9,3.9))*scale_factor,"a_limit":np.array((-6.82,6.82))*scale_factor}
        self.constraintMap["j1"] = {"j_limit":np.array((-1.98,1.98)),"v_limit":np.array((-6.3,6.3))*scale_factor,"a_limit":np.array((-11.5,11.5))*scale_factor}
        self.constraintMap["j2"] = {"j_limit":np.array((-2.8,2.8)),"v_limit":np.array((-10,10))*scale_factor,"a_limit":np.array((-17,17))*scale_factor}
        self.constraintMap["j3"] = {"j_limit":np.array((-0.9,3.1)),"v_limit":np.array((-10,10))*scale_factor,"a_limit":np.array((-21.5,21.5))*scale_factor}
        self.constraintMap["j4"] = {"j_limit":np.array((-4.65,1.35)),"v_limit":np.array((-24,24))*scale_factor,"a_limit":np.array((-84,84))*scale_factor}
        self.constraintMap["j5"] = {"j_limit":np.array((-1.57,1.57)),"v_limit":np.array((-19,19))*scale_factor,"a_limit":np.array((-110,110))*scale_factor}
        self.constraintMap["j6"] = {"j_limit":np.array((-2.95,2.95)),"v_limit":np.array((-27,27))*scale_factor,"a_limit":np.array((-100,100))*scale_factor}

        self.init_pos = np.array(qpos.split(" "),dtype=np.float64)
        self.datatoplot = self._get_info()

        logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
        self.env_logger = logging.getLogger()
        self.env_logger.info(f"Env reset with target position: {self.init_pos}")
        self.env_logger.info("============================================================")

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
        r,theta = np.random.randint(low = 45 ,high = 75)/100 + np.random.uniform(low=noise_low, high=noise_high),\
                  np.random.randint(low = 0, high = np.pi) + np.random.uniform(low=noise_low, high=noise_high)

        tx,ty = r*np.cos(theta), r*np.sin(theta)
        tz = np.random.randint(low = self.reset_noise_scale ,high = 75)/100 + np.random.uniform(low=noise_low, high=noise_high)
        
        self.target_pos = np.array((tx,ty,tz))
        self.env_logger.info(f"Target coordinates: {self.target_pos}")
        self.set_state(self.data.qpos,self.data.qvel)        

        self.prev_distance = self.distance(self.target_pos,self.data.xpos[15].copy())
        observation = self._get_obs()
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
        ## End effector position
        ee_pos = self.data.xpos[15]

        normalised_joint_angles = np.zeros_like(joint_angles)
        normalised_joint_vels = np.zeros_like(joint_vels)

        for i in range(len(joint_angles)):
            normalised_joint_angles[i] = self.normalise(joint_angles[i])/360.0 # [0,1]
        
        # Normalise the joint vels between 0 and 1
        for i in range(len(joint_vels)):
            normalised_joint_vels[i] = joint_vels[i]/self.constraintMap[f"j{i}"].loc["v_limit"][0]
        normalised_distance =  self.distance(ee_pos,self.target_pos)/self.max_reachable_distance
        self.prev_distance = self.distance(ee_pos,self.target_pos)
        observation = np.hstack((normalised_joint_angles, normalised_joint_vels,normalised_distance))
        return observation
    

    def control_cost(self, action):
        control_cost = 1e-3 * np.sum(np.square(action))
        return control_cost
    
    @property
    def reward(self):
        reward,payoff = 0,0
        penalty = self.constraintLimits ## Joint limits penalty
        ## Distance Rate
        curr_target_distance = self.distance(self.data.xpos[15],self.target_pos)
        distance_rate = self.prev_distance - curr_target_distance 

        if self.failed: 
            payoff = -100
        if self.reached: 
            payoff = 120
        reward = distance_rate - penalty + payoff 
        return reward

    @property 
    def constraintLimits(self):
        penalty = 0
        for i in range(len(self.constraintMap.T)):
            ## Check for joint limits
            if abs(self.data.qpos[14 + i]) > self.constraintMap[f"j{i}"].loc["j_limit"].max() or \
               abs(self.data.qpos[14 + i]) < self.constraintMap[f"j{i}"].loc["j_limit"].min():
                jmax = self.constraintMap[f"j{i}"].loc["j_limit"].max()
                jmin = self.constraintMap[f"j{i}"].loc["j_limit"].min()
                jp = min(abs(self.data.qpos[14 + i])-jmax, abs(self.data.qpos[14 + i])-jmin)
                penalty += jp

            if abs(self.data.qvel[12 + i]) > self.constraintMap[f"j{i}"].loc["v_limit"].max(): 
                vp = (self.constraintMap[f"j{i}"].loc["v_limit"].max() - abs(self.data.qvel[12 + i]))
                penalty += vp

            if abs(self.data.qacc[12 + i]) > self.constraintMap[f"j{i}"].loc["a_limit"].max(): 
                ap = (self.constraintMap[f"j{i}"].loc["a_limit"].max() - abs(self.data.qacc[12 + i]))
                penalty += ap

            return penalty / len(self.constraintMap.T)
    
    def checkFlags(self):
        ## Check for target reach
        ee_pos = self.data.xpos[15].copy()
        curr_goal_distance = self.distance(ee_pos,self.target_pos)
        if curr_goal_distance <= self.radius: 
            self.reached = True

        # Check for target timely reach and out of bounds error
        conditions = [self.data.time > 2000 * self.dt]
        if any(conditions):
            self.failed = True
    @property
    def truncated(self):
        return self.failed
    @property
    def terminated(self):
        return self.reached

    def step(self,action):
        ctrl = self.data.ctrl.copy()
        ctrl[:7] = action
        ctrl[1] = 2.8
        x_prev = self.data.xpos[15]
        
        self.do_simulation(ctrl,self.frame_skip)
        x_curr = self.data.xpos[15]
        velocity_curr = (x_curr - x_prev)/self.dt
        finalVelocity = 1/(1 + np.linalg.norm(np.linalg.norm(velocity_curr) - self.vf))
        self.checkFlags()
        ctrl_cost = self.control_cost(action)
        total_reward = self.reward - ctrl_cost + finalVelocity
        terminated,truncated = self.terminated, self.truncated
        observation = self._get_obs()


        info = self._get_info()
        info['total_reward'] = total_reward
        for i in range(len(self.constraintMap.T)):
            info[f'angular_velocity_{i}'] = self.data.qvel[12 + i]
            info[f'angular_acceleration_{i}'] = self.data.qacc[12 + i]

        info['positional_error'] = self.distance(self.target_pos,self.data.xpos[15])
        info['velocity_differential'] = np.linalg.norm(velocity_curr) - self.vf
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
            info[f'angular_velocity_{i}'] = float('-inf')
            info[f'angular_acceleration_{i}'] = float('-inf')
        info['positional_error'] = float('-inf')
        info['velocity_differential'] = float('-inf')
        info['is_success'] = False
        return info

    def close(self):
        super().close()


if __name__ == "__main__":
    env = WTRArmReacherEnv()
    env.reset()
    breakpoint()
