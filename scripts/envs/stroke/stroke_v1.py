from typing import Dict, Tuple
import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box,Tuple
import yaml
import mujoco
from utils import load_config, distance, control_cost, getTime
import logging
from pandas import DataFrame as DF
from solver import Solver


class WTRStrokeEnv(MujocoEnv, utils.EzPickle):
    
    metadata = {
    "render_modes": [
        "human",
        "rgb_array",
        "depth_array"
    ]
    }
    config = load_config("wam7_stroke.yaml")
    DEFAULT_CAMERA_CONFIG = {
    "lookat": np.array(config["mujoco"].get("camera"))}
    full_path = "/home/skumar671/Capstone/wam_model/" 

    def __init__(
                self,
                xml_file: str = os.path.join(full_path,"robots_v1/wam7_test_v10.xml"),
                frame_skip: int = config["mujoco"].get("frame_skip"),
                default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
                reset_noise_scale: float = config["mujoco"].get("reset_noise_scale"),
                error:float = config["mujoco"].get("error"),
                render_mode: str = None,
                config:Dict[str,float] = config,
                **kwargs):

        utils.EzPickle.__init__(self,
        xml_file,
        frame_skip,
        default_camera_config,
        reset_noise_scale,
        error,
        render_mode,
        config,
        **kwargs
        )

        self.config = config
        self.reset_noise_scale = float(config["mujoco"].get("reset_noise_scale"))
        self.error = float(config["mujoco"].get("error"))
        self.render_mode = render_mode
        self.goal = np.zeros(2)

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
        #self.datatoplot = self._get_info()

        logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
        self.env_logger = logging.getLogger()
        self.observation_space = self._get_observation_space()
        
        self.limits = [self.config["mujoco"]["coordinate_limits"].get("x_limit"), self.config["mujoco"]["coordinate_limits"].get("y_limit")] 
        self.wall_limits = [self.config["mujoco"]["wall_limits"].get("x_limit"), \
                            self.config["mujoco"]["wall_limits"].get("y_limit"), \
                            self.config["mujoco"]["wall_limits"].get("z_limit")]
        self.COR = self.config["mujoco"].get("COR")
        self.g = self.config["mujoco"].get("g")
        self.forward_vel =  self.config["mujoco"].get("forward_vel")
        self.action_space = Box(low = -torque_limits, high = torque_limits, shape=(action_size,), dtype=np.float32)
        self.solver = Solver(np.zeros(6),np.zeros(3),kd = 0.04,km = 0)
        self.bpos, self.bvel = None, None
        self.prev_action = None
        self.hit, self.failed = False, False
        self.prev_distance = None
        self.curr_reward = float('inf')
        self.elapsed_steps = 0

    def generateGoalCoordinates(self):
        low, high = - self.reset_noise_scale, self.reset_noise_scale
        x_lim, y_lim = self.limits
        gx,gy = np.random.randint(low = min(x_lim), high = max(x_lim)) + np.random.uniform(low = low, high = high),\
                np.random.randint(low = min(y_lim), high = max(y_lim)) + np.random.uniform(low = low, high = high)
        
        return np.array([gx, gy])
    
    def generateBallCoordinates(self):
        x_lim, y_lim, z_lim = self.wall_limits     
        bx = Box(low = min(x_lim), high = max(x_lim) , shape=(1,), dtype = np.float64).sample()[0]
        by = Box(low = min(y_lim), high = max(y_lim) , shape=(1,), dtype = np.float64).sample()[0]      
        bz = Box(low = min(z_lim), high = max(z_lim) , shape=(1,), dtype = np.float64).sample()[0]
        return np.array([bx, by, bz])
    
    def getLaunchVel(self, z1, R):
        vz = np.sqrt(2 * self.g * z1)
        r2 = self.COR * vz / self.g
        r1 = np.sqrt((2 * z1)/self.g)
        return R / (r1 + r2)
    
    def getKinInt(self,pos,vel):
        self.solver.reset(vel, pos)
        t = np.arange(0, self.config["mujoco"].get("Tmax"), self.dt)
        velocity, position = np.zeros((t.shape[0],6)), np.zeros((t.shape[0],3))
        velocity[0], position[0] = vel, pos
        penetration_distance = self.config["mujoco"].get("penetration_depth")
        for i in range(1,t.shape[0]):
            p1,v1 = self.solver.solve(self.dt)
            velocity[i],position[i] = v1, p1
            if p1[-1] < penetration_distance:
                v1[1], v1[0]  = abs(v1[1]), abs(v1[0])
                v1[2]  = self.COR * abs(v1[2])
            self.solver.reset(v1, p1)
        target_pos, target_vel = self.solver.maxHeight(position,velocity)
        return target_pos, target_vel
    
    def get_racket_pos(self):
        return (self.data.xpos[15], self.data.qpos[17])
        # return (self.data.xpos[15] + np.array([0, self.config["mujoco"].get("racket_length") * np.sin(self.data.qpos[17]), \
        #                                           self.config["mujoco"].get("racket_length") * np.cos(self.data.qpos[17])]), self.data.qpos[17])
    def get_racket_vel(self):
        return self.data.qvel[15]
    def get_ball_pos(self):
        return self.data.xpos[1]
    def get_ball_vel(self):
        return self.data.qvel[:3], self.data.qvel[3:6]

    def _get_obs(self):
        # Get the ball dynamics at every point in time
        ball_pos, (ball_vel,_) = self.get_ball_pos(), self.get_ball_vel()
        # Get the robot dynamics at every point in time
        rob_pos = np.hstack((self.data.qpos[14:19], self.data.qpos[36:]))
        rob_vel = np.hstack((self.data.qvel[12:17], self.data.qvel[31:]))
        observation = np.hstack((ball_pos, ball_vel, rob_pos, rob_vel))
        return observation
    
    ## Define Custom Observation Space instead of imposing constraint cost 
    def _get_observation_space(self):
        min_ball_pos, max_ball_pos = np.array([ -np.inf] * 3), np.array([np.inf] * 3)
        min_ball_vel, max_ball_vel = np.array([ -np.inf] * 3), np.array([np.inf] * 3)

        min_rob_pos, max_rob_pos = np.array([ -np.inf] * 7), np.array([np.inf] * 7)
        min_rob_vel, max_rob_vel = np.array([ -np.inf] * 7), np.array([np.inf] * 7)

   
        min_obs = np.hstack((min_ball_pos, min_ball_vel, min_rob_pos, min_rob_vel))
        max_obs = np.hstack((max_ball_pos, max_ball_vel, max_rob_pos, max_rob_vel))
        return Box(low = min_obs, high = max_obs, dtype = np.float64)
    
    
    def reset(self,seed = None, options = None):
        self.curr_reward = float('inf')
        self.elapsed_steps = 0
        self.failed, self.hit = False, False
        np.random.seed(seed)
        # Resetting the data
        self.data.qpos[:] = self.init_pos.copy()
        self.data.qvel[:] = np.zeros(self.model.nv)
        self.data.ctrl[:] = np.zeros(self.model.nu)
        self.data.time = 0.0
        
        ball_pos = self.generateBallCoordinates()
        ball_vel = np.array([-self.config["mujoco"].get("forward_vel"), 0, 0, 0, 0, 0])
        self.target_pos, self.target_vel = self.getKinInt(ball_pos, ball_vel)
        self.goal = self.generateGoalCoordinates()

        self.data.qpos[:3] = ball_pos
        self.data.qvel[:6] = ball_vel

        self.env_logger.info(f"Ball spawn coordinates: {ball_pos}")
        self.set_state(self.data.qpos,self.data.qvel)        
        observation = self._get_obs()
        info = self._get_reset_info()
        self.prev_distance = distance(self.goal, ball_pos[:2])
        return observation, info
    

    def checkFlags(self):
        ball_pos, (racket_pos,_) = self.get_ball_pos(), self.get_racket_pos()
        if distance(racket_pos, ball_pos) < self.config["mujoco"].get("threshold"): self.hit = True
        
        conditions = [(ball_pos[0] < -20 or ball_pos[0] > 20),
                      (ball_pos[1] < -20 or ball_pos[1] > 20),
                      self.elapsed_steps > self.config["mujoco"].get("max_steps")
                     ]
        if any(conditions): self.failed = True

    @property
    def terminated(self):
        return self.hit
    
    @property
    def truncated(self):
        return self.failed

    @property
    def reward(self):
        ball_pos, (racket_pos,_) = self.get_ball_pos(), self.get_racket_pos()
        self.curr_reward = min(distance(ball_pos, racket_pos), self.curr_reward)
        # print('ball pos', ball_pos)
        # print('racket pos', racket_pos)
        # print('distance', distance(ball_pos, racket_pos)
        if self.truncated or self.terminated: 
            return -self.curr_reward
        else:return 0
        
    def step(self,action):
        ball_pos, (racket_pos,_) = self.get_ball_pos(), self.get_racket_pos()
        ctrl = self.data.ctrl.copy()
        ctrl[:7] = action      
        self.do_simulation(ctrl,self.frame_skip)
        # ctrl_cost = min(control_cost(self.config["reward_scales"].get("control_cost"), action, self.prev_action, self.dt), 0.2)
        self.checkFlags()
        total_reward = self.reward #- ctrl_cost 
        terminated,truncated = self.terminated, self.truncated
        observation = self._get_obs()
        info = self._get_info()
        info['total_reward'] = total_reward
        for i in range(5):
            info[f'joint_positions_{i}'] = self.data.qpos[14 + i]
            info[f'angular_velocity_{i}'] = self.data.qvel[12 + i]
        info['bat_ball_dist'] = self.curr_reward
        # info['positional_error'] = distance(self.goal,ball_pos[:2])
        info['is_success'] = terminated
        self.datatoplot = info
        self.elapsed_steps += 1
        if self.render_mode == "human":
            self.render()
        return observation, total_reward,terminated,truncated,info

    def _get_reset_info(self):
        return self._get_info()     
    def _get_info(self):
        info = {}
        info['total_reward'] = float('-inf')
        for i in range(5):
            info[f'joint_positions_{i}'] = float('-inf')
            info[f'angular_velocity_{i}'] = float('-inf')
            info[f'angular_acceleration_{i}'] = float('-inf')
        info['positional_error'] = float('-inf')
        info['bat_ball_dist'] = float('-inf')
        info['is_success'] = False
        return info
    
    def close(self):
        super().close()    


if __name__ == "__main__":
    env = WTRStrokeEnv()
    env.reset()