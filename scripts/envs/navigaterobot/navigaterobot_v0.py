
from typing import Dict, Tuple
import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box,Tuple


DEFAULT_CAMERA_CONFIG = {
    "lookat": np.array([-19.788, 0.463, 9.488]),
}

class WTRBlockReacherEnv(MujocoEnv, utils.EzPickle):
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
        xml_file: str = os.path.join(full_path,"robots/wam7_blockreacher.xml"),
        frame_skip: int = 4,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        goal_not_in_t: float = 2,
        goal_in_t: float = 4,
        toppling_cost: float = 0.5,
        collision_cost: float = 1,
        forward_reward_weight: float = 1e-10,
        terminate_when_unstable: bool = True,
        reset_noise_scale: float = 5e-3,
        error:float = 1e-7,
        radius:float = 1,
        maxv:float = 10,
        maxw:float = 20,
        render_mode: str = None,
        **kwargs):

        utils.EzPickle.__init__(self,
        xml_file,
        frame_skip,
        default_camera_config,
        goal_not_in_t,
        goal_in_t,
        toppling_cost,
        collision_cost,
        forward_reward_weight,
        terminate_when_unstable,
        reset_noise_scale,
        error,
        radius,
        maxv,
        maxw,
        render_mode,
        **kwargs
        )

        ## Defining Rewards 
        self.goal_not_in_t = goal_not_in_t
        self.goal_in_t = goal_in_t
        self.toppling_cost = toppling_cost
        self.collision_cost = collision_cost
        self.reset_noise_scale = reset_noise_scale
        self.forward_reward_weight = forward_reward_weight
  
        self.max_v,self.max_w = maxv,maxw
        self.error = error
        self.radius = radius
        self.collided,self.toppled = False,False
        
        self.render_mode = render_mode
        self.original_goal = np.zeros(2)
        self.t_accepted = 0
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
        
        obs_size,action_size = 14,2 # 2 (xr,yr) + 2 (xb,yb), (w1,w2)
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64) 
        self.observation_structure = {
            "skipped_qpos": 0,
            "qpos": self.data.qpos.size- (1 * 0),
            "qvel": self.data.qvel.size,
        }
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        ## Set linear and angular velocity limits
        
        self.action_space = Box(low=-self.max_v, high=self.max_v, shape=(action_size,), dtype=np.float64)
        self.og_qpos = self.data.qpos.copy()
        
    def distance(self,pos1,pos2):return np.linalg.norm(pos1-pos2)
    
    def differential_drive(self,action):
        axle_length = 0.8671 # Original length x scaling factor (0.667*1.3)
        diameter = 0.6936 # Original diameter x scaling factor (0.578*1.2)
        v_des, w_des = action
        # Linear velocities
        vr,vl = (2*v_des + w_des*axle_length)/2,(2*v_des - w_des*axle_length)/2 
        # Angular velocities
        w1,w2 = 2*vr/diameter, 2*vl/diameter
        return np.array([w1,w2])
    
    '''
    Reward definitions
    '''
    @property
    def forward_reward(self):
        curr_dist = self.distance(self.data.qpos[7:9],self.original_goal)
        reward = 1e-2*(self.prev_distance-curr_dist)
        # self.prev_distance = min(curr_dist,self.prev_distance)

        # dratio = self.distance(self.data.qpos[7:9],self.original_goal)/self.distance(self.og_qpos[7:9],self.original_goal)
        # reward = np.clip(1-dratio,-1,1)
        return reward
    
    @property
    def velocity_penalty(self):
        if abs(self.data.qvel[9])+abs(self.data.qvel[10])+abs(self.data.qvel[11]) > 0.09: return -1
        else: return np.clip(1-(abs(self.data.qvel[9])+abs(self.data.qvel[10])+abs(self.data.qvel[11])),-1,1)

    @property
    def goal_reward(self):
        if self.reached_goal():return self.goal_in_t
        else: return 0

    @property
    def stability_reward(self):
        return self.has_collided()*self.collision_cost #+ self.has_toppled()*self.toppling_cost
    
    def reached_goal(self):
        pos = self.data.qpos[7:9]
        distance = self.distance(pos,self.original_goal)
        if distance <= self.radius:return True
        else:return False
    
    def encourage_linear_reward(self,xpos1,xpos2):
        velocity_x = (xpos1-xpos2)/self.dt
        return self.forward_reward_weight*(velocity_x)
    
    def control_cost(self, action):
        control_cost = 1e-3 * np.sum(np.square(action))
        return control_cost

    # def reached_in_time(self):
    #     if self.reached_goal() and self.data.time <= self.t_accepted: return True
    #     else: return False

    def timeout(self):
        if self.data.time > self.t_accepted: 
            self.toppled = True
            return -1
        else: return 0

        
    def has_collided(self):
        ball_pos = self.original_goal
        robot_pos = self.data.qpos[7:9]

        if self.distance(ball_pos,robot_pos) < 0.2*self.radius: 
            self.collided = True
            return True
        else:return False

    def has_toppled(self):
        sw1,sw2,sw3 = self.data.xpos[17],self.data.xpos[20],self.data.xpos[22]
        if not sw1[-1] < self.error or not sw2[-1] < self.error or not sw3[-1] < self.error:
            self.toppled = True
            return True
        else: return False


    def _get_obs(self):
        robot_pos = self.data.qpos[7:14].copy()
        distance_to_goal = np.array(self.distance(robot_pos[:2],self.original_goal))
        robot_vel = self.data.qvel[6:12].copy()
        observation = np.hstack((robot_pos,robot_vel,distance_to_goal))
        return observation

    def reset(self,seed=None,options=None):
        np.random.seed(seed)
        # Random initialization of the ball coordinates
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        # Setting the flags to false
        self.collided,self.toppled = False,False
        
        self.data.qpos[0] = np.random.randint(low=-6,high=6) + np.random.uniform(low=noise_low, high=noise_high) 
        self.data.qpos[1] = np.random.randint(low=-6,high=6) + np.random.uniform(low=noise_low, high=noise_high)
        self.data.qpos[7:] = self.og_qpos[7:].copy()
        self.data.qvel[:] = np.zeros(self.model.nv)
        self.set_state(self.data.qpos,self.data.qvel)
        observation = self._get_obs()
        self.data.time = 0.0
        ## Setting the original goal position
        self.original_goal = self.data.qpos[:2].copy()
        self.prev_distance = self.distance(self.original_goal,self.data.qpos[7:9])

        ## Estimating the minimum time required to reach the goal from the starting position and addigng grace time period of 5 seconds

        self.t_accepted = (self.distance(self.original_goal,self.data.qpos[7:9])/self.max_v) + 15
        return observation,{}

    @property
    def terminated(self):return self.reached_goal()
    @property
    def truncated(self):
        return self.collided or self.toppled

    def step(self,action):
        x_position_before = self.data.qpos[7]
        ## Action is an array of size 2 containing the angular velocities of the wheels
        ctrl = self.data.ctrl.copy()
        ctrl[-2:] = self.differential_drive(action)
        self.do_simulation(ctrl,self.frame_skip)
        x_position_after = self.data.qpos[7]

        linear_reward = self.encourage_linear_reward(x_position_before,x_position_after)
        ctrl_cost = self.control_cost(action)
        goal_reward,stability_reward = self.goal_reward,self.stability_reward
        
        reward = goal_reward + self.forward_reward - ctrl_cost + linear_reward + self.timeout() #+ (self.t_accepted-self.data.time) 
        #+ self.velocity_penalty - stability_reward #- stability_reward + goal_reward 
        
        terminated = self.terminated
        truncated = self.truncated
        observation = self._get_obs()
        info = {
            "reward_goal": goal_reward,
            "reward_stability": -stability_reward,
            "x_distance_from_origin": self.data.qpos[7] - self.init_qpos[7],
            "y_distance_from_origin": self.data.qpos[8] - self.init_qpos[8],
        }
        if self.render_mode == "human":self.render()
        return observation, reward,terminated,truncated,info
    
    def _get_reset_info(self):
        return {
            "reward_goal": 0,
            "reward_stability": 0,
            "x_distance_from_origin": 0,
            "y_distance_from_origin": 0,
        }
    def close(self):pass
    


if __name__ == "__main__":
    env = WTRBlockReacherEnv()
    breakpoint()
