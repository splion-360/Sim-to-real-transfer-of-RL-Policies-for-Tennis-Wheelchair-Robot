
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
        goal_in_t: float = 4,
        collision_cost: float = 1,
        forward_reward_weight: float = 1e-5,
        terminate_when_unstable: bool = True,
        reset_noise_scale: float = 5e-3,
        error:float = 1e-7,
        radius:float = 0.5,
        maxv:float = 2.4,
        maxw:float = 4.8,
        render_mode: str = None,
        **kwargs):

        utils.EzPickle.__init__(self,
        xml_file,
        frame_skip,
        default_camera_config,
        goal_in_t,
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
        self.goal_in_t = goal_in_t
        self.collision_cost = collision_cost

        self.reset_noise_scale = reset_noise_scale
        self.forward_reward_weight = forward_reward_weight
  
        self.max_v,self.max_w = maxv,maxw
        self.error = error
        self.radius = radius
        self.failed,self.reached = False, False
        
        self.render_mode = render_mode
        self.goal = np.zeros(2)
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
        
        obs_size,action_size = 10,2

        self.observation_structure = {
            "skipped_qpos": 0,
            "qpos": self.data.qpos.size- (1 * 0),
            "qvel": self.data.qvel.size,
        }
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        ## Set linear and angular velocity limits based on the output from the differntial controller
        ## Linear Velocity limit is set as 4 m/s and Angular velocity limit is set as 8 rad/s (based on the Athletic paper following 1:2 ratio)
        ## Corresponding angular velocity limits for the wheels are -10 rad/s and 10 rad/s
        ## Corresponding linear velocity limits for the wheels are -11 m/s and 11 m/s
        ## Acceleration limit is still not set. |v| <= amax * dt and |w| <= alpha(tan) * dt. Centrifugal force limitation: v*w <= a(rad) * dt
        wheel_limits = 10 * 0.6 # Operating at 60% of max wheel velocity limits

        # atan, self.arad = 1, 1
        # wheel_limits = wheel_limits + atan * self.dt
        self.action_space = Box(low=-wheel_limits, high=wheel_limits, shape=(action_size,), dtype=np.float64)
        self.diagnoal_length = 16 * np.sqrt(2)
        self.init_pos = self.data.qpos.copy()
        
    def distance(self,pos1,pos2):
        return np.linalg.norm(pos1-pos2)
    
    def differential_drive(self,action):
        axle_length = 0.8671 # Original length x scaling factor (0.667*1.3)
        diameter = 0.6936 # Original diameter x scaling factor (0.578*1.2)
        v_des, w_des = action
        # Linear velocities
        vr,vl = (2*v_des + w_des*axle_length)/2,(2*v_des - w_des*axle_length)/2 
        # Angular velocities
        w1,w2 = 2*vr/diameter, 2*vl/diameter
        return np.array([w1,w2])
    
    def _get_obs(self):
        robot_pos = self.data.qpos[7:9].copy()
        qx,qy,qz,qw =  self.data.qpos[10:14].copy() # Quaternions 

        robot_vel = self.data.qvel[6:12].copy() # Robot Velocities

        yaw = np.arctan2(2*(qx*qy + qw*qz),1-2*(qy**2 + qz**2)) # Euler z
        if yaw < 0: yaw += 360 # Wrapping angles between [0,360)

        rel_x,rel_y = self.goal[0]-robot_pos[0], self.goal[1]-robot_pos[1]
        ## Compute the relative angle between the robot and the target [0,360)
        uncalibrated_angle = np.arctan(rel_y/rel_x)
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
        if rel_theta > 180: 
            rel_theta = 360 - rel_theta
        self.sweep = rel_theta
        ## Get goal distance
        goal_distance = self.distance(robot_pos,self.goal)
        self.prev_distance = goal_distance

        ## Normalise all the attributes before stacking as an observation vector
        goal_distance_normalised = goal_distance/self.diagnoal_length
        yaw_normalised = yaw/360
        theta_normalised = theta/360
        rel_theta_normalised = rel_theta/180

        observation = np.hstack((goal_distance_normalised, theta_normalised, rel_theta_normalised, yaw_normalised, robot_vel))
        return observation
    
    
    def control_cost(self, action):
        control_cost = 1e-3 * np.sum(np.square(action))
        return control_cost
    
    def has_toppled(self):
        sw1,sw2,sw3 = self.data.xpos[17],self.data.xpos[20],self.data.xpos[22]
        if not sw1[-1] < self.error or not sw2[-1] < self.error or not sw3[-1] < self.error:
            return True
        else: return False

    def has_collided(self):
        ball_pos = self.goal
        if self.data.qpos[0]-ball_pos[0] > self.error or self.data.qpos[1]-ball_pos[1] > self.error: 
            return True
        else: return False

    @property
    def reward(self):
        robot_pos = self.data.qpos[7:9].copy()
        curr_goal_distance = self.distance(robot_pos,self.goal)
        distance_rate = self.prev_distance - curr_goal_distance
        
        vx,vy,vz = self.data.qvel[6:9]
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        w = self.data.qvel[11]
        # centrifugal_error  = self.arad - abs(v*w) # Making sure the robot does not topple 
        reward = 500*distance_rate # Distance incentive

        if self.failed: 
            reward = -100 # Truncation if the robot hits the ball or does not reach the goal within the specified time. 
        if self.reached:
            reward = 120
        return reward #+ centrifugal_error

    def checkFlags(self):
        ## Check for target reach
        robot_pos = self.data.qpos[7:9].copy()
        curr_goal_distance = self.distance(robot_pos,self.goal)
        if curr_goal_distance <= self.radius: 
            self.reached = True
        
        ## Check for target timely reach and out of bounds error
        x,y = robot_pos
        conditions = [ (x < -8 or x > 8),
                       (y < -8 or y > 8),
                       self.data.time > self.t_accepted,]
                    #    self.has_toppled(),
                    #    self.has_collided()]
        if any(conditions):
            self.failed = True

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
        cx,cy = np.random.randint(low=-6,high=6) + np.random.uniform(low=noise_low, high=noise_high),\
                np.random.randint(low=-6,high=6) + np.random.uniform(low=noise_low, high=noise_high)
        self.data.qpos[0] = cx
        self.data.qpos[1] = cy
        self.set_state(self.data.qpos,self.data.qvel)
        ## Setting the original goal position
        self.goal = self.data.qpos[:2].copy()
        self.prev_distance = self.distance(self.goal,self.data.qpos[7:9].copy())
        
        ## Estimating the minimum time required to reach the goal from the starting position and adding grace time period of 2 seconds
        observation = self._get_obs()
        self.t_accepted = (self.distance(self.goal,self.data.qpos[7:9].copy()) / self.max_v) + (self.sweep/self.max_w) + 1
        return observation,{}

    @property
    def truncated(self):
        return self.failed
    @property
    def terminated(self):
        return self.reached

    def step(self,action):
        ## Action is an array of size 2 containing the angular velocities of the wheels
        ctrl = self.data.ctrl.copy()
        ctrl[-2:] = action
        self.do_simulation(ctrl,self.frame_skip)

        ctrl_cost = self.control_cost(action)  
        self.checkFlags()
        reward = self.reward - ctrl_cost
        terminated,truncated = self.terminated, self.truncated
        observation = self._get_obs()

        info = {
            "reward_goal": reward,
            "x_distance_from_origin": self.data.qpos[7] - self.init_qpos[8],
            "y_distance_from_origin": self.data.qpos[7] - self.init_qpos[8],
            "is_success": terminated
        }
        if self.render_mode == "human":
            self.render()
        
        return observation, reward,terminated,truncated,info
    
    def _get_reset_info(self):
        return {
            "reward_goal": 0,
            "x_distance_from_origin": 0,
            "y_distance_from_origin": 0,
            "is_success":False
        }
    
    def close(self):
        super().close()
    


if __name__ == "__main__":
    env = WTRBlockReacherEnv()
    breakpoint()
