import gymnasium as gym
from stable_baselines3 import PPO,A2C,DDPG,TD3, SAC
import os
from typing import Callable
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers.monitoring import video_recorder
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from gymnasium.envs.registration import register
import argparse
import warnings
warnings.filterwarnings("ignore")


def main():
     parser = argparse.ArgumentParser()

     parser.add_argument('--env', type=str, help="Environment to be used for training. Currently only support for BlockReacher_v2 & ArmReacher_v1 is provided", 
                         choices=["blockreacher","armreacher"])
     parser.add_argument('--max_episode_steps', type=int,default = 1000, help='Max number of steps to run per episode irrespective of termination')
     parser.add_argument('--model', type=str, default='SAC', help="Model to be loaded", choices=["DDPG","SAC","TD3", "PPO"])
     parser.add_argument('--save_dir', type=str, default="./saved_models/", help="directory where the model to be loaded from")
     parser.add_argument('--filename',type = str,help="file path")
     parser.add_argument('--render_mode', type=str, default=None, help='Dynamic rendering of the environment during training',choices=["human","rgb_array","depth_array",None])
     parser.add_argument('--record', type=bool, default=False, help='Records the environment. Please set the render_mode to rgb_array if saving')
     
     args = parser.parse_args()

     # Register as gym environment 
     if args.env == "blockreacher":
        env_id = "BlockReacher_v2"
        entry_point = "envs.navigaterobot.navigaterobot_v2:WTRBlockReacherEnv"
        task = "navigaterobot"
     elif args.env == "armreacher":
        env_id = "ArmReacher_v1"
        entry_point = "envs.armreacher.armreacher_v1:WTRArmReacherEnv"
        task = "armreacher"

     register(
          id= env_id,
          entry_point= entry_point,
          max_episode_steps=args.max_episode_steps)

     if not args.record: 
          env =  gym.make(env_id,render_mode=args.render_mode)

     else:
          vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode=args.render_mode)])
          # Record the video starting at the first step
          if not os.path.isdir('./assets/videos'):
               os.makedirs(path)
          env = VecVideoRecorder(vec_env, './assets/videos/',
                              record_video_trigger=lambda x: x == 0, video_length=200,
                              name_prefix=f"agent-{env_id}")
     ## load the model 
     env.reset()
     path = os.path.join(args.save_dir,task,args.filename)
  
     if args.model == "PPO":
          model = PPO.load(path,env)
     elif args.model == "DDPG":
          model = DDPG.load(path,env)
     elif args.model == "TD3":
          model = TD3.load(path,env)
     elif args.model == "SAC":
          model = SAC.load(path,env)

     mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
     # vec_env = model.get_env()
     obs = env.reset()
     for i in range(1000):
          action, _states = model.predict(obs, deterministic=True)
          obs, rewards, dones, info = env.step(action)
          env.render(args.render_mode)
     env.close()

if __name__ == "__main__":
     main()