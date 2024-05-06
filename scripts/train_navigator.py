import gymnasium as gym
from stable_baselines3 import PPO,A2C,DDPG,TD3,SAC
import os
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse
import torch 
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env


NCPUS = 2
def create_env(env_id, rank, render_mode, seed = 0):
    def _init():
        env = gym.make(env_id, render_mode = render_mode)
        env.reset(seed = seed + rank)
        return env.unwrapped
    set_random_seed(seed)
    return _init


class PlotsCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(PlotsCallback, self).__init__(verbose)

    def _on_rollout_end(self) -> None:
    
        try:
            ## Logging velocities and errors with respect to the number of episodes
            info = self.model.env.unwrapped.get_attr("datatoplot")[0]
            self.logger.record("rollout/linear_velocity",  info["linear_velocity"])
            self.logger.record("rollout/angular_velocity", info["angular_velocity"])
            self.logger.record("rollout/positional_error", info["positional_error"])
            self.logger.record("rollout/goal_velocity",    info["velocity_differential"])
        except: pass

        ### Logging for a different task
        try:
            for i in range(7):
                self.logger.record(f"rollout/joint_positions_{i}",  info[f"joint_positions_{i}"])
                self.logger.record(f"rollout/angular_velocity_{i}",  info[f"angular_velocity_{i}"])
                self.logger.record(f"rollout/angular_acceleration_{i}",  info[f"angular_acceleration_{i}"])
            self.logger.record("rollout/positional_error", info["positional_error"])
        except:pass


    def _on_step(self) -> bool:
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="Environment to be used for training. Currently only support for BlockReacher_v2 & ArmReacher_v1 is provided", 
                        choices=["blockreacher","armreacher"])
    parser.add_argument('--max_episode_steps', type=int,default=2000, help='Max number of steps to run per episode irrespective of termination')
    parser.add_argument('--model', type=str, default='DDPG', help="Model to use for training", choices=["DDPG","SAC","TD3", "PPO"])
    parser.add_argument('--save_dir', type=str, default="./saved_models/", help="directory to save the models")
    parser.add_argument('--save_frequency',type=int,default= 50 ,help="model saving frequency")
    parser.add_argument('--total_timesteps', type=int, default=10000, help="Number of timesteps to train the model")
    parser.add_argument('--render_mode', type=str, default="human", help='Dynamic rendering of the environment during training',choices=["human","rgb_array","depth_array",None])
    parser.add_argument('--version', type=int, default=0)

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
     max_episode_steps=args.max_episode_steps
     )
    
    logdir = os.path.join("./runs",task)
    # env = SubprocVecEnv([create_env("BlockReacher-v2",rank,args.render_mode) for rank in range(NCPUS)],'fork')
    env =  gym.make(env_id,render_mode=args.render_mode)
    # policy_kwargs = dict(activation_fn=torch.nn.ReLU,
    #                  net_arch=dict(pi=[400, 300, 300], qf=[400, 300, 300]))
    if args.model == "PPO":
        model = PPO("MlpPolicy", env, verbose=1,tensorboard_log=logdir)
    elif args.model == "DDPG":
        model = DDPG("MlpPolicy", env, verbose=1,tensorboard_log=logdir)
    elif args.model == "TD3":
        model = TD3("MlpPolicy", env, verbose=1,tensorboard_log=logdir)
    elif args.model == "SAC":
        model = SAC("MlpPolicy", env, verbose=1,tensorboard_log=logdir)

    ## Save the model 
    filename = f"{args.model}_v{args.version}"
    path = os.path.join(args.save_dir,task, filename)
    if not os.path.isdir(path):
        os.makedirs(path)
    env.close()


    ## Start training
    callback = PlotsCallback()
    for i in range(args.save_frequency):
        model.learn(total_timesteps=args.total_timesteps,reset_num_timesteps=False,tb_log_name=filename,callback=callback)
        model.save(os.path.join(path,f"v{args.version}_{i}"))
    env.close()


if __name__ == "__main__":
    main()