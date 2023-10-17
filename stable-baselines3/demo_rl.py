import argparse
import numpy as np
import pathlib
import json
import sys
import time

import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')

from network_gym_client import load_config_file
from network_gym_client import Env as NetworkGymEnv
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers import NormalizeObservation

def train(agent, config_json):

    steps_per_episode = int(config_json['env_config']['steps_per_episode'])
    episodes_per_session = int(config_json['env_config']['episodes_per_session'])
    num_steps = steps_per_episode*episodes_per_session
    
    model = agent.learn(total_timesteps=num_steps)
    model.save(config_json['rl_config']['agent'] )

def system_default_policy(env, config_json):

    steps_per_episode = int(config_json['env_config']['steps_per_episode'])
    episodes_per_session = int(config_json['env_config']['episodes_per_session'])
    num_steps = steps_per_episode*episodes_per_session

    truncated = True # episode end
    terminated = False # episode end and simulation end
    obs, info = env.reset()
    for step in range(num_steps):

        action = np.array([])#no action from the rl agent

        # apply the action
        obs, reward, terminated, truncated, info = env.step(action)
        #print(obs)

        # If the environment is end, exit
        if terminated:
            break

        # If the epsiode is up (environment still running), then start another one
        if truncated:
            obs, info = env.reset()

def random_policy(env, config_json):

    steps_per_episode = int(config_json['env_config']['steps_per_episode'])
    episodes_per_session = int(config_json['env_config']['episodes_per_session'])
    num_steps = steps_per_episode*episodes_per_session

    truncated = True # episode end
    terminated = False # episode end and simulation end
    obs, info = env.reset()
    for step in range(num_steps):

        action = env.action_space.sample()  # agent policy that uses the observation and info
        # apply the action
        obs, reward, terminated, truncated, info = env.step(action)
        #print(obs)

        # If the environment is end, exit
        if terminated:
            break

        # If the epsiode is up (environment still running), then start another one
        if truncated:
            obs, info = env.reset()


def evaluate(model, env, n_episodes):
    rewards = []
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    avg_reward = sum(rewards) / n_episodes
    return avg_reward

def main():

    args = arg_parser()

    #load config files
    config_json = load_config_file(args.env)
    config_json['env_config']['env'] = args.env
    config_json['rl_config']['agent']  = args.agent
    
    rl_alg = config_json['rl_config']['agent'] 

    config = {
        "policy_type": "MlpPolicy",
        "env_id": "network_gym_client",
        "RL_algo" : rl_alg
    }

    alg_map = {
        'PPO': PPO,
        'DDPG': DDPG,
        'SAC': SAC,
        'TD3': TD3,
        'A2C': A2C,
        'system_default': system_default_policy,
        'random': random_policy,
    }

    # Choose the agent
    agent_class = alg_map.get(rl_alg, None)
    if agent_class is None:
        raise ValueError(f"Invalid RL algorithm name: {rl_alg}")
    client_id = args.client_id
    # Create the environment
    print("[" + args.env + "] environment with [" + args.agent + "] agent.")
    env = NetworkGymEnv(client_id, config_json) # make a network env using pass client id, adatper and configure file arguements.
    normal_obs_env = NormalizeObservation(env)
    # It will check your custom environment and output additional warnings if needed
    # only use this function for debug, 
    # check_env(env)

    if rl_alg != "system_default" and rl_alg != "random":

        train_flag = True

        # Load the model if eval is True
        if not train_flag:
            # Testing/Evaluation
            path = "models/trained_models/" + rl_alg
            agent = agent_class.load(path)
            # n_episodes = config_json['rl_config']['timesteps'] / 100

            evaluate(agent, normal_obs_env, 5)
        else:
            # Train the agent
            agent = agent_class('MlpPolicy', normal_obs_env, verbose=1)
            train(agent, config_json)
    else:
        #use the system_default algorithm...
        agent_class(normal_obs_env, config_json)
        
def arg_parser():
    parser = argparse.ArgumentParser(description='Network Gym Client')
    parser.add_argument('--env', type=str, required=True, choices=['nqos_split', 'qos_steer', 'network_slicing'],
                        help='Select a environment to start Network Gym Client (nqos_split, qos_steer, network_slicing)')
    parser.add_argument('--agent', type=str, required=False, default='system_default', choices=['PPO', 'DDPG', 'SAC', 'TD3', 'A2C', 'system_default', 'random'],
                        help='Select agent from stable-baselines3')
    parser.add_argument('--client_id', type=int, required=False, default=0,
                        help='Select client id to start simulation')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    time.sleep(1)