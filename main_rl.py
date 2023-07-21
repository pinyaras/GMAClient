import argparse
import gymnasium as gym
import numpy as np
import pathlib
import json
import sys
import time
from network_gym_client import Env as NetworkGymEnv
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers import NormalizeObservation

#MODEL_SAVE_FREQ = 1000
#LOG_INTERVAL = 10
NUM_OF_EVALUATE_EPISODES = 5

#checkpoint_callback = CheckpointCallback(
#    save_freq=MODEL_SAVE_FREQ,  # Save the model every 1 episode
#    save_path='./models/',
#    name_prefix='rl_model' # should be passed from the json file.
#)

def train(agent, config_json):

    num_steps = 0

    #configure the num_steps based on the json file
    if (config_json['gmasim_config']['GMA']['measurement_interval_ms'] + config_json['gmasim_config']['GMA']['measurement_guard_interval_ms']
        == config_json['gmasim_config']['Wi-Fi']['measurement_interval_ms'] + config_json['gmasim_config']['Wi-Fi']['measurement_guard_interval_ms']
        == config_json['gmasim_config']['LTE']['measurement_interval_ms'] + config_json['gmasim_config']['LTE']['measurement_guard_interval_ms']):
        num_steps = int(config_json['gmasim_config']['simulation_time_s'] * 1000/config_json['gmasim_config']['GMA']['measurement_interval_ms'])
    else:
        print(config_json['gmasim_config']['GMA']['measurement_interval_ms'])
        print(config_json['gmasim_config']['GMA']['measurement_guard_interval_ms'])
        print(config_json['gmasim_config']['Wi-Fi']['measurement_interval_ms'])
        print(config_json['gmasim_config']['Wi-Fi']['measurement_guard_interval_ms'])
        print(config_json['gmasim_config']['LTE']['measurement_interval_ms'])
        print(config_json['gmasim_config']['LTE']['measurement_guard_interval_ms'])
        sys.exit('[Error!] The value of GMA, Wi-Fi, and LTE measurement_interval_ms + measurement_guard_interval_ms should be the same!')
    
    #model = agent.learn(total_timesteps=num_steps,log_interval=LOG_INTERVAL, callback=checkpoint_callback)
    model = agent.learn(total_timesteps=num_steps)
    model.save(config_json['rl_agent_config']['agent'] )

    #TODD Terminate the RL agent when the simulation ends.
def system_default_policy(env, config_json):

    num_steps = 0

    #configure the num_steps based on the json file
    if (config_json['gmasim_config']['GMA']['measurement_interval_ms'] + config_json['gmasim_config']['GMA']['measurement_guard_interval_ms']
        == config_json['gmasim_config']['Wi-Fi']['measurement_interval_ms'] + config_json['gmasim_config']['Wi-Fi']['measurement_guard_interval_ms']
        == config_json['gmasim_config']['LTE']['measurement_interval_ms'] + config_json['gmasim_config']['LTE']['measurement_guard_interval_ms']):
        num_steps = int(config_json['gmasim_config']['simulation_time_s'] * 1000/config_json['gmasim_config']['GMA']['measurement_interval_ms'])
    else:
        print(config_json['gmasim_config']['GMA']['measurement_interval_ms'])
        print(config_json['gmasim_config']['GMA']['measurement_guard_interval_ms'])
        print(config_json['gmasim_config']['Wi-Fi']['measurement_interval_ms'])
        print(config_json['gmasim_config']['Wi-Fi']['measurement_guard_interval_ms'])
        print(config_json['gmasim_config']['LTE']['measurement_interval_ms'])
        print(config_json['gmasim_config']['LTE']['measurement_guard_interval_ms'])
        sys.exit('[Error!] The value of GMA, Wi-Fi, and LTE measurement_interval_ms + measurement_guard_interval_ms should be the same!')

    done = True
    for step in range(num_steps):
        # If the epsiode is up, then start another one
        if done:
            obs = env.reset()

        action = np.array([])#no action from the rl agent

        # apply the action
        obs, reward, done, done, info = env.step(action)

        print(obs)

        if info['terminate_flag']:
            break

def evaluate(model, env, n_episodes=NUM_OF_EVALUATE_EPISODES):
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
    FILE_PATH = pathlib.Path(__file__).parent
    #common_config.json is shared by all environments
    f = open(FILE_PATH / 'network_gym_client/common_config.json')
    common_config_json = json.load(f)
    
    #load the environment dependent config file
    file_name = 'network_gym_client/envs/' +args.env + '/config.json'
    f = open(FILE_PATH / file_name)

    use_case_config_json = json.load(f)
    config_json = {**common_config_json, **use_case_config_json}
    config_json['gmasim_config']['env'] = args.env

    if args.lte_rb !=-1:
        config_json['gmasim_config']['LTE']['resource_block_num'] = args.lte_rb

    if config_json['rl_agent_config']['agent'] == "" or config_json['rl_agent_config']['agent'] == "system_default":
        # rl agent disabled, use the default policy from the system
        config_json['rl_agent_config']['agent']  = 'system_default'
        config_json['gmasim_config']['GMA']['respond_action_after_measurement'] = False
    else:
        #ml algorithm
        if not config_json['gmasim_config']['GMA']['respond_action_after_measurement']:
            sys.exit('[Error!] RL agent must set "respond_action_after_measurement" to true !')
    
    rl_alg = config_json['rl_agent_config']['agent'] 

    config = {
        "policy_type": "MlpPolicy",
        "env_id": "network_gym_client",
        "RL_algo" : rl_alg
    }

    run = wandb.init(
        # name=rl_alg + "_" + str(config_json['gmasim_config']['num_users']) + "_LTE_" +  str(config_json['gmasim_config']['LTE']['resource_block_num']),
        #name=rl_alg + "_" + str(config_json['gmasim_config']['num_users']) + "_" +  str(config_json['gmasim_config']['LTE']['resource_block_num']),
        name=rl_alg,
        # project="netai-gym",
        project="gmasim-gym",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # save_code=True,  # optional
    )

    alg_map = {
        'PPO': PPO,
        'DDPG': DDPG,
        'SAC': SAC,
        'TD3': TD3,
        'A2C': A2C,
        'system_default': system_default_policy,
    }

    # Choose the agent
    agent_class = alg_map.get(rl_alg, None)
    if agent_class is None:
        raise ValueError(f"Invalid RL algorithm name: {rl_alg}")
    client_id = args.client_id
    # Create the environment
    print("[" + args.env + "] environment selected.")
    env = NetworkGymEnv(client_id, wandb, config_json) # make a network env using pass client id, adatper and configure file arguements.
    normal_obs_env = NormalizeObservation(env)
    # It will check your custom environment and output additional warnings if needed
    # only use this function for debug, 
    # check_env(env)

    if rl_alg != "system_default":

        train_flag = config_json['rl_agent_config']['train']
        #link_type = config_json['rl_agent_config']['link_type']

        # Load the model if eval is True
        if not train_flag:
            # Testing/Evaluation
            path = "models/trained_models/" + rl_alg
            agent = agent_class.load(path)
            # n_episodes = config_json['rl_agent_config']['timesteps'] / 100

            evaluate(agent, normal_obs_env)
        else:
            # Train the agent
            agent = agent_class(config_json['rl_agent_config']['policy'], normal_obs_env, verbose=1, tensorboard_log=f"runs/{run.id}")
            train(agent, config_json)
    else:
        #use the system_default algorithm...
        agent_class(normal_obs_env, config_json)
        
def arg_parser():
    parser = argparse.ArgumentParser(description='Network Gym Client')
    parser.add_argument('--env', type=str, required=True, choices=['nqos_split', 'qos_steer', 'network_slicing'],
                        help='Select a environment to start Network Gym Client (nqos_split, qos_steer, network_slicing)')
    parser.add_argument('--client_id', type=int, required=False, default=0,
                        help='Select client id to start simulation')
    parser.add_argument('--lte_rb', type=int, required=False, default=-1,
                        help='Select number of LTE Resource Blocks')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    time.sleep(10)