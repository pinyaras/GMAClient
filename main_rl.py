import argparse
import gym
import numpy as np
import pathlib
import json
import sys
import time
#Import gmagym environment
from gma_gym import GmaSimEnv

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from nqos_split.nqos_split_helper import single_link_policy
from nqos_split.nqos_split_helper import nqos_split_helper
from qos_steer.qos_steer_helper import qos_steer_helper
from network_slicing.network_slicing_helper import network_slicing_helper

MODEL_SAVE_FREQ = 1000
LOG_INTERVAL = 10
NUM_OF_EVALUATE_EPISODES = 5

checkpoint_callback = CheckpointCallback(
    save_freq=MODEL_SAVE_FREQ,  # Save the model every 1 episode
    save_path='./models/',
    name_prefix='rl_model' # should be passed from the json file.
)

def train(agent, config_json):

    num_steps = 0

    #configure the num_steps based on the JSON file
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
    
    model = agent.learn(total_timesteps=num_steps,log_interval=LOG_INTERVAL, callback=checkpoint_callback)
    model.save(config_json['rl_agent_config']['agent'] )

def gma_policy(env, config_json):

    num_steps = 0

    #configure the num_steps based on the JSON file
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

        # take random action, but you can also do something more intelligent
        # action = my_intelligent_agent_fn(obs) 

        action = np.array([])

        # apply the action
        obs, reward, done, info = env.step(action)


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
    #common_config.json is shared by all use cases
    f = open(FILE_PATH / 'common_config.json')
    common_config_json = json.load(f)
    
    #load the use case dependent config file
    file_name = args.use_case + '/' + args.use_case +'_config.json'
    f = open(FILE_PATH / file_name)

    use_case_config_json = json.load(f)
    config_json = {**common_config_json, **use_case_config_json}
    config_json['gmasim_config']['use_case'] = args.use_case

    if args.use_case == "network_slicing":
        # for network slicing, the user number is configured using the slice list. Cannot use the argument parser!
        config_json['gmasim_config']['num_users'] = 0

        for item in config_json['gmasim_config']['slice_list']:
            config_json['gmasim_config']['num_users'] += item['num_users']
        if args.num_users != -1:
            sys.exit("cannot config user number in terminal number for network slicing use case.")

    if args.num_users != -1:
        config_json['gmasim_config']['num_users'] = args.num_users
    
    print("[" + str(config_json['gmasim_config']['num_users']) + "] Number of users selected.")

    if args.lte_rb !=-1:
        config_json['gmasim_config']['LTE']['resource_block_num'] = args.lte_rb


    rl_alg = config_json['rl_agent_config']['agent'] 

    if not config_json['enable_rl_agent'] :
        # rl agent disabled, use the default policy from GMA
        rl_alg = 'GMA'
        config_json['gmasim_config']['GMA']['respond_action_after_measurement'] = False
    else:
        #ml algorithm
        if not config_json['gmasim_config']['GMA']['respond_action_after_measurement']:
            sys.exit('[Error!] RL agent must set "respond_action_after_measurement" to true !')

    config = {
        "policy_type": "MlpPolicy",
        "env_id": "Gmasim",
        "RL_algo" : rl_alg
    }

    run = wandb.init(
        # name=rl_alg + "_" + str(config_json['gmasim_config']['num_users']) + "_LTE_" +  str(config_json['gmasim_config']['LTE']['resource_block_num']),
        #name=rl_alg + "_" + str(config_json['gmasim_config']['num_users']) + "_" +  str(config_json['gmasim_config']['LTE']['resource_block_num']),
        name=rl_alg,
        project="gmasim-gym",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # save_code=True,  # optional
    )

    if config_json['algorithm_client_identity'] == 'test_id':
        #username = wandb.api_key.split()[0].split("@")[0]
        username = run.entity
        print(username)
        if username == '':
            print('***[WARNING]*** You are using the default "test_id" to connect to the server, which may conflict with the simulations launched by other users.')
            print('***[WARNING]*** Please change the "algorithm_client_identity" attribute in the common_config.json file to your email address.')
        else:
            print('***[WARNING]*** You are using wandb username ['+username+'] as algorithm client id to connect to the server')
            print('***[WARNING]*** Please change the "algorithm_client_identity" attribute in the common_config.json file to your email address.')
            config_json['algorithm_client_identity'] = username

    alg_map = {
        'PPO': PPO,
        'DDPG': DDPG,
        'SAC': SAC,
        'TD3': TD3,
        'A2C': A2C,
        'GMA': gma_policy,
        'LTE': single_link_policy,
        'Wi-Fi': single_link_policy
    }

    #model_map = {
    #    'MlpPolicy': "MlpPolicy",
    #    'CustomLSTMPolicy': "CustomLSTMPolicy",
    #}

    # Choose the agent
    agent_class = alg_map.get(rl_alg, None)
    if agent_class is None:
        raise ValueError(f"Invalid RL algorithm name: {rl_alg}")
    #client_id = list(alg_map.keys()).index(rl_alg) + 1
    client_id = args.client_id
    # Create the environment
    if(args.use_case == "nqos_split"):
        use_case_helper = nqos_split_helper(wandb)
    elif(args.use_case == "qos_steer"):
        use_case_helper = qos_steer_helper(wandb)
    elif(args.use_case == "network_slicing"):
        use_case_helper = network_slicing_helper (wandb)
    else:
       sys.exit("[" + args.use_case + "] use case is not implemented.")

    print("[" + args.use_case + "] use case selected.")

    use_case_helper.set_config(config_json)
    env = GmaSimEnv(client_id, use_case_helper, config_json) # pass id, and configure file

    if config_json['enable_rl_agent']:

        train_flag = config_json['rl_agent_config']['train']
        #link_type = config_json['rl_agent_config']['link_type']

        if rl_alg == "LTE" or rl_alg == "Wi-Fi" :
            single_link_policy(env, config_json)
            return
        # Load the model if eval is True
        if not train_flag:
            # Testing/Evaluation
            path = "models/trained_models/" + rl_alg
            agent = agent_class.load(path)
            # n_episodes = config_json['rl_agent_config']['timesteps'] / 100

            evaluate(agent, env)
        else:
            # Train the agent
            agent = agent_class(config_json['rl_agent_config']['policy'], env, verbose=1, tensorboard_log=f"runs/{run.id}")
            train(agent, config_json)
    else:
        #use the GMA algorithm...
        agent_class(env, config_json)
        
def arg_parser():
    parser = argparse.ArgumentParser(description='GMAsim Client')
    parser.add_argument('--use_case', type=str, required=True, choices=['nqos_split', 'qos_steer', 'network_slicing'],
                        help='Select a use case to start GMAsim (nqos_split, qos_steer, network_slicing).')
    parser.add_argument('--client_id', type=int, required=False, default=0,
                        help='Select client id to start simulation).')
    parser.add_argument('--num_users', type=int, required=False, default=-1,
                        help='Select number of users')
    parser.add_argument('--lte_rb', type=int, required=False, default=-1,
                        help='Select number of LTE Resource Blocks')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    time.sleep(10)