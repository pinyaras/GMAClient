import gym
import numpy as np
from gym import spaces
from gym.spaces import Box
import pandas as pd
from stable_baselines3.common.running_mean_std import RunningMeanStd

import pathlib
import json
import sys
try:
    # Try to import from the same directory
    from netai_gym_open_api import api_client as netai_gym_api_client
except ImportError:
    from .netai_gym_open_api import api_client as netai_gym_api_client

np.set_printoptions(precision=3)

FILE_PATH = pathlib.Path(__file__).parent

STEPS_PER_EPISODE = 100

class NetAIEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, id, use_case_helper, config_json):
        super().__init__()

        #Define config params
        self.use_case_helper = use_case_helper
        self.num_users = int(config_json['gmasim_config']['num_users'])
        self.rl_alg = config_json['rl_agent_config']['agent'] 
        self.enable_rl_agent = config_json['enable_rl_agent'] 
        self.action_space = use_case_helper.get_action_space()

        num_features = use_case_helper.get_num_of_observation_features()

        if config_json["rl_agent_config"]['input']  == "flat":
            self.observation_space = spaces.Box(low=0, high=1000,
                                                shape=(self.num_users*num_features,), dtype=np.float32)
        elif config_json["rl_agent_config"]['input']  == "matrix":
            self.observation_space = spaces.Box(low=0, high=1000,
                                                shape=(num_features,self.num_users), dtype=np.float32)
        else:                                                
            sys.exit("[" + config_json["rl_agent_config"]['input']  + "] input type not valid.")

        self.normalize_obs = RunningMeanStd(shape=self.observation_space.shape)
        self.netai_gym_api_client = netai_gym_api_client(id, config_json) #initial netai_gym_api_client
        #self.max_counter = int(config_json['gmasim_config']['simulation_time_s'] * 1000/config_json['gmasim_config']['GMA']['measurement_interval_ms'])# Already checked the interval for Wi-Fi and LTE in the main file

        #self.link_type = config_json['rl_agent_config']['link_type'] 
        self.current_step = 0
        self.max_steps = STEPS_PER_EPISODE
        self.current_ep = 0
        self.first_episode = True
        self.netai_gym_api_client.connect()

        self.last_action_list = []

        
    def reset(self):
        self.counter = 0
        self.current_step = 0
        # connect to the netai server and and receive the first measurement
        if not self.first_episode:
            self.netai_gym_api_client.send(self.last_action_list) #send action to netai server

<<<<<<< HEAD:gma_gym.py
        ok_flag, terminate_flag, df_list = self.gmasim_client.recv()#first measurement
=======
        ok_flag, df_list = self.netai_gym_api_client.recv()#first measurement
>>>>>>> b39d339b1482a7ea76a6449fc592f65ef2a4b84d:netai_gym.py
        df_phy_lte_max_rate = df_list[0]
        df_phy_wifi_max_rate = df_list[1]
        df_load = df_list[2]
        df_rate = df_list[3]
        df_qos_rate = df_list[4]
        df_owd = df_list[5]
        df_split_ratio = df_list[6]
        df_ap_id = df_list[7]
        df_phy_lte_slice_id = df_list[8]
        df_phy_lte_rb_usage = df_list[9]
        df_delay_violation = df_list[10]
        #print(df_phy_lte_slice_id)
        #print(df_phy_lte_rb_usage)
        #print(df_delay_violation)

        if self.enable_rl_agent and not ok_flag:
            print("[WARNING], some users may not have a valid measurement, for qos_steering case, the qos_test is not finished before a measurement return...")

        observation = self.use_case_helper.prepare_observation(df_list)

        # print(observation.shape)
        self.normalize_obs.update(observation)
        normalized_obs = (observation - self.normalize_obs.mean) / np.sqrt(self.normalize_obs.var)

        self.current_ep += 1

        return normalized_obs.astype(np.float32)
        # return observation  # reward, done, info can't be included

    def send_action(self, actions):

        if not self.enable_rl_agent or actions.size == 0:
            #empty action
            self.netai_gym_api_client.send([]) #send empty action to netai server
            return
        if self.rl_alg == "Wi-Fi" or  self.rl_alg == "LTE":
            self.netai_gym_api_client.send(actions) #send empty action to netai server
            return           

        action_list = self.use_case_helper.prepare_action(actions)

        self.last_action_list = action_list
        self.netai_gym_api_client.send(action_list) #send action to netai server



    #def df_load_to_dict(self, df):
    #    df['user'] = df['user'].map(lambda u: f'UE{u}_tx_rate')
    #    # Set the index to the 'user' column
    #    df = df.set_index('user')
    #    # Convert the DataFrame to a dictionary
    #    data = df['value'].to_dict()
    #    return data


    def get_obs_reward(self):
<<<<<<< HEAD:gma_gym.py
        #receive measurement from GMAsim server
        ok_flag, terminate_flag, df_list = self.gmasim_client.recv()

        if terminate_flag == True:
            return [], 0, [], [], terminate_flag
=======
        #receive measurement from netai server
        ok_flag, df_list = self.netai_gym_api_client.recv()
>>>>>>> b39d339b1482a7ea76a6449fc592f65ef2a4b84d:netai_gym.py

        #while self.enable_rl_agent and not ok_flag:
        #    print("[WARNING], some users may not have a valid measurement, for qos_steering case, the qos_test is not finished before a measurement return...")
        #    self.netai_gym_api_client.send(self.last_action_list) #send the same action to netai server
        #    ok_flag, df_list = self.netai_gym_api_client.recv() 
        if self.enable_rl_agent and not ok_flag:
            print("[WARNING], some users may not have a valid measurement, for qos_steering case, the qos_test is not finished before a measurement return...")
        df_phy_lte_max_rate = df_list[0]
        df_phy_wifi_max_rate = df_list[1]
        df_load = df_list[2]
        df_rate = df_list[3]
        df_qos_rate = df_list[4]
        df_owd = df_list[5]
        df_split_ratio = df_list[6]
        df_ap_id = df_list[7]
        df_phy_lte_slice_id = df_list[8]
        df_phy_lte_rb_usage = df_list[9]
        df_delay_violation = df_list[10]
        #print(df_phy_lte_slice_id)
        #print(df_phy_lte_rb_usage)
        #print(df_delay_violation)

        observation = self.use_case_helper.prepare_observation(df_list)

        print("step function at time:" + str(df_load["end_ts"][0]))
        # print(observation)

        self.normalize_obs.update(observation)
        normalized_obs = (observation - self.normalize_obs.mean) / np.sqrt(self.normalize_obs.var)
        
        #Get reward
        rewards = self.use_case_helper.prepare_reward(df_list)
        
        return normalized_obs, rewards, df_owd, observation, terminate_flag

    def step(self, actions):
        '''
        1.) Get action lists from RL agent and send to netai server
        2.) Get measurements from gamsim and normalize obs and reward
        3.) Check if it is the last step in the episode
        4.) return obs,reward,done,info
        '''

        #1.) Get action lists from RL agent and send to netai server
        self.send_action(actions)

        #2.) Get measurements from gamsim and normalize obs and reward
        normalized_obs, reward, df_owd, obs, terminate_flag = self.get_obs_reward()
        if terminate_flag:
            return [], 0, True,  {"df_owd": [], "obs" : [], "terminate_flag": terminate_flag}


        self.use_case_helper.wandb_log()

        #3.) Check end of Episode
        done = self.current_step >= self.max_steps

        self.current_step += 1

        # print("Episdoe", self.current_ep ,"step", self.current_step, "reward", reward, "Done", done)

        if done:
            if self.first_episode:
                self.first_episode = False

        #4.) return observation, reward, done, info
        return normalized_obs.astype(np.float32), reward, done,  {"df_owd": df_owd, "obs" : obs, "terminate_flag": terminate_flag}
