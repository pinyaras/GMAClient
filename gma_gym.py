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
    from gmasim_open_api import gmasim_client
except ImportError:
    from .gmasim_open_api import gmasim_client

import math

np.set_printoptions(precision=3)

FILE_PATH = pathlib.Path(__file__).parent

NUM_FEATURES = 3
STEPS_PER_EPISODE = 100

MIN_DATA_RATE = 0
MAX_DATA_RATE = 100

MIN_DELAY_MS = 0
MAX_DELAY_MS = 500

MIN_QOS_RATE = 0.1 #we assume the min qos rate is 0.1 mbps

class GmaSimEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, id, use_case_helper, config_json, wandb):
        super().__init__()

        #Define config params
        self.use_case_helper = use_case_helper
        self.num_features = NUM_FEATURES
        self.num_users = int(config_json['gmasim_config']['num_users'])
        self.use_case = config_json['gmasim_config']['use_case']
        if self.use_case == "network_slicing":
            self.num_slices = len(config_json['gmasim_config']['slice_list'])
        self.input = config_json["rl_agent_config"]['input'] 
        self.rl_alg = config_json['rl_agent_config']['agent'] 
        self.reward_type = config_json["rl_agent_config"]["reward_type"]
        self.enable_rl_agent = config_json['enable_rl_agent'] 
        self.action_space = use_case_helper.get_action_space()

        if self.input == "flat":
            self.observation_space = spaces.Box(low=0, high=1000,
                                                shape=(self.num_users*self.num_features,), dtype=np.float32)
        elif self.input == "matrix":
            self.observation_space = spaces.Box(low=0, high=1000,
                                                shape=(self.num_features,self.num_users), dtype=np.float32)
        else:                                                
            sys.exit("[" + config_json["rl_agent_config"]['input']  + "] input type not valid.")

        self.normalize_obs = RunningMeanStd(shape=self.observation_space.shape)
        self.gmasim_client = gmasim_client(id, config_json) #initial gmasim_client
        self.max_counter = int(config_json['gmasim_config']['simulation_time_s'] * 1000/config_json['gmasim_config']['GMA']['measurement_interval_ms'])# Already checked the interval for Wi-Fi and LTE in the main file

        #self.link_type = config_json['rl_agent_config']['link_type'] 
        self.current_step = 0
        self.max_steps = STEPS_PER_EPISODE
        self.current_ep = 0
        self.first_episode = True

        self.wandb_log_info = None
        self.use_case_helper.set_wandb_log(self.wandb_log_info)
        self.wandb = wandb
        self.gmasim_client.connect()

        self.last_action_list = []

    def reset(self):
        self.counter = 0
        self.current_step = 0
        # connect to the gmasim server and and receive the first measurement
        if not self.first_episode:
            self.gmasim_client.send(self.last_action_list) #send action to GMAsim server

        ok_flag, df_list = self.gmasim_client.recv()#first measurement
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
            self.gmasim_client.send([]) #send empty action to GMAsim server
            return
        # elif self.link_type == "wifi" or self.link_type == "lte" :
        if self.rl_alg == "SingleLink" :
            self.gmasim_client.send(actions) #send empty action to GMAsim server
            return           

        action_list = self.use_case_helper.prepare_action(actions)

        self.last_action_list = action_list
        self.gmasim_client.send(action_list) #send action to GMAsim server



    #def df_load_to_dict(self, df):
    #    df['user'] = df['user'].map(lambda u: f'UE{u}_tx_rate')
    #    # Set the index to the 'user' column
    #    df = df.set_index('user')
    #    # Convert the DataFrame to a dictionary
    #    data = df['value'].to_dict()
    #    return data

    def df_to_dict(self, df, description):
        df_cp = df.copy()
        df_cp['user'] = df_cp['user'].map(lambda u: f'UE{u}_'+description)
        # Set the index to the 'user' column
        df_cp = df_cp.set_index('user')
        # Convert the DataFrame to a dictionary
        data = df_cp['value'].to_dict()
        return data

    def get_obs_reward(self):
        #receive measurement from GMAsim server
        ok_flag, df_list = self.gmasim_client.recv()

        #while self.enable_rl_agent and not ok_flag:
        #    print("[WARNING], some users may not have a valid measurement, for qos_steering case, the qos_test is not finished before a measurement return...")
        #    self.gmasim_client.send(self.last_action_list) #send the same action to GMAsim server
        #    ok_flag, df_list = self.gmasim_client.recv() 
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
        rewards, avg_delay, avg_datarate = self.get_reward(df_owd, df_load, df_rate, df_qos_rate)
        
        return normalized_obs, rewards, avg_delay, df_owd, observation

    def netowrk_util(self, throughput, delay, alpha=0.5):
        """
        Calculates a network utility function based on throughput and delay, with a specified alpha value for balancing.
        
        Args:
        - throughput: a float representing the network throughput in bits per second
        - delay: a float representing the network delay in seconds
        - alpha: a float representing the alpha value for balancing (default is 0.5)
        
        Returns:
        - a float representing the alpha-balanced metric
        """
        # Calculate the logarithm of the delay in milliseconds
        log_delay = -10
        if delay>0:
            log_delay = math.log(delay)

        # Calculate the logarithm of the throughput in mb per second
        log_throughput = -10
        if throughput>0:
            log_throughput = math.log(throughput)

        #print("delay:"+str(delay) +" log(owd):"+str(log_delay) + " throughput:" + str(throughput)+ " log(throughput):" + str(log_throughput))
        
        # Calculate the alpha-balanced metric
        alpha_balanced_metric = alpha * log_throughput - (1 - alpha) * log_delay

        alpha_balanced_metric = np.clip(alpha_balanced_metric, -10, 10)
        
        return alpha_balanced_metric

    def rescale_datarate(self,data_rate):
        """
        Rescales a given reward to the range [-10, 10].
        """
        # we should not assume the max throughput is known!!
        rescaled_reward = ((data_rate - MIN_DATA_RATE) / (MAX_DATA_RATE - MIN_DATA_RATE)) * 20 - 10
        return rescaled_reward

    def calculate_wifi_qos_user_num(self, qos_rate):
        #print(qos_rate)
        reward = np.sum(qos_rate>MIN_QOS_RATE)
        return reward

    def calculate_delay_diff(self, df_owd):

        # can you add a check what if Wi-Fi or LTE link does not have measurement....
        
        #print(qos_rate)
        df_pivot = df_owd.pivot_table(index="user", columns="cid", values="value", aggfunc="first")[["Wi-Fi", "LTE"]]
        # Rename the columns to "wi-fi" and "lte"
        df_pivot.columns = ["wi-fi", "lte"]
        # Compute the delay difference between 'Wi-Fi' and 'LTE' for each user
        delay_diffs = df_pivot['wi-fi'].subtract(df_pivot['lte'], axis=0)
        abs_delay_diffs = delay_diffs.abs()
        # print(abs_delay_diffs)
        local_reward = 1/abs_delay_diffs*100
        reward = abs_delay_diffs.mean()
        return local_reward

    def get_reward(self, df_owd, df_load, df_rate, df_qos_rate):

        #Convert dataframe of Txrate state to python dict
        dict_rate = self.df_to_dict(df_rate, 'rate')
        dict_rate["sum_rate"] = df_rate[:]["value"].sum()

        df_qos_rate_all = df_qos_rate[df_qos_rate['cid'] == 'All'].reset_index(drop=True)
        df_qos_rate_wifi = df_qos_rate[df_qos_rate['cid'] == 'Wi-Fi'].reset_index(drop=True)
        dict_qos_rate_all = self.df_to_dict(df_qos_rate_all, 'qos_rate')
        dict_qos_rate_all["sum_qos_rate"] = df_qos_rate_all[:]["value"].sum()

        dict_qos_rate_wifi = self.df_to_dict(df_qos_rate_wifi, 'wifi_qos_rate')
        dict_qos_rate_wifi["sum_wifi_qos_rate"] = df_qos_rate_wifi[:]["value"].sum()

        df_owd_fill = df_owd[df_owd['cid'] == 'All'].reset_index(drop=True)

        df_owd_fill = df_owd_fill[["user", "value"]].copy()
        df_owd_fill["value"] = df_owd_fill["value"].replace(0, 1)#change 0 delay to 1 for plotting
        df_owd_fill.index = df_owd_fill['user']
        df_owd_fill = df_owd_fill.reindex(np.arange(0, self.num_users)).fillna(df_owd_fill["value"].max())#fill empty measurement with max delay
        df_owd_fill = df_owd_fill[["value"]].reset_index()
        dict_owd = self.df_to_dict(df_owd_fill, 'owd')

        df_dict = self.df_to_dict(df_load, 'tx_rate')
        df_dict["sum_tx_rate"] = df_load[:]["value"].sum()

        # _ = self.calculate_delay_diff(df_owd)

        avg_delay = df_owd["value"].mean()
        max_delay = df_owd["value"].max()

        # Pivot the DataFrame to extract "Wi-Fi" and "LTE" values
        # df_pivot = df_owd.pivot_table(index="user", columns="cid", values="value", aggfunc="first")[["Wi-Fi", "LTE"]]

        # Rename the columns to "wi-fi" and "lte"
        # df_pivot.columns = ["wi-fi", "lte"]

        # Sort the index in ascending order
        # df_pivot.sort_index(inplace=True)

        #check reward type, TODO: add reward combination of delay and throughput from network util function
        if self.reward_type =="delay":
            reward = self.delay_to_scale(avg_delay)
        elif self.reward_type =="throughput":
            reward = self.rescale_datarate(df_rate[:]["value"].mean())
        elif self.reward_type == "utility":
            reward = self.netowrk_util(df_rate[:]["value"].mean(), avg_delay)
        elif self.reward_type == "wifi_qos_user_num":
            reward = self.calculate_wifi_qos_user_num(df_qos_rate_wifi[:]["value"])
        elif self.reward_type == "delay_diff":
            # reward = self.delay_to_scale(self.calculate_delay_diff(df_owd))
            reward = self.calculate_delay_diff(df_owd)

        else:
            print("reward type not supported yet")

        #self.wandb.log(df_dict)

        #self.wandb.log({"step": self.current_step, "reward": reward, "avg_delay": avg_delay, "max_delay": max_delay})
        if not self.wandb_log_info:
            self.wandb_log_info = df_dict
        else:
            self.wandb_log_info.update(df_dict)
        self.wandb_log_info.update(dict_rate)
        self.wandb_log_info.update(dict_qos_rate_all)
        self.wandb_log_info.update(dict_qos_rate_wifi)
        self.wandb_log_info.update(dict_owd)
        self.wandb_log_info.update({"step": self.current_step, "reward": reward, "avg_delay": avg_delay, "max_delay": max_delay})

        return reward, avg_delay, df_rate[:]["value"].mean()

    #I don't like this function...
    def delay_to_scale(self, data):
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action_space: (gym.spaces.box.Box)
        :param action: (np.ndarray)
        :return: (np.ndarray)
        """
        # low, high = 0,220
        # return -10*(2.0 * ((data - low) / (high - low)) - 1.0)
        low, high = MIN_DELAY_MS, MAX_DELAY_MS
        norm = np.clip(data, low, high)

        norm = ((data - low) / (high - low)) * -20 + 10
        # norm = (-1*np.log(norm) + 3) * 2.5
        #norm = np.clip(norm, -10, 20)

        return norm


    def step(self, actions):
        '''
        1.) Get action lists from RL agent and send to gmasim server
        2.) Get measurements from gamsim and normalize obs and reward
        3.) Check if it is the last step in the episode
        4.) return obs,reward,done,info
        '''

        #1.) Get action lists from RL agent and send to gmasim server
        self.send_action(actions)

        #2.) Get measurements from gamsim and normalize obs and reward
        normalized_obs, reward, avg_delay, df_owd, obs = self.get_obs_reward()

        # send info to wandb
        self.wandb.log(self.wandb_log_info)
        self.wandb_log_info = None

        #3.) Check end of Episode
        done = self.current_step >= self.max_steps

        self.current_step += 1

        # print("Episdoe", self.current_ep ,"step", self.current_step, "reward", reward, "Done", done)

        if done:
            if self.first_episode:
                self.first_episode = False

        #4.) return observation, reward, done, info
        return normalized_obs.astype(np.float32), reward, done, {"df_owd": df_owd, "obs" : obs}
