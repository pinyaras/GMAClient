import gym
import numpy as np
from gym import spaces
from gym.spaces import Box

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.running_mean_std import RunningMeanStd

import pathlib
import json
from gmasim_gym_client import gmasim_client
import math
import sys

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

    def __init__(self, id, config_json, wandb):
        super().__init__()
        # Define action and observation space
        self.num_features = NUM_FEATURES
        self.num_users = int(config_json['gmasim_config']['num_users'])
        if (config_json['gmasim_config']['use_case'] == "qos_steer"):
            #self.action_space = spaces.Box(low=0, high=1,
            #                                shape=(self.num_users,), dtype=np.uint8)
            myarray = np.empty([self.num_users,], dtype=np.int)
            myarray.fill(2)
            #print(myarray)
            self.action_space = spaces.MultiDiscrete(myarray)
            self.split_ratio_size = 1
        elif (config_json['gmasim_config']['use_case'] == "nqos_split"):
            self.action_space = spaces.Box(low=0, high=1,
                                            shape=(self.num_users,), dtype=np.float32)
            self.split_ratio_size = 32
        else:
            sys.exit("[" + config_json['gmasim_config']['use_case'] + "] use case is not implemented.")

        self.observation_space = spaces.Box(low=0, high=1000,
                                            shape=(self.num_users*self.num_features,), dtype=np.float32)
        self.normalize_obs = RunningMeanStd(shape=self.observation_space.shape)
        self.gmasim_client = gmasim_client(id, config_json) #initial gmasim_client
        self.max_counter = int(config_json['gmasim_config']['simulation_time_s'] * 1000/config_json['gmasim_config']['GMA']['measurement_interval_ms'])# Already checked the interval for Wi-Fi and LTE in the main file
        self.reward_type = config_json["rl_agent_config"]["reward_type"]
        self.enable_rl_agent = config_json['enable_rl_agent'] 
        self.current_step = 0
        self.max_steps = STEPS_PER_EPISODE
        self.current_ep = 0
        self.first_episode = True

        self.wandb_log_info = None
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

        print("Reset Function at time:" + str(df_load["end_ts"][0]))
        if self.enable_rl_agent and not ok_flag:
            sys.exit('[Error!] The first observation should always be okey!!!!')


        #use 3 features
        emptyFeatureArray = np.empty([self.num_users,], dtype=np.int)
        emptyFeatureArray.fill(-1)
        observation = []

        #check if there are mepty features
        if len(df_phy_lte_max_rate)> 0:
            observation = np.concatenate([observation, df_phy_lte_max_rate[:]["value"]])
        else:
            observation = np.concatenate([observation, emptyFeatureArray])
        
        if len(df_phy_wifi_max_rate)> 0:
            observation = np.concatenate([observation, df_phy_wifi_max_rate[:]["value"]])
        else:
            observation = np.concatenate([observation, emptyFeatureArray])

        if len(df_rate)> 0:
            observation = np.concatenate([observation, df_rate[:]["value"]])
        else:
            observation = np.concatenate([observation, emptyFeatureArray])

        if(np.min(observation) < 0):
            print("[WARNING] some feature returns empty measurement, e.g., -1")
        
        # print(observation)
        # print(observation.shape)
        self.normalize_obs.update(observation)
        normalized_obs = (observation - self.normalize_obs.mean) / np.sqrt(self.normalize_obs.var)

        self.current_ep += 1

        return normalized_obs.astype(np.float32)
        # return observation  # reward, done, info can't be included

    def send_action(self, actions):

        if actions.size == 0:
            #empty action
            self.gmasim_client.send([]) #send empty action to GMAsim server
            return
        
        # Subtract 1 from the actions array
        subtracted_actions = 1- actions 
        # print(subtracted_actions)

        # Stack the subtracted and original actions arrays
        stacked_actions = np.vstack((actions, subtracted_actions))

        # Scale the subtracted actions to the range [0, self.split_ratio_size]
        scaled_stacked_actions= np.interp(stacked_actions, (0, 1), (0, self.split_ratio_size))

        # Round the scaled subtracted actions to integers
        rounded_scaled_stacked_actions = np.round(scaled_stacked_actions).astype(int)

        print("action --> " + str(rounded_scaled_stacked_actions))
        action_list = []
        for user_id in range(self.num_users):
            #wifi_ratio + lte_ratio = step size == self.split_ratio_size
            # wifi_ratio = 14 #place holder
            # lte_ratio = 18 #place holder
            action_list.append({"cid":"Wi-Fi","user":int(user_id),"value":int(rounded_scaled_stacked_actions[0][user_id])})#config wifi ratio for user: user_id
            action_list.append({"cid":"LTE","user":int(user_id),"value":int(rounded_scaled_stacked_actions[1][user_id])})#config lte ratio for user: user_id

        self.last_action_list = action_list
        self.gmasim_client.send(action_list) #send action to GMAsim server

        for user_id in range(self.num_users):

            wifistr = f'UE%d_Wi-Fi_ACTION' % (user_id)
            ltestr = f'UE%d_LTE_ACTION' % (user_id)
            df_dict = {wifistr:rounded_scaled_stacked_actions[0][user_id], ltestr:rounded_scaled_stacked_actions[1][user_id]}

            if not self.wandb_log_info:
                self.wandb_log_info = df_dict
            else:
                self.wandb_log_info.update(df_dict)



    #def df_load_to_dict(self, df):
    #    df['user'] = df['user'].map(lambda u: f'UE{u}_tx_rate')
    #    # Set the index to the 'user' column
    #    df = df.set_index('user')
    #    # Convert the DataFrame to a dictionary
    #    data = df['value'].to_dict()
    #    return data

    def df_rate_to_dict(self, df, description):
        df['user'] = df['user'].map(lambda u: f'UE{u}_'+description)
        # Set the index to the 'user' column
        df = df.set_index('user')
        # Convert the DataFrame to a dictionary
        data = df['value'].to_dict()
        return data

    def df_split_ratio_to_dict(self, df, cid):
        df = df[df['cid'] == cid].reset_index(drop=True)
        df['user'] = df['user'].map(lambda u: f'UE{u}_{cid}_TSU')
        # Set the index to the 'user' column
        df = df.set_index('user')
        # Convert the DataFrame to a dictionary
        data = df['value'].to_dict()
        return data

    def get_obs_reward(self):
        #receive measurement from GMAsim server
        ok_flag, df_list = self.gmasim_client.recv()

        while self.enable_rl_agent and not ok_flag:
            print("[WARNING], some users may not have a valid measurement, for qos_steering case, the qos_test is not finished before a measurement return...")
            self.gmasim_client.send(self.last_action_list) #send the same action to GMAsim server
            ok_flag, df_list = self.gmasim_client.recv() 

        df_phy_lte_max_rate = df_list[0]
        df_phy_wifi_max_rate = df_list[1]
        df_load = df_list[2]
        df_rate = df_list[3]
        df_qos_rate = df_list[4]
        df_owd = df_list[5]
        df_split_ratio = df_list[6]

        print("step function at time:" + str(df_load["end_ts"][0]))
        dict_wifi_split_ratio = self.df_split_ratio_to_dict(df_split_ratio, "Wi-Fi")

        if not self.wandb_log_info:
            self.wandb_log_info = dict_wifi_split_ratio
        else:
            self.wandb_log_info.update(dict_wifi_split_ratio)

        #dict_lte_split_ratio  = self.df_split_ratio_to_dict(df_split_ratio, "LTE")
        #self.wandb_log_info.update(dict_lte_split_ratio)

        #Concat all the observations
        #observation = np.concatenate([df_phy_lte_max_rate[:]["value"], df_phy_wifi_max_rate[:]["value"], df_rate[:]["value"]])

        #check if there is empty features
        emptyFeatureArray = np.empty([self.num_users,], dtype=np.int)
        emptyFeatureArray.fill(-1)
        observation = []

        if len(df_phy_lte_max_rate)> 0:
            observation = np.concatenate([observation, df_phy_lte_max_rate[:]["value"]])
        else:
            observation = np.concatenate([observation, emptyFeatureArray])
        
        if len(df_phy_wifi_max_rate)> 0:
            observation = np.concatenate([observation, df_phy_wifi_max_rate[:]["value"]])
        else:
            observation = np.concatenate([observation, emptyFeatureArray])

        if len(df_rate)> 0:
            observation = np.concatenate([observation, df_rate[:]["value"]])
        else:
            observation = np.concatenate([observation, emptyFeatureArray])

        #print (observation)

        if(np.min(observation) < 0):
            print("[WARNING] some feature returns empty measurement, e.g., -1")

        self.normalize_obs.update(observation)
        normalized_obs = (observation - self.normalize_obs.mean) / np.sqrt(self.normalize_obs.var)
        
        #Get reward
        rewards, avg_delay, avg_datarate = self.get_reward(df_owd, df_load, df_rate, df_qos_rate)
        
        return normalized_obs, rewards, avg_delay

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
            math.log(delay)

        # Calculate the logarithm of the throughput in mb per second
        log_throughput = -10
        if throughput>0:
            math.log(throughput)

        #print("log(owd):"+str(log_delay) + " log(rate):" + str(log_throughput))
        
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

    def get_reward(self, df_owd, df_load, df_rate, df_qos_rate):

        #Convert dataframe of Txrate state to python dict
        dict_rate = self.df_rate_to_dict(df_rate, 'rate')
        dict_rate["avg_rate"] = df_rate[:]["value"].mean()

        df_qos_rate_all = df_qos_rate[df_qos_rate['cid'] == 'All'].reset_index(drop=True)
        df_qos_rate_wifi = df_qos_rate[df_qos_rate['cid'] == 'Wi-Fi'].reset_index(drop=True)
        dict_qos_rate_all = self.df_rate_to_dict(df_qos_rate_all, 'qos_rate')
        dict_qos_rate_all["avg_qos_rate"] = df_qos_rate_all[:]["value"].mean()

        dict_qos_rate_wifi = self.df_rate_to_dict(df_qos_rate_wifi, 'wifi_qos_rate')
        dict_qos_rate_wifi["avg_wifi_qos_rate"] = df_qos_rate_wifi[:]["value"].mean()


        df_dict = self.df_rate_to_dict(df_load, 'tx_rate')
        df_dict["avg_tx_rate"] = df_load[:]["value"].mean()


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
        normalized_obs, reward, avg_delay = self.get_obs_reward()

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
        return normalized_obs.astype(np.float32), reward, done, {}

# def main():

#     #Config the default GMAsim simulation setup in the json file
#     f = open(FILE_PATH / 'gma-algorithm-client-config.json')
#     config_json = json.load(f)

#     env = GmaSimEnv(0, config_json) # pass id, and configure file

#     # Number of steps you run the agent for 
#     num_steps = 50

#     obs = env.reset()

#     for step in range(num_steps):
#         # take random action, but you can also do something more intelligent
#         # action = my_intelligent_agent_fn(obs) 

#         print("step", step)
#         action = env.action_space.sample()
        
#         # apply the action
#         obs, reward, done, info = env.step(action)
        
#         # If the epsiode is up, then start another one
#         if done:
#             break
            

# if __name__ == "__main__":
    # main()