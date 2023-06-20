try:
    # Try to import from the same directory
    from use_case_base_helper import use_case_base_helper
except ImportError:
    #Handle when import from custom RL
    from GMAClient.use_case_base_helper import use_case_base_helper

import sys
from gym import spaces
import numpy as np
import pandas as pd
import math

MIN_DATA_RATE = 0
MAX_DATA_RATE = 100

MIN_DELAY_MS = 0
MAX_DELAY_MS = 500

def single_link_policy(env, config_json):

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

        #action = env.action_space.sample()

        link_type = config_json['rl_agent_config']['agent']
        user_number = config_json['gmasim_config']['num_users']
        action_list = []

        if link_type == "Wi-Fi":
            wifi_ratio = 32
            lte_ratio = 0

        elif link_type == "LTE":
            wifi_ratio = 0
            lte_ratio = 32

        for user_id in range(user_number):

            action_list.append({"cid":"Wi-Fi","user":int(user_id),"value":int(wifi_ratio)})#config wifi ratio for user: user_id
            action_list.append({"cid":"LTE","user":int(user_id),"value":int(lte_ratio)})#config lte ratio for user: user_id

        # apply the action
        obs, reward, done, info = env.step(action_list)

class nqos_split_helper(use_case_base_helper):
    def __init__(self, wandb):
        self.use_case = "nqos_split"
        self.action_max_value = 32
        super().__init__(wandb)

    def get_action_space(self): 
        if (self.use_case == self.config_json['gmasim_config']['use_case'] and self.config_json['rl_agent_config']['agent'] != "custom" and "GMA" ):
            return spaces.Box(low=0, high=1,
                                            shape=(int(self.config_json['gmasim_config']['num_users']),), dtype=np.float32)
        #This is for CleanRL custom
        elif (self.use_case == self.config_json['gmasim_config']['use_case'] and self.config_json['rl_agent_config']['agent'] == "custom"):
            #Define your own action space to match with output of actor network
            return spaces.Box(low=0, high=1,
                                            shape=(1,), dtype=np.float32)
        else:
            sys.exit("[ERROR] wrong use case or RL agent.")

    #consistent with the prepare_observation function.
    def get_num_of_observation_features(self):
        return 3
    
    def prepare_observation(self, df_list):

        df_phy_lte_max_rate = df_list[0]
        df_phy_wifi_max_rate = df_list[1]
        df_load = df_list[2]
        df_rate = df_list[3]
        df_split_ratio = df_list[6]
        df_ap_id = df_list[7]

        wifi_util, lte_util_0 = self.process_util(df_rate, df_load, df_phy_wifi_max_rate, df_phy_lte_max_rate, df_ap_id)

        dict_wifi_split_ratio = self.df_split_ratio_to_dict(df_split_ratio, "Wi-Fi")

        #print("Wi-Fi Split Ratio:" + str(dict_wifi_split_ratio))
        if not self.wandb_log_info:
            self.wandb_log_info = dict_wifi_split_ratio
        else:
            self.wandb_log_info.update(dict_wifi_split_ratio)


        #check if the data frame is empty
        if len(df_phy_wifi_max_rate)> 0:
            dict_phy_wifi = self.df_wifi_to_dict(df_phy_wifi_max_rate, "Max-Wi-Fi")
            if not self.wandb_log_info:
                self.wandb_log_info = dict_phy_wifi
            else:
                self.wandb_log_info.update(dict_phy_wifi)
        
        if len(df_phy_lte_max_rate)> 0:
            dict_phy_lte = self.df_lte_to_dict(df_phy_lte_max_rate, "Max-LTE")
            if not self.wandb_log_info:
                self.wandb_log_info = dict_phy_lte
            else:
                self.wandb_log_info.update(dict_phy_lte)
        
        #use 3 features
        emptyFeatureArray = np.empty([self.config_json['gmasim_config']['num_users'],], dtype=int)
        emptyFeatureArray.fill(-1)
        observation = []


        #check if there are mepty features
        if len(df_phy_lte_max_rate)> 0:
            # observation = np.concatenate([observation, df_phy_lte_max_rate[:]["value"]])
            phy_lte_max_rate = df_phy_lte_max_rate[:]["value"]

        else:
            # observation = np.concatenate([observation, emptyFeatureArray])
            phy_lte_max_rate = emptyFeatureArray
        
        if len(df_phy_wifi_max_rate)> 0:
            # observation = np.concatenate([observation, df_phy_wifi_max_rate[:]["value"]])
            phy_wifi_max_rate = df_phy_wifi_max_rate[:]["value"]

        else:
            # observation = np.concatenate([observation, emptyFeatureArray])
            phy_wifi_max_rate = emptyFeatureArray

        df_rate = df_rate[df_rate['cid'] == 'All'].reset_index(drop=True) #keep the flow rate.

        # print(df_rate)
        # print(df_rate.shape)


        if len(df_rate)> 0:
            # observation = np.concatenate([observation, df_rate[:]["value"]])
            phy_df_rate = df_rate[:]["value"]

        else:
            # observation = np.concatenate([observation, emptyFeatureArray])
            phy_df_rate = emptyFeatureArray

        # observation = np.ones((3, 4))
        if self.config_json["rl_agent_config"]['input'] == "flat":
            observation = np.concatenate([phy_lte_max_rate, phy_wifi_max_rate, phy_df_rate])
            if(np.min(observation) < 0):
                print("[WARNING] some feature returns empty measurement, e.g., -1")
        elif self.config_json["rl_agent_config"]['input'] == "matrix":
            observation = np.vstack([phy_lte_max_rate, phy_wifi_max_rate, phy_df_rate])
            if (observation < 0).any():
                print("[WARNING] some feature returns empty measurement, e.g., -1")
        else:
            print("Please specify the input format to flat or matrix")
        return observation

    def prepare_action(self, actions):

        if self.config_json['rl_agent_config']['agent']  != "custom": 
            # Subtract 1 from the actions array
            subtracted_actions = 1- actions 
            # print(subtracted_actions)

            # Stack the subtracted and original actions arrays
            stacked_actions = np.vstack((actions, subtracted_actions))

            # Scale the subtracted actions to the range [0, self.action_max_value]
            scaled_stacked_actions= np.interp(stacked_actions, (0, 1), (0, self.action_max_value))
        #RL action for CleanRL
        else:
            opposite_actions = -1* actions
            # print(actions)
            # print(opposite_actions)
            # Stack the subtracted and original actions arrays
            stacked_actions = np.vstack((actions, opposite_actions))

            # Scale the subtracted actions to the range [0, self.action_max_value]
            scaled_stacked_actions= np.interp(stacked_actions, (-1, 1), (0, self.action_max_value))


        # Round the scaled subtracted actions to integers
        rounded_scaled_stacked_actions = np.round(scaled_stacked_actions).astype(int)

        print("action --> " + str(rounded_scaled_stacked_actions))
        action_list = []

        for user_id in range(self.config_json['gmasim_config']['num_users']):
            #wifi_ratio + lte_ratio = step size == self.action_max_value
            # wifi_ratio = 14 #place holder
            # lte_ratio = 18 #place holder
            action_list.append({"cid":"Wi-Fi","user":int(user_id),"value":int(rounded_scaled_stacked_actions[0][user_id])})#config wifi ratio for user: user_id
            action_list.append({"cid":"LTE","user":int(user_id),"value":int(rounded_scaled_stacked_actions[1][user_id])})#config lte ratio for user: user_id

            #wifistr = f'UE%d_Wi-Fi_ACTION' % (user_id)
            #ltestr = f'UE%d_LTE_ACTION' % (user_id)
            #df_dict = {wifistr:rounded_scaled_stacked_actions[0][user_id], ltestr:rounded_scaled_stacked_actions[1][user_id]}

            #if not self.wandb_log_info:
            #    self.wandb_log_info = df_dict
            #else:
            #    self.wandb_log_info.update(df_dict)
        return action_list

    def process_util(self, df_rate, df_load, df_phy_wifi_max_rate, df_phy_lte_max_rate, df_ap_id):

        wifi_list, lte_list = self.sta_count(df_ap_id)

        est_util_list = []

        df_wifi_rate = df_rate[df_rate['cid'] == 'Wi-Fi'].reset_index(drop=True) #keep the Wi-Fi rate.
        df_lte_rate = df_rate[df_rate['cid'] == 'LTE'].reset_index(drop=True) #keep the LTE rate.

        #Handle when there is traffic over LTE link        
        if int(self.config_json['gmasim_config']['num_users']) != len(lte_list):
            lte_list = list(range(0, int(self.config_json['gmasim_config']['num_users'])))

            missing_users = list(set(lte_list) - set(df_lte_rate['user']))
            if len(missing_users) > 0:
                missing_rows = pd.DataFrame({'user': missing_users, 'value': 0.1})
                df_lte_rate = pd.concat([df_lte_rate, missing_rows], ignore_index=True)
                # print("new rate lte",df_lte_rate)

        df_wifi_rate['value'] = df_wifi_rate['value'].replace(0, 0.1)
        df_lte_rate['value'] = df_lte_rate['value'].replace(0, 0.1)

        df_load['value'] = df_load['value'].replace(0, 0.1)

        for wifi_sta in wifi_list:
            # print(wifi_sta)
            if df_wifi_rate.empty or len(df_phy_wifi_max_rate)==0:
                est_util = [0.0, 0]
            else:
                est_util = self.estimate_util(wifi_sta, df_phy_wifi_max_rate, df_wifi_rate, df_load)
            est_util_list.append(est_util)

        if df_lte_rate.empty or len(df_phy_lte_max_rate)==0:
            # print("The DataFrame df_lte_rate is empty.")
            est_util_cell0 = [0.0, 0]
        else:
            # print("The DataFrame df_lte_rate is not empty.")
            est_util_cell0 = self.estimate_util(lte_list, df_phy_lte_max_rate, df_lte_rate, df_load)

        dict_wifi_rate = self.df_to_dict(df_wifi_rate, 'wifi-rate')
        dict_lte_rate = self.df_to_dict(df_lte_rate, 'lte-rate')

        if not self.wandb_log_info:
            self.wandb_log_info = dict_wifi_rate
        else:
            self.wandb_log_info.update(dict_wifi_rate)
        self.wandb_log_info.update(dict_lte_rate)
        self.wandb_log_info.update(dict_lte_rate)

        self.wandb_log_info.update({
                                    "AP0_util_rate": est_util_list[0][0] ,"AP1_util_rate": est_util_list[1][0],
                                    "AP0_num_user": est_util_list[0][1] ,"AP1_num_user": est_util_list[1][1],
                                    "BS0_num_user": est_util_cell0[1] ,"BS0_util_rate": est_util_cell0[0]
                                     })

        return est_util_list, est_util_cell0

    def sta_count(self,df ):
        wifi_df = df.loc[df['cid'] == 'Wi-Fi']
        wifi_df['value'] = wifi_df['value'].astype(int)

        wifi_df = wifi_df.groupby('value')['user'].agg(self.user_list).reset_index()
        wifi_df.columns = ['id', 'user_list']

        ap_list = list(range(2))

        unique_values = set(wifi_df['id'])

        if set(unique_values) == set(ap_list):
            wifi_list = wifi_df["user_list"].tolist()
        else:
            wifi_df_copy = wifi_df.copy()

            for ap_value in ap_list:
                if ap_value not in unique_values:
                    new_row = pd.DataFrame({'id': [ap_value], 'user_list': [[]]})
                    wifi_df_copy = pd.concat([wifi_df_copy, new_row], ignore_index=True)
            wifi_list = wifi_df_copy["user_list"].tolist()

        lte_df = df.loc[df['cid'] == 'LTE']
        lte_list = lte_df['user'].tolist()

        return wifi_list,lte_list

    def user_list(self, x):
        return list(x.unique())

    def estimate_util(self, user_list, max_rate_df, rate_df, load_df):
        """
        Estimates the utilization of a Wi-Fi network based on traffic arrival and throughputs.

        Args:
            user_list (list): A list of users connected to the access point.
            max_rate_df (pd.DataFrame): A dataframe containing the maximum rate of each user in the network.
            rate_df (pd.DataFrame): A dataframe containing the delivery rate for each user in the network.
            load_df (pd.DataFrame): A dataframe containing the traffic arrival rate for each user in the network.

        Returns:
            tuple: A tuple containing the estimated utilization based on traffic arrival rate and delivery rate, 
                and the number of users per access point.
        """

        # Input validation
        required_cols = ["user", "value"]
        assert all(col in max_rate_df.columns for col in required_cols), "max_rate_df is missing required columns"
        assert all(col in rate_df.columns for col in required_cols), "rate_df is missing required columns"
        assert all(col in load_df.columns for col in required_cols), "load_df is missing required columns"
        assert isinstance(user_list, list), "user_list must be a list"
        # print(user_list)
        # print(max_rate_df)
        # print(rate_df)

        # Subset dataframes for user list
        #selected_users = rate_df['user'].iloc[user_list].tolist()

        max_rate_subset = max_rate_df[max_rate_df["user"].isin(user_list)]

        rate_subset = rate_df[rate_df["user"].isin(user_list)]
        # load_subset = load_df[load_df["user"].isin(user_list)]


        # Calculate delivery rate and traffic arrival rate
        delivery_rate = rate_subset["value"].sum()
        # traffic_arrival = load_subset["value"].sum()
        # print("max_rate_subset", max_rate_subset["value"])
        # print("rate_subset",rate_subset["value"])

        # Calculate maximum capacity
        num_users = max(len(user_list), 0.01)
        max_capacity = max_rate_subset["value"].sum() / num_users 
        weighted_sum = np.sum(max_rate_subset["value"].values * rate_subset["value"].values)
        weighted_max_capacity = weighted_sum / delivery_rate

    
        # Calculate estimated utilization
        # est_util_load = traffic_arrival / max_capacity

        est_util_rate = delivery_rate / weighted_max_capacity
        # print("est_util_rate:", est_util_rate)

        return est_util_rate, num_users

    def df_to_dict(self, df, description):
        df_cp = df.copy()
        df_cp['user'] = df_cp['user'].map(lambda u: f'UE{u}_'+description)
        # Set the index to the 'user' column
        df_cp = df_cp.set_index('user')
        # Convert the DataFrame to a dictionary
        data = df_cp['value'].to_dict()
        return data

    def df_lte_to_dict(self, df, description):
        df_cp = df.copy()
        df_cp['user'] = df_cp['user'].map(lambda u: f'UE{u}_'+description)
        # Set the index to the 'user' column
        df_cp = df_cp.set_index('user')
        # Convert the DataFrame to a dictionary
        data = df_cp['value'].to_dict()
        data["LTE_avg_rate"] = df_cp[:]['value'].mean()
        data["LTE_total"] = df_cp['value'].sum()
        return data

    def df_split_ratio_to_dict(self, df, cid):
        df_cp = df.copy()
        df_cp = df_cp[df_cp['cid'] == cid].reset_index(drop=True)
        df_cp['user'] = df_cp['user'].map(lambda u: f'UE{u}_{cid}_TSU')
        # Set the index to the 'user' column
        df_cp = df_cp.set_index('user')
        # Convert the DataFrame to a dictionary
        data = df_cp['value'].to_dict()
        return data

    def df_wifi_to_dict(self, df, description):
        df_cp = df.copy()
        df_cp['user'] = df_cp['user'].map(lambda u: f'UE{u}_'+description)
        # Set the index to the 'user' column
        df_cp = df_cp.set_index('user')
        # Convert the DataFrame to a dictionary
        data = df_cp['value'].to_dict()
        data["WiFI_avg_rate"] = df_cp['value'].mean()
        data["WiFI_total"] = df_cp['value'].sum()
        return data

    def prepare_reward(self, df_list):

        df_load = df_list[2]
        df_rate = df_list[3]
        df_qos_rate = df_list[4]
        df_owd = df_list[5]
        
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
        df_owd_fill = df_owd_fill.reindex(np.arange(0, self.config_json['gmasim_config']['num_users'])).fillna(df_owd_fill["value"].max())#fill empty measurement with max delay
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
        reward = 0
        if self.config_json["rl_agent_config"]["reward_type"] =="delay":
            reward = self.delay_to_scale(avg_delay)
        elif self.config_json["rl_agent_config"]["reward_type"] =="throughput":
            reward = self.rescale_datarate(df_rate[:]["value"].mean())
        elif self.config_json["rl_agent_config"]["reward_type"] == "utility":
            reward = self.netowrk_util(df_rate[:]["value"].mean(), avg_delay)
        elif self.config_json["rl_agent_config"]["reward_type"] == "delay_diff":
            # reward = self.delay_to_scale(self.calculate_delay_diff(df_owd))
            reward = self.calculate_delay_diff(df_owd)
        else:
            sys.exit("[ERROR] reward type not supported yet")

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
        self.wandb_log_info.update({"reward": reward, "avg_delay": avg_delay, "max_delay": max_delay})

        return reward

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