from network_gym.adapter import adapter
import sys
from gymnasium import spaces
import numpy as np


class qos_steer_adapter(adapter):
    def __init__(self, wandb):
        self.env = "qos_steer"
        self.action_max_value = 1
        super().__init__(wandb)

    def get_action_space (self):
        if (self.env == self.config_json['gmasim_config']['env']):
            #self.action_space = spaces.Box(low=0, high=1,
            #                                shape=(self.num_users,), dtype=np.uint8)
            myarray = np.empty([int(self.config_json['gmasim_config']['num_users']),], dtype=int)
            myarray.fill(2)
            #print(myarray)
            return spaces.MultiDiscrete(myarray)
        else:
            sys.exit("[ERROR] wrong environment or RL agent.")

    #consistent with the prepare_observation function.
    def get_num_of_observation_features(self):
        return 3
    
    def prepare_observation(self, df_list):
        
        df_phy_lte_max_rate = df_list[0]
        df_phy_wifi_max_rate = df_list[1]
        df_rate = df_list[3]
        df_split_ratio = df_list[6]

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
        if self.config_json["rl_agent_config"]["reward_type"] == "wifi_qos_user_num":
            reward = self.calculate_wifi_qos_user_num(df_qos_rate_wifi[:]["value"])
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

    def calculate_wifi_qos_user_num(self, qos_rate):
        #print(qos_rate)
        reward = np.sum(qos_rate>0.1) #we assume the min qos rate is 0.1 mbps
        return reward