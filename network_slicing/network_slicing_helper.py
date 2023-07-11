from use_case_base_helper import use_case_base_helper
import sys
from gym import spaces
import numpy as np
import pandas as pd

def get_rbg_size (bandwidth):
    # PF type 0 allocation RBG
    PfType0AllocationRbg = [10,26,63,110]      # see table 7.1.6.1-1 of 36.213

    for i in range(len(PfType0AllocationRbg)):
        if (bandwidth < PfType0AllocationRbg[i]):
            return (i + 1)
    return (-1)

class network_slicing_helper(use_case_base_helper):
    def __init__(self, wandb):
        self.use_case = "network_slicing"
        super().__init__(wandb)

    def get_action_space(self):
        if (self.use_case == self.config_json['gmasim_config']['use_case']):
            return spaces.Box(low=0, high=1, shape=(len(self.config_json['gmasim_config']['slice_list']),), dtype=np.float32)
        else:
            sys.exit("[ERROR] wrong use case or RL agent.")
    
    #consistent with the prepare_observation function.
    def get_num_of_observation_features(self):

        return 5 * len(self.config_json['gmasim_config']['slice_list'])

    # def get_observation_space



    def df_list_to_observation(self, df_list):
        # Extract necessary dataframes from the list
        df_phy_lte_max_rate = df_list[0]
        df_load = df_list[2]
        df_rate = df_list[3]
        df_phy_lte_rb_usage = df_list[9]
        df_delay_violation = df_list[10]
        df_phy_lte_slice_id = df_list[8]
        df_owd = df_list[5]  # assuming df_owd is the 6th dataframe in df_list

        # Create a mapping of user to slice ID
        user_to_slice_id = np.zeros(len(df_phy_lte_slice_id))
        df_phy_lte_slice_id = df_phy_lte_slice_id.reset_index()  # make sure indexes pair with number of rows
        for index, row in df_phy_lte_slice_id.iterrows():
            user_to_slice_id[row['user']] = int(row['value'])

        # Assign slice_id to each dataframe using user_to_slice_id mapping
        for df in [df_phy_lte_max_rate, df_load, df_rate, df_phy_lte_rb_usage, df_delay_violation, df_owd]:
            df['slice_id'] = user_to_slice_id[df['user']]

        # Calculate per slice metrics
        per_slice_rate = df_rate.groupby('slice_id')['value'].sum()
        per_slice_load = df_load.groupby('slice_id')['value'].sum()
        per_slice_achieved = per_slice_rate/per_slice_load
        per_slice_rb_usage = df_phy_lte_rb_usage.groupby('slice_id')['value'].sum() / 100
        per_slice_delay_violation_rate = df_delay_violation.groupby('slice_id')['value'].mean() / 100

        delay_threshold = self.config_json['gmasim_config']['qos_requirement']['delay_bound_ms']
        
        per_slice_max_delay = df_owd.groupby('slice_id')['value'].max() / delay_threshold # calculate per-slice max delay
        per_slice_mean_delay = df_owd.groupby('slice_id')['value'].mean() / delay_threshold  # calculate per-slice mean delay
        # breakpoint()
        # Construct final dataframe
        df_final = pd.DataFrame({
            'per_slice_achieved': per_slice_achieved,
            'per_slice_rb_usage': per_slice_rb_usage,
            'per_slice_delay_violation_rate': per_slice_delay_violation_rate,
            'per_slice_max_delay': per_slice_max_delay,
            'per_slice_mean_delay': per_slice_mean_delay
        })

        # Convert the final dataframe to numpy array
        result = df_final.values.flatten()

        return result


    
    def prepare_observation(self, df_list):

        '''
        We define the observation per_slice wise, where for each slice, we collect:
        {slice_rate, slice_load, slice_rb_usage, delay_violation_rate, max_delay, mean_delay}
        '''

        observation = self.df_list_to_observation(df_list)
        # breakpoint()
        # df_phy_lte_max_rate = df_list[0]
        # df_phy_wifi_max_rate = df_list[1]
        # df_load = df_list[2]
        # df_rate = df_list[3]
        # df_phy_lte_slice_id = df_list[8]
        # df_phy_lte_rb_usage = df_list[9]
        # df_delay_violation = df_list[10]

        # #use 3 features
        # emptyFeatureArray = np.empty([self.config_json['gmasim_config']['num_users'],], dtype=int)
        # emptyFeatureArray.fill(-1)
        # observation = []


        # #check if there are mepty features
        # if len(df_phy_lte_max_rate)> 0:
        #     # observation = np.concatenate([observation, df_phy_lte_max_rate[:]["value"]])
        #     phy_lte_max_rate = df_phy_lte_max_rate[:]["value"]

        # else:
        #     # observation = np.concatenate([observation, emptyFeatureArray])
        #     phy_lte_max_rate = emptyFeatureArray
        
        # if len(df_phy_wifi_max_rate)> 0:
        #     # observation = np.concatenate([observation, df_phy_wifi_max_rate[:]["value"]])
        #     phy_wifi_max_rate = df_phy_wifi_max_rate[:]["value"]

        # else:
        #     # observation = np.concatenate([observation, emptyFeatureArray])
        #     phy_wifi_max_rate = emptyFeatureArray

        # df_rate = df_rate[df_rate['cid'] == 'All'].reset_index(drop=True) #keep the flow rate.

        # # print(df_rate)
        # # print(df_rate.shape)


        # if len(df_rate)> 0:
        #     # observation = np.concatenate([observation, df_rate[:]["value"]])
        #     phy_df_rate = df_rate[:]["value"]

        # else:
        #     # observation = np.concatenate([observation, emptyFeatureArray])
        #     phy_df_rate = emptyFeatureArray

        # # observation = np.ones((3, 4))
        # if self.config_json["rl_agent_config"]['input'] == "flat":
        #     observation = np.concatenate([phy_lte_max_rate, phy_wifi_max_rate, phy_df_rate])
        #     if(np.min(observation) < 0):
        #         print("[WARNING] some feature returns empty measurement, e.g., -1")
        # elif self.config_json["rl_agent_config"]['input'] == "matrix":
        #     observation = np.vstack([phy_lte_max_rate, phy_wifi_max_rate, phy_df_rate])
        #     if (observation < 0).any():
        #         print("[WARNING] some feature returns empty measurement, e.g., -1")
        # else:
        #     print("Please specify the input format to flat or matrix")
        return observation

    def prepare_action(self, actions):
        
        breakpoint()
        rbg_size = get_rbg_size(self.config_json['gmasim_config']['LTE']['resource_block_num'])
        rbg_num = self.config_json['gmasim_config']['LTE']['resource_block_num']/rbg_size
        scaled_actions= np.interp(actions, (0, 1), (0, rbg_num/len(self.config_json['gmasim_config']['slice_list'])))
        #scaled_actions= np.interp(actions, (0, 1), (0, rbg_num))

        # Round the scaled subtracted actions to integers
        rounded_scaled_actions = np.round(scaled_actions).astype(int)

        print("action --> " + str(rounded_scaled_actions))
        action_list = []

        for slice_id in range(len(self.config_json['gmasim_config']['slice_list'])):
            action_list.append({"slice":int(slice_id),"D":int(rounded_scaled_actions[slice_id]),"P":int(0),"S":int(25)})

        # the unit of the action is resource block group number, not resource block!!!
        # please make sure the sum of the dedicated ("D") and priorititized ("P") resouce block group # is smaller than total resource block group number.
        return action_list

    def prepare_reward(self, df_list):
        '''
        This is the place to compute the reward for network slicing usecase,
        from the NS3-simulator, we need to use: {load, rate, rb_usage_rate, delay_violation}
        to compute our reward. 
        The reward is defined as following, for slice i:
        R_i = rate_i/load_i - gamma * delay_vio_rate - delta * usage_rate
        This means if the transfer rate meets the load, we give the system a positive reward,
        we will also punish the delay violation and the usage of the resource for each slice. 
        '''



        delta = .25
        gamma = 2
        observation = self.df_list_to_observation(df_list)
        # breakpoint()

        df_phy_lte_max_rate = df_list[0]
        df_phy_wifi_max_rate = df_list[1]
        df_load = df_list[2]
        df_rate = df_list[3]
        df_qos_rate = df_list[4]
        df_owd = df_list[5]
        df_phy_lte_slice_id = df_list[8]
        df_phy_lte_rb_usage = df_list[9]
        df_delay_violation = df_list[10]

        # breakpoint()

        #print (df_phy_lte_slice_id)
        #print (df_rate)
        user_to_slice_id = np.zeros(len(df_phy_lte_slice_id))
        df_phy_lte_slice_id = df_phy_lte_slice_id.reset_index()  # make sure indexes pair with number of rows
        for index, row in df_phy_lte_slice_id.iterrows():
            user_to_slice_id[row['user']] = int(row['value'])
        #print (user_to_slice_id)

        df_load['slice_id']=user_to_slice_id[df_load['user']]
        df_load['slice_value']= df_load.groupby(['slice_id'])['value'].transform('sum')

        df_slice_load = df_load.drop_duplicates(subset=['slice_id'])
        df_slice_load = df_slice_load.drop(columns=['user'])
        df_slice_load = df_slice_load.drop(columns=['value'])

        np_total_load = df_slice_load["slice_value"].sum()

        #print (df_load)
        print (df_slice_load)

        df_phy_lte_max_rate['slice_id']=user_to_slice_id[df_phy_lte_max_rate['user']]
        df_phy_lte_max_rate['slice_value']= df_phy_lte_max_rate.groupby(['slice_id'])['value'].transform('mean')

        df_slice_lte_max_rate = df_phy_lte_max_rate.drop_duplicates(subset=['slice_id'])
        df_slice_lte_max_rate = df_slice_lte_max_rate.drop(columns=['user'])
        df_slice_lte_max_rate = df_slice_lte_max_rate.drop(columns=['value'])

        #print (df_phy_lte_max_rate)
        print (df_slice_lte_max_rate)

        df_lte_rate = df_rate[df_rate['cid'] == 'LTE'].reset_index(drop=True) #keep the LTE rate.
        df_lte_rate['slice_id']=user_to_slice_id[df_lte_rate['user']]
        df_lte_rate['slice_value']= df_lte_rate.groupby(['slice_id'])['value'].transform('sum')

        df_lte_slice_rate = df_lte_rate.drop_duplicates(subset=['slice_id'])
        df_lte_slice_rate = df_lte_slice_rate.drop(columns=['user'])
        df_lte_slice_rate = df_lte_slice_rate.drop(columns=['value'])

        np_lte_total_rate = df_lte_slice_rate["slice_value"].sum()

        #print (df_lte_rate)
        # print (df_lte_slice_rate)

        df_lte_qos_rate = df_qos_rate[df_qos_rate['cid'] == 'LTE'].reset_index(drop=True) #keep the LTE rate.
        df_lte_qos_rate['slice_id']=user_to_slice_id[df_lte_qos_rate['user']]
        df_lte_qos_rate['slice_value']= df_lte_qos_rate.groupby(['slice_id'])['value'].transform('sum')

        df_lte_qos_slice_rate = df_lte_qos_rate.drop_duplicates(subset=['slice_id'])
        df_lte_qos_slice_rate = df_lte_qos_slice_rate.drop(columns=['user'])
        df_lte_qos_slice_rate = df_lte_qos_slice_rate.drop(columns=['value'])

        #print (df_lte_qos_rate)
        # print (df_lte_qos_slice_rate)

        df_phy_lte_rb_usage['slice_id']=user_to_slice_id[df_phy_lte_rb_usage['user']]
        df_phy_lte_rb_usage['slice_value']= df_phy_lte_rb_usage.groupby(['slice_id'])['value'].transform('sum')

        df_slice_lte_rb_usage = df_phy_lte_rb_usage.drop_duplicates(subset=['slice_id'])
        df_slice_lte_rb_usage = df_slice_lte_rb_usage.drop(columns=['user'])
        df_slice_lte_rb_usage = df_slice_lte_rb_usage.drop(columns=['value'])

        np_total_rb_usage = df_slice_lte_rb_usage["slice_value"].sum() / 100

        #print (df_phy_lte_rb_usage)
        # print (df_slice_lte_rb_usage)

        df_delay_violation['slice_id']=user_to_slice_id[df_delay_violation['user']]
        df_delay_violation['slice_value']= df_delay_violation.groupby(['slice_id'])['value'].transform('mean')

        df_slice_delay_violation = df_delay_violation.drop_duplicates(subset=['slice_id'])
        df_slice_delay_violation = df_slice_delay_violation.drop(columns=['user'])
        df_slice_delay_violation = df_slice_delay_violation.drop(columns=['value'])


        np_delay_violation = df_slice_delay_violation["slice_value"].sum() / 100
        #print (df_delay_violation)
        # print (df_slice_delay_violation)
        
        #Convert dataframe of Txrate state to python dict

        dict_slice_load = self.slice_df_to_dict(df_slice_load, 'tx_rate')
        #print(dict_slice_load)

        dict_slice_lte_max_rate = self.slice_df_to_dict(df_slice_lte_max_rate, 'lte_max_rate')
        #print(dict_slice_lte_max_rate)

        dict_lte_slice_rate = self.slice_df_to_dict(df_lte_slice_rate, 'rate')
        dict_lte_slice_rate["sum_rate"] = df_lte_slice_rate[:]["slice_value"].sum()
        #print(dict_lte_slice_rate)

        dict_lte_qos_slice_rate = self.slice_df_to_dict(df_lte_qos_slice_rate, 'qos_rate')
        dict_lte_qos_slice_rate["sum_rate"] = df_lte_qos_slice_rate[:]["slice_value"].sum()
        #print(dict_lte_qos_slice_rate)

        dict_slice_delay_violation = self.slice_df_to_dict(df_slice_delay_violation, 'delay_violation_per')
        #print(dict_slice_delay_violation)

        dict_slice_lte_rb_usage = self.slice_df_to_dict(df_slice_lte_rb_usage, 'rb_usage_per')
        #print(dict_slice_lte_rb_usage)

        df_owd_fill = df_owd[df_owd['cid'] == 'All'].reset_index(drop=True)

        df_owd_fill = df_owd_fill[["user", "value"]].copy()
        df_owd_fill["value"] = df_owd_fill["value"].replace(0, 1)#change 0 delay to 1 for plotting
        df_owd_fill.index = df_owd_fill['user']
        df_owd_fill = df_owd_fill.reindex(np.arange(0, self.config_json['gmasim_config']['num_users'])).fillna(df_owd_fill["value"].max())#fill empty measurement with max delay
        df_owd_fill = df_owd_fill[["value"]].reset_index()
        dict_owd = self.df_to_dict(df_owd_fill, 'owd')


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
        reward = np_lte_total_rate/np_total_load - delta * np_total_rb_usage - gamma * np_delay_violation
        # print("[WARNING] reward fucntion not defined yet")
        print(f"Current reward: {reward:.2f}")


        if not self.wandb_log_info:
            self.wandb_log_info = dict_slice_load
        else:
            self.wandb_log_info.update(dict_slice_load)
        self.wandb_log_info.update(dict_owd)
        self.wandb_log_info.update(dict_slice_lte_max_rate)
        self.wandb_log_info.update(dict_lte_slice_rate)
        self.wandb_log_info.update(dict_lte_qos_slice_rate)

        self.wandb_log_info.update(dict_slice_delay_violation)
        self.wandb_log_info.update(dict_slice_lte_rb_usage)
        
        self.wandb_log_info.update({"reward": reward, "avg_delay": avg_delay, "max_delay": max_delay})

        return reward

    def slice_df_to_dict(self, df, description):
        df_cp = df.copy()
        df_cp['slice_id'] = df_cp['slice_id'].map(lambda u: f'slice_{int(u)}_'+description)
        # Set the index to the 'user' column
        df_cp = df_cp.set_index('slice_id')
        # Convert the DataFrame to a dictionary
        data = df_cp['slice_value'].to_dict()
        return data