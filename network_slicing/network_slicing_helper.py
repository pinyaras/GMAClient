from use_case_base_helper import use_case_base_helper
import sys
from gym import spaces
import numpy as np

class network_slicing_helper(use_case_base_helper):
    def __init__(self):
        self.use_case = "network_slicing"
        self.config_json = None
        self.action_max_value = 5

    def get_action_space(self):
        if (self.use_case == self.config_json['gmasim_config']['use_case']):
            return spaces.Box(low=0, high=1, shape=(len(self.config_json['gmasim_config']['slice_list']),), dtype=np.float32)
        else:
            sys.exit("[ERROR] wrong use case or RL agent.")
    
    def prepare_observation(self, df_list):

        df_phy_lte_max_rate = df_list[0]
        df_phy_wifi_max_rate = df_list[1]
        df_rate = df_list[3]

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

        for slice_id in range(len(self.config_json['gmasim_config']['slice_list'])):
            action_list.append({"slice":int(slice_id),"D":int(rounded_scaled_stacked_actions[0][slice_id]),"P":int(0),"S":int(50)})

        return action_list