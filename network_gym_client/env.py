import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box
import pandas as pd

import pathlib
import json
import sys
from network_gym_client.northbound_interface import NorthboundInterface
from network_gym_client.envs.nqos_split.adapter import Adapter as NqosSplitAdapter
from network_gym_client.envs.qos_steer.adapter import Adapter as QosSteerAdapter
from network_gym_client.envs.network_slicing.adapter import Adapter as NetworkSlicingAdapter
np.set_printoptions(precision=3)

FILE_PATH = pathlib.Path(__file__).parent

STEPS_PER_EPISODE = 100

class Env(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, id, wandb, config_json):
        """Initilize networkgym_env class

        Args:
            id (int): the client ID number
            wandb (wandb): wandb database
            config_json (json): configuration file
        """
        super().__init__()

        if config_json['session_name'] == 'test':
            #username = wandb.api_key.split()[0].split("@")[0]
                print('***[WARNING]*** You are using the default "test" to connect to the server, which may conflict with the simulations launched by other users.')
                print('***[WARNING]*** Please change the "session_name" attribute in the common_config.json file to your assigned session name. If you do not have one, contact menglei.zhang@intel.com.')
        
        #Define config params
        if(config_json['gmasim_config']['env'] == "nqos_split"):
            self.adapter = NqosSplitAdapter(wandb)
        elif(config_json['gmasim_config']['env'] == "qos_steer"):
            self.adapter = QosSteerAdapter(wandb)
        elif(config_json['gmasim_config']['env'] == "network_slicing"):
            self.adapter = NetworkSlicingAdapter (wandb)
        else:
            sys.exit("[" + config_json['gmasim_config']['env'] + "] environment is not implemented.")

        self.adapter.set_config(config_json)

        self.enable_rl_agent = True
        if config_json['rl_agent_config']['agent']=="system_default":
            self.enable_rl_agent = False

        self.action_space = self.adapter.get_action_space()
        self.observation_space = self.adapter.get_observation_space()

        self.northbound_interface_client = NorthboundInterface(id, config_json) #initial northbound_interface_client
        #self.max_counter = int(config_json['gmasim_config']['simulation_time_s'] * 1000/config_json['gmasim_config']['GMA']['measurement_interval_ms'])# Already checked the interval for Wi-Fi and LTE in the main file

        #self.link_type = config_json['rl_agent_config']['link_type'] 
        self.current_step = 0
        self.max_steps = STEPS_PER_EPISODE
        self.current_ep = 0
        self.first_episode = True

        self.last_policy = []

        
    def reset(self, seed=None, options=None):
        """Resets the environment to an initial internal state, returning an initial observation and info.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        self.current_step = 0
        # if a new simulation starts (first episode) in the reset function, we need to connect to server
        # else a new episode of the same simulation.
            # do not need to connect, send action directly
        if self.first_episode:
            self.northbound_interface_client.connect()
        else:
            self.northbound_interface_client.send(self.last_policy) #send action to network gym server

        measurement_report = self.northbound_interface_client.recv()#first measurement
        ok_flag = measurement_report.ok_flag
        terminate_flag = measurement_report.terminate_flag
        df_list =  measurement_report.df_list

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

        observation = self.adapter.prepare_observation(df_list)

        # print(observation.shape)

        self.current_ep += 1

        return observation.astype(np.float32), {"df_owd": df_owd, "obs" : observation, "terminate_flag": terminate_flag}

    def get_obs_reward(self):
        """Call the northbound interface to receive the newotk stats, then transform to observation and reward using environment data format adatper.

        Returns:
            observation (ObsType): An element of the environment's `observation_space` as the next observation due to the agent actions.
            reward (SupportsFloat): The reward as a result of taking the action.
            df_owd (pandas.dataframe): One-way delay measurement.
            observation (ObsType): The raw network stats measurements.
            terminate_flag (Bool): A flag to indicate environment is stopped.
        """
        #receive measurement from network gym server
        measurement_report = self.northbound_interface_client.recv()
        ok_flag = measurement_report.ok_flag
        terminate_flag = measurement_report.terminate_flag
        df_list =  measurement_report.df_list
        if terminate_flag == True:
            self.first_episode = True
            self.current_ep = 0
            quit()
            #simulation already ended, connect again in the reset function to start a new one...
            return [], 0, [], [], terminate_flag
       
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

        observation = self.adapter.prepare_observation(df_list)

        print("step function at time:" + str(df_load["end_ts"][0]))
        # print(observation)

        #Get reward
        rewards = self.adapter.prepare_reward(df_list)
        
        return observation, rewards, df_owd, observation, terminate_flag

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        Get action lists from RL agent and send to network gym server
        Get measurements from gamsim and obs and reward
        Check if it is the last step in the episode
        Return obs,reward,done,info

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's `observation_space` as the next observation due to the agent actions.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). {one-way delay, raw observation, and termination flag} 
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """

        #1.) Get action from RL agent and send to network gym server
        if not self.enable_rl_agent or action.size == 0:
            #empty action
            self.northbound_interface_client.send([]) #send empty action to network gym server
        else:
            # TODO: we need to have the same action format... e.g., [0, 1]
            policy = self.adapter.prepare_policy(action)
            self.last_policy = policy
            self.northbound_interface_client.send(policy) #send network policy to network gym server

        #2.) Get measurements from gamsim and obs and reward
        observation, reward, df_owd, obs, terminate_flag = self.get_obs_reward()
        if terminate_flag:
            return [], 0, True,  {"df_owd": [], "obs" : [], "terminate_flag": terminate_flag}


        self.adapter.wandb_log()

        #3.) Check end of Episode
        done = self.current_step >= self.max_steps

        self.current_step += 1

        # print("Episdoe", self.current_ep ,"step", self.current_step, "reward", reward, "Done", done)

        if done:
            if self.first_episode:
                self.first_episode = False

        #4.) return observation, reward, done, info
        return observation.astype(np.float32), reward, done, done, {"df_owd": df_owd, "obs" : obs, "terminate_flag": terminate_flag}
