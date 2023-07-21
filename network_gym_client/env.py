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

def load_config_file(env):
    #load config files
    FILE_PATH = pathlib.Path(__file__).parent
    #common_config.json is shared by all environments
    f = open(FILE_PATH / 'common_config.json')
    common_config_json = json.load(f)
    
    #load the environment dependent config file
    file_name = 'envs/' +env + '/config.json'
    f = open(FILE_PATH / file_name)

    use_case_config_json = json.load(f)
    config_json = {**common_config_json, **use_case_config_json}
    config_json['gmasim_config']['env'] = env
    return config_json

class Env(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, id, config_json):
        """Initilize networkgym_env class

        Args:
            id (int): the client ID number
            config_json (json): configuration file
        """
        super().__init__()

        if config_json['session_name'] == 'test':
            print('***[WARNING]*** You are using the default "test" to connect to the server, which may conflict with the simulations launched by other users.')
            print('***[WARNING]*** Please change the "session_name" attribute in the common_config.json file to your assigned session name.')
        
        #check if the measurement interval for all measurements are the same.
        if (config_json['gmasim_config']['GMA']['measurement_interval_ms'] + config_json['gmasim_config']['GMA']['measurement_guard_interval_ms']
            == config_json['gmasim_config']['Wi-Fi']['measurement_interval_ms'] + config_json['gmasim_config']['Wi-Fi']['measurement_guard_interval_ms']
            == config_json['gmasim_config']['LTE']['measurement_interval_ms'] + config_json['gmasim_config']['LTE']['measurement_guard_interval_ms']):
            pass
        else:
            print(config_json['gmasim_config']['GMA']['measurement_interval_ms'])
            print(config_json['gmasim_config']['GMA']['measurement_guard_interval_ms'])
            print(config_json['gmasim_config']['Wi-Fi']['measurement_interval_ms'])
            print(config_json['gmasim_config']['Wi-Fi']['measurement_guard_interval_ms'])
            print(config_json['gmasim_config']['LTE']['measurement_interval_ms'])
            print(config_json['gmasim_config']['LTE']['measurement_guard_interval_ms'])
            sys.exit('[Error!] The value of GMA, Wi-Fi, and LTE measurement_interval_ms + measurement_guard_interval_ms should be the same!')


        self.steps_per_episode = int(config_json['gmasim_config']['steps_per_episode'])
        self.episodes_per_session = int(config_json['gmasim_config']['episodes_per_session'])

        step_length = config_json['gmasim_config']['GMA']['measurement_interval_ms'] + config_json['gmasim_config']['GMA']['measurement_guard_interval_ms']
        # compute the simulation time based on setting
        config_json['gmasim_config']['simulation_time_s'] = int((config_json['gmasim_config']['app_and_measurement_start_time_ms'] + step_length * self.steps_per_episode * self.episodes_per_session)/1000)
        print("Environment duration: " + str(config_json['gmasim_config']['simulation_time_s']) + "s")
        #Define config params
        if(config_json['gmasim_config']['env'] == "nqos_split"):
            self.adapter = NqosSplitAdapter()
        elif(config_json['gmasim_config']['env'] == "qos_steer"):
            self.adapter = QosSteerAdapter()
        elif(config_json['gmasim_config']['env'] == "network_slicing"):
            self.adapter = NetworkSlicingAdapter ()
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
        self.current_step = 1
        self.current_ep += 1

        # if a new simulation starts (first episode) in the reset function, we need to connect to server
        # else a new episode of the same simulation.
            # do not need to connect, send action directly
        if self.first_episode:
            self.northbound_interface_client.connect()
        else:
            self.northbound_interface_client.send(self.last_policy) #send action to network gym server

        measurement_report = self.northbound_interface_client.recv()#first measurement
        ok_flag = measurement_report.ok_flag
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

        print("reset() at time:" + str(df_load["end_ts"][0]) + "ms, episode:" + str(self.current_ep) + ", step:" + str(self.current_step))

        if self.enable_rl_agent and not ok_flag:
            print("[WARNING], some users may not have a valid measurement, for qos_steering case, the qos_test is not finished before a measurement return...")

        observation = self.adapter.prepare_observation(df_list)

        # print(observation.shape)


        return observation.astype(np.float32), {"network_stats": df_list}

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
        self.current_step += 1

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
        measurement_report = self.northbound_interface_client.recv()
        ok_flag = measurement_report.ok_flag
        df_list =  measurement_report.df_list
       
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

        print("step() at time:" + str(df_load["end_ts"][0]) + "ms, episode:" + str(self.current_ep) + ", step:" + str(self.current_step))

        observation = self.adapter.prepare_observation(df_list)

        # print(observation)

        #Get reward
        reward = self.adapter.prepare_reward(df_list)


        self.adapter.wandb_log()

        #3.) Check end of Episode
        truncated = self.current_step >= self.steps_per_episode

        # print("Episdoe", self.current_ep ,"step", self.current_step, "reward", reward, "Done", done)

        terminated = False
        if truncated:
            if self.first_episode:
                self.first_episode = False

            if self.current_ep == self.episodes_per_session:
                terminated = True   
        #4.) return observation, reward, done, info
        # print("terminated:" + str(terminated) + " truncated:" + str(truncated))
        return observation.astype(np.float32), reward, terminated, truncated, {"network_stats": df_list}
