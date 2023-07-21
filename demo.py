import argparse
import numpy as np
import time
from network_gym_client import Env as NetworkGymEnv
from network_gym_client import load_config_file
from gymnasium.wrappers import NormalizeObservation

def main():
    # this demo script connects to a environment with system default policy, e.g., send empty action.
    # Replace the empty action with the action generated from your agent.

    args = arg_parser()
    config_json = load_config_file(args.env)

    # overwrite the configuration to use the random action.
    config_json['rl_agent_config']['agent']  = 'random'

    # Create the environment
    print("[" + args.env + "] environment selected.")
    env = NetworkGymEnv(args.client_id, config_json) # make a network env using pass client id and configure file arguements.
    
    # use the wrapper to return nomailized observation.
    # normal_obs_env = NormalizeObservation(env)

    #use the random action ...
    num_steps = 1000
    obs, info = env.reset()

    for step in range(num_steps):

        # replace the action 
        #action = np.array([])#no action from the rl agent -> the environment will use system default policy
        action = env.action_space.sample()  # agent policy that uses the observation and info
        # apply the action with the action generated from your agent.
        obs, reward, terminated, truncated, info = env.step(action)

        print(obs)
        # If the epsiode is up (simulation still running), then start another one
        if truncated:
            obs, info = env.reset()

        # If the simulation is end, exit
        if terminated:
            break
        
def arg_parser():
    parser = argparse.ArgumentParser(description='Network Gym Client')
    parser.add_argument('--env', type=str, required=False, choices=['nqos_split', 'qos_steer', 'network_slicing'], default='nqos_split',
                        help='Select a environment to start Network Gym Client (nqos_split, qos_steer, network_slicing)')
    parser.add_argument('--client_id', type=int, required=False, default=0,
                        help='Select client id, use different client id to launch parallel environments')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    time.sleep(10)