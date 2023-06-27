# NetAI Client 
NetAI Client is a python-based client for NetAI Gym: a ns3-based Data-Driven AI/ML-enabled Multi-Access Network Simulator. In this release, NetAI Client supports three use cases nqos_split, qos_steer, and network slicing. It also includes the SOTA RL algorithms from the [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/), e.g., PPO, DDPG, SAC, TD3, and A2C.

## ⌛ Installation:
- Clone this repo.
```
git clone https://github.com/pinyaras/GMAClient.git
```
- Download the [netai_gym_open_api](https://github.com/IntelLabs/gma/blob/master/netai_gym/netai_gym_open_api.py) library.

```
cd GMAClient
wget https://raw.githubusercontent.com/IntelLabs/gma/master/netai_gym/netai_gym_open_api.py
```
- (Optional) Create a new virtual python environment.
```
python3 -m venv netai-venv
source netai-venv/bin/activate
```
- Install Required Libraries.
```
pip3 install pyzmq
pip3 install pandas
pip3 install stable-baselines3
pip3 install wandb
pip3 install tensorboard
```

## 🔗 Port Forwarding (Skip this if client is deployed on the mlwins-v01):
- To setup port forwarding from the local port 8088 to the mlwins-v01 external server port 8088 via the SSH gateway, run the following command in a screen session, e.g., `screen -S port8088`.
``` 
ssh -L 8088:mlwins-v01.research.intel-research.net:8088 ssh.intel-research.net
```
- If the previous command does not work, add your user account before the `ssh.intel-research.net` as follows.
```
ssh -L 8088:mlwins-v01.research.intel-research.net:8088 [YOUR_USER_NAME]@ssh.intel-research.net
```
 - If the previous command also does not work, add the following instructions to your ssh configure file, replace **[YOUR_USER_NAME]** with your user name and update **[PATH_TO_SSH]** accordingly.
```
# COMMAND: ssh mlwins

Host gateway
  HostName ssh.intel-research.net
  User [YOUR_USER_NAME]
  Port 22
  IdentityFile /home/[PATH_TO_SSH]/.ssh/id_rsa

Host mlwins
  HostName mlwins-v01.research.intel-research.net
  User [YOUR_USER_NAME]
  Port 22
  IdentityFile /home/[PATH_TO_SSH]/.ssh/id_rsa
  ProxyJump gateway
  LocalForward 8088 localhost:8088
```

## 🚀 Start NetAI Algorithm Client:

- Update the common configuration file [common_config.json](common_config.json). Go to the ⚙️ Configurable File Format Section for more details.

- Update the use case depend configuration file [qos_steer_config.json](qos_steer/qos_steer_config.json) or [nqos_split_config.json](nqos_split/nqos_split_config.json) or [network_slicint.json](network_slicing/network_slicing_config.json)


- Start the client using the following command, and visualize the output in WanDB website.
```
python3 main_rl.py --use_case=[USE_CASE]
```
- where [USE_CASE] has 3 options: `nqso_split`, `qos_steer` and `network_slicing`. If the python program stops after sending out the start request as shown in the following, check if the port fowarding is broken.
```
[qos_steer] use case selected.
[30] Number of users selected.
...
[YOUR_ALGORITHM_NAME]-0 started
[YOUR_ALGORITHM_NAME]-0 Sending GMASim Start Request…
```

## 📁 File Structure:

```
📦NetAIClient
┣ 📜main_rl.py (➡️stable-baselines3, ➡️WanDB)
┣ 📜common_config.json
┣ 📜netai_gym.py
┣ 📜netai_gym_open_api (➡️netai-simulator)
┗ 📂[USE_CASE]
  ┣ 📜[USE_CASE]_config.json
  ┗ 📜[USE_CASE]_helper.py
```

- Excuting the 📜 main_rl.py file will start a new simulation. The use case must be selected using the `--use_case` command. The 📜common_config.json is used in all use cases. Depends on the selected use cases, th associated 📜[USE_CASE]_config.json and 📜[USE_CASE]_helper.py will be loaded. The 📜[USE_CASE]_helper.py helps preparing observations, rewards and actions for the selected use case.
- The 📜 main_rl.py create a NetAI Gym environment (imported from 📜netai_gym.py), which remotely connects to the ns-3 based NetAI Simualtor (hosted in vLab machine) using the 📜netai_gym_open_api. 📜 main_rl.py also creates a reinforcement learning model (imported from ➡️stable-baselines3) to interact with the NetAI Gym environment. The results are synced to ➡️WanDB database. We provide the following code snippet from the 📜 main_rl.py as an example. After the model is trained using the NetAI Gym's environment, it can be easily deployed in any environment.
- This file strcuture isolates the function and variables associated to each use case. Therefore, a new use case can be added by adding ad new [USE_CASE] folder with 📜[USE_CASE]_config.json and 📜[USE_CASE]_helper.py files. 

```python
from stable_baselines3 import PPO
from netai_gym import NetAIEnv
from nqos_split_helper import use_case_helper
import wandb

...
# training
env = NetAIEnv(client_id, use_case_helper, config_json, wandb) # passing id, use case helper, configure file and wanDb as arguments
model = PPO("MlpPolicy", env, verbose=1) # you can change the env to your deployment environment when the model is trained.
model.learn(total_timesteps=10_000)
...
```
 
## ⚙️ Configurable File Format:
- [common_config.json](common_config.json)

```json
{
  "algorithm_client_port": 8088,//do not change
  "session_name": "test",//Make sure to change the "session_name" to your assgined session name.
  "session_id": "test",//Make sure to change the "session_id" to your assgined keys.
}
```
- [qos_steer_config.json](qos_steer/qos_steer_config.json) or [nqos_split_config.json](nqos_split/nqos_split_config.json) or [network_slicint.json](network_slicing/network_slicing_config.json)
```json
{
  //never use negative value for any configure vale!!!
  "gmasim_config":{
      "type": "gmasim-start", //do not change
      "use_case": "nqos_split" or "qos_steer" or "network_slicing", //do not change
      "simulation_time_s": 10,
      "random_run": 2, //change the random seed for this simulation run
      "downlink": true, //set to true to simulate downlink data flow, set to false to simulate uplink data flow.
      "max_wait_time_for_action_ms": -1, //the max time the netaisim worker will wait for an action. set to -1 will cap the wait time to 100 seconds.
      "enb_locations":{//x, y and z locations of the base station, we support 1 base station only
        "x":40,
        "y":0,
        "z":3
      },
      "ap_locations":[//x, y and z location of the Wi-Fi access point, add or remove element in this list to increase or reduce AP number. We support 0 AP as well.
        {"x":30,"y":0,"z":3},
        {"x":50,"y":0,"z":3}
      ],
      "num_users" : 4,
      "slice_list":[ //network slicing use case only, resouce block group (rbg) size maybe 1, 2, 3 or 4, it depends on the resource block num, see table 7.1.6.1-1 of 36.213
        {"num_users":5,"dedicated_rbg":2,"prioritized_rbg":3,"shared_rbg":4},
        {"num_users":5,"dedicated_rbg":5,"prioritized_rbg":6,"shared_rbg":7},
        {"num_users":5,"dedicated_rbg":0,"prioritized_rbg":0,"shared_rbg":100}
      ],
      "user_left_right_speed_m/s": 1,
      "user_location_range":{//initially, users will be randomly deployed within this x, y range. if user_left_right_speed_m > 0, the user will move left and right within this boundary.
        "min_x":0,
        "max_x":80,
        "min_y":0,
        "max_y":10,
        "z":1.5
      },
      "app_and_measurement_start_time_ms": 1000, //when the application starts traffic and send measurement to RL agent
      "transport_protocol": "tcp", //"tcp" or "udp"
      "min_udp_rate_per_user_mbps": 2, // if "transport_protocol" is "udp", this para controls the min sending rate.
      "max_udp_rate_per_user_mbps": 3, // if "transport_protocol" is "udp", this para controls the max sending rate.
      "qos_requirement": {//only for qos_steer use case
        "test_duration_ms": 500,//duration for qos testing
        "delay_bound_ms": 100,//max delay for qos flow
        "delay_violation_target":0.02, //delay violation target for qos flow
        "loss_target": 0.001 //loss target for qos flow
      },
      "GMA": {
          "downlink_mode": "auto", //"auto", "split", or "steer". "auto" will config UDP and TCP ACK as steer and TCP data as split.
          "uplink_mode": "auto", //"auto", "split", or "steer". "auto" will config UDP and TCP ACK as steer and TCP data as split.
          "measurement_interval_ms": 100, //duration of a measurement interval.
          "measurement_guard_interval_ms": 0, //gap between 2 measurement interval
          "respond_action_after_measurement": true //do not change
        },
        "Wi-Fi": {
          "ap_share_same_band": false, //set to true, ap will share the same frequency band.
          "measurement_interval_ms": 100,
          "measurement_guard_interval_ms": 0
        },
        "LTE": {
          "qos_aware_scheduler": true, //qos_steer use case only, set to true to enable qos aware scheduler for LTE.
          "resource_block_num": 25, //number of resouce blocks for LTE, 25 for 5 MHZ, 50 for 10 MHZ, 75 for 15 MHZ and 100 for 20 MHZ.
          "measurement_interval_ms": 100,
          "measurement_guard_interval_ms": 0
        }
      },

  "rl_agent_config":{//see the following for config options 
    "agent": "PPO",
    "policy": "MlpPolicy",
    "train": true,
    "reward_type" : "delay",
    "input": "matrix"
  },

  "rl_agent_config_option_list"://do not change this list, it provides the available inputs for the rl_agent_config
  {
    "agent": ["", "PPO", "DDPG"],// set to "" will disable agent, i.e., use the system's default algorithm for offline data collection
    "policy": ["MlpPolicy"],
    "train": [true, false],//set to true to train model, set to false to test pretrained model.
    "reward_type" : ["delay", "throughput"],
    "input": ["matrix", "flat"]//set the input type for observations
  },

  "gmasim_action_template":{//do not change
  }
}
```

## 🚩 TODOs

- TODO 1: Create a Website for NetAI Gym including the 3 Scenarios.
- TODO 2: show two instance (System default vs DDPG) at the same time, visualize results using WanDB and influxDB.
- TODO 3: two slice plus demo video for NetAI Gym.
- TODO 4: move the main_rl.py ousite the folder...
