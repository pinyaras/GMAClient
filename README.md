# gmasim-gym-client
The python-based algorithm client for DD-GMAsim: a ns3-based Data-Driven AI/ML-enabled Multi-Access Network Simulator.
This release includes the ML algorithms from the [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/), e.g., PPO, DDPG, SAC, TD3, and A2C.

```
git clone https://github.com/pinyaras/GMAClient.git
```

## Prerequisite:
- Download [gmasim_open_api](https://github.com/IntelLabs/gma/blob/master/GMAsim/gmasim_open_api.py) library. If the GMA algorithm client returns errors, try to check if the [gmasim_open_api](https://github.com/IntelLabs/gma/blob/master/GMAsim/gmasim_open_api.py) has any updates.

```
cd GMAClient
wget https://raw.githubusercontent.com/IntelLabs/gma/master/GMAsim/gmasim_open_api.py
```

## Start GMA Algorithm Client:
- Install Required Libraries:
```
pip3 install pyzmq
pip3 install pandas
pip3 install stable-baselines3
pip3 install wandb
pip3 install tensorboard
```
- ⚠️ Setup the port forwarding from the local port 8088 to the mlwins-v01 external server port 8088 via the SSH gateway (If you launch the algorithm client at the mlwins-v01 machine, you do not need port forwarding): 
`ssh -L 8088:mlwins-v01.research.intel-research.net:8088 ssh.intel-research.net`. 
If the command does not work, add your user account before the `ssh.intel-research.net` as follows,
`ssh -L 8088:mlwins-v01.research.intel-research.net:8088 [YOUR_USER_NAME]@ssh.intel-research.net`. Also Add the following instructions to your ssh configure file, replace **[YOUR_USER_NAME]** with your user name and update **[PATH_TO_SSH]** accordingly.
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
- Update the common configuration file [common_config.json](common_config.json)

```json
{
  "algorithm_client_port": 8088,//do not change
  "algorithm_client_identity": "test-gym-client",//Make sure to change the "algorithm_client_identity": from "test-algorithm" to "[YOUR_INITIALS]-[ALGORITHM_NAME]". E.g., "mz-baseline".
  "enable_rl_agent": true,//set to true to enable rl agent, set to false to use GMA's baseline algorithm.

  "rl_agent_config":{
    "agent": "PPO",//supported agents are "PPO", "DDPG", "SAC", "TD3", "A2C", "SingleLink".
    "link_type": "lte",//"lte" or "wifi", only work if the "agent" is set to "SingleLink"
    "policy": "MlpPolicy",
    "train": true,//set to true to train model, set to false to test pretrained model.
    "reward_type" : "throughput",//supported reward types are "throughput", "delay", "utility", and "wifi_qos_user_num".
    "input": "matrix"//set the input type for observations, "flat" or "matrix".
  }
}
```
- Update the use case depend configuration file [qos_steer_config.json](qos_steer_config.json) or [nqos_split_config.json](nqos_split_config.json)
```json
{
  "gmasim_config":{
      "type": "gmasim-start", //do not change
      "use_case": "nqos_split" or "qos_steer", //do not change
      "simulation_time_s": 10,
      "random_run": 2, //change the random seed for this simulation run
      "downlink": true, //set to true to simulate downlink data flow, set to false to simulate uplink data flow.
      "max_wait_time_for_action_ms": -1, //the max time the gmasim worker will wait for an action. set to -1 will cap the wait time to 100 seconds.
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
        "WIFI": {
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
  "gmasim_action_template":{//do not change
      "type": "gmasim-action",
      "end_ts": 0,
      "downlink": true,
      "split_ratio_list":[
        {"cid":"Wi-Fi","user":0,"value":1},
        {"cid":"LTE","user":0,"value":1}
      ]
  }
}
```



- Start the client using the following command, and visualize the output in WANDB.
```
python3 main_rl.py --use_case=nqos_split
```
or
```
python3 main_rl.py --use_case=qos_steer
```
- If the port forwading is broken, the python program stops after sending out the start request as shown in the following:
```
[qos_steer] use case selected.
[30] Number of users selected.
...
[YOUR_ALGORITHM_NAME]-gym-client-GMA-0 started
[YOUR_ALGORITHM_NAME]-gym-client-GMA-0 Sending GMASim Start Request…
```

TODO: Website for NetAIGym. 3 Scenarios.

