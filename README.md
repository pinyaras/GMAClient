# Network Agent for NetworkGym


Within this repository, you will find the State-of-the-Art (SOTA) Reinforcement Learning (RL) algorithms sourced from [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/). These algorithms include popular ones such as PPO (Proximal Policy Optimization), DDPG (Deep Deterministic Policy Gradient), SAC (Soft Actor-Critic), TD3 (Twin Delayed Deep Deterministic Policy Gradient), and A2C (Advantage Actor-Critic). Moreover, these algorithms have been integrated to seamlessly interact with the [NetworkGym](https://github.com/IntelLabs/gma/tree/network-gym) Environment.

## ⌛ Installation:
- Cloene the NetworkGymClient:
```
git clone --branch network-gym https://github.com/IntelLabs/gma.git
```
- Step into the NetworkGym folder and Clone this repo:
```
cd NetworkGym
git clone https://github.com/pinyaras/GMAClient NetworkAgent
```
The file structure is organized as follows:
```
📦 NetworkGym
┣ 📂 network_gym_client
┗ 📂 NetworkAgent
  ┗ 📂 stable-baselines3
    ┗ 📜 main_rl.py
  ┗ 📂 cleanRL
    ┗ 📜 custom.py
```

- (Optional) Create a new virtual python environment.
```
python3 -m venv network_venv
source network_venv/bin/activate
```
- Install Required Libraries `pip install -r requirements.txt` or:
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

## 🚀 Start Agent:

- Config the environment using the [json files](https://github.com/IntelLabs/gma/tree/network-gym#%EF%B8%8F-configurable-file-format) provided by NetworkGymClient.
- Start the agent using the following command, and visualize the output in WanDB website.
```
NetworkGym/NetworkAgent
python3 stable-baselines3/main_rl.py --env=[ENV]
```
- where [ENV] has 3 options: `nqso_split`, `qos_steer` and `network_slicing`. If the python program stops after sending out the start request as shown in the following, check if the port fowarding is broken.
```
[qos_steer] environment selected.
[30] Number of users selected.
...
[YOUR_ALGORITHM_NAME]-0 started
[YOUR_ALGORITHM_NAME]-0 Sending GMASim Start Request…
```
