---
title: Overview
---
# Overview

The NetworkGym contains three components (client, server, and environment) and two interfaces (northbound and southbound). As shown in the following graph. 


```{mermaid}
flowchart TB

subgraph network_gym_server
northbound <--> southbound[[southbound_interface]]
end

subgraph network_gym_env
southbound_interface
simulator
emulator
testbed 
end

agent <--> gymnasium.env

gymnasium.env -- action --> adapter
adapter -- obs,rewards --> gymnasium.env

subgraph network_gym_client
gymnasium.env
adapter
northbound_interface[[northbound_interface]]
end



adapter --policy--> northbound_interface
northbound_interface --network_stats--> adapter


northbound_interface --env_config,policy--> northbound[[northbound_interface]]
northbound --network_stats--> northbound_interface




southbound --env_config,policy--> southbound_interface[[southbound_interface]]
southbound_interface --network_stats--> southbound


click gymnasium.env "/api/network_gym_client/env.html" _blank
style gymnasium.env fill:#1E90FF,color:white,stroke:white

click adapter "/api/network_gym_client/adapter.html" _blank
style adapter fill:#1E90FF,color:white,stroke:white

click northbound_interface "/api/network_gym_client/northbound_interface.html" _blank
style northbound_interface fill:#1E90FF,color:white,stroke:white
```

## Client
Within the NetworkGymClient, it consits of a norhtbound interface, an adapter and custumized Gymnasium environment. The northbound interface connects the client to the server. It also allows the client to select and configure network environment. Finally, the environment specific adapters to transform the NetworkGym dataformat to the gymnasium data format, then communicate with the gymnasium compatible agents such as stable-baselines3 and cleanRL.

## Server
For the NetworkGymServer, it communicates with the client with northbound interface and interacts with the environment using the southbound interface. It also keeps a routing map for each active client and its assigned environment during a connected session.

```{note}
The southbound interface will be released in the future.
```

## Environment
The NetworkGym environment connects the worker (either a simulator, emulator or testbed) to the server using the southbound interface. At present, we only support the ns-3 based simulator with **Three** distinct network environments and provides support for **Ten** different network stats measurement metrics.


## Class Diagram

```{mermaid}
classDiagram
network_gym_client *-- env
env *-- adapter
env *-- northbound_interface

northbound_interface -- network_gym_server
network_gym_server  -- southbound_interface

network_gym_env *-- southbound_interface

class env
env: +reset() obs, info
env: +step(action) obs, reward, ...

class network_gym_client

class network_gym_server
network_gym_server: +northbound_interface
network_gym_server: +southbound_interface

class network_gym_env

class northbound_interface
northbound_interface: +connect() void
northbound_interface: +send(policy) void
northbound_interface: +recv() network_stats

class adapter 
adapter : +prepare_obs(network_stats) obs
adapter : +prepare_reward(network_stats) reward
adapter : +prepare_policy(action) policy
```
