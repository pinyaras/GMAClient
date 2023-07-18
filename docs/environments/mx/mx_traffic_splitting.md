---
title: Multi-Access(MX) Traffic Splitting
---
# Multi-Access(MX) Traffic Splitting

This environment is part of the multi-access traffic management environments which contains general information about the environment.
the agent performs the following action at regular intervals: it updates the traffic split ratio for each user, considering both Wi-Fi and LTE connections.

| | |
| ----- | ---- |
| Action Space    |  `Box(0.0, 1.0, (N,), float32)`  |
| Observation Space | `blabla`  |
| Select Environment | `python3 main_rl.py --env=nqos_split`  |

## Description
The Multi-Access(MX) Traffic Splitting is a traffic management problem that consists of multiple users randomly distributed on a 2D plane, and all users connect to Cellualr and Wi-Fi links.
The goal of the traffic management is to strategically split traffic over both links, such that the throughput is high and latency is low.

**TODO: add the figure...**

## Prerequisite

Make sure you have the access to the NetworkGym Server on [vLab](https://registration.intel-research.net/) machines and downloaded the [NetworkGymClient](https://github.com/pinyaras/GMAClient).

## Action Space
The action space is a `ndarray` with shape `(N,) representing the traffic ratio over Wi-Fi for N users. The traffic ratio over Cellualr equals (1.0 - action).
| Num | Action | Min | Max |
| ----- | ---- | ----- | ---- |
| 0 | Wi-Fi traffic ratio for user 0 | 0.0 | 1.0 |
| 1 | Wi-Fi traffic ratio for user 1| 0.0 | 1.0 |
| ... | | | |
| N-1 | Wi-Fi traffic ratio for user N-1| 0.0 | 1.0 |

## Observation Space
The observation is a `ndarray` with shape `(3,N,)` representing the x-y coordinates of the pendulum’s free end and its angular velocity.
| Num | Observation | Min | Max |
| ----- | ---- | ----- | ---- |
| 0 | Wi-Fi traffic ratio for user 0 | 0.0 | 1.0 |
| 1 | Wi-Fi traffic ratio for user 1| 0.0 | 1.0 |
| ... | | | |
| N-1 | Wi-Fi traffic ratio for user N-1| 0.0 | 1.0 |

## Transition Dynamics
Given a policy, the users will split traffic over Wi-Fi and Celluar using the following ratio.
- Wi-Fi traffic : Wi-Fi traffic ratio/32
- Cellualr: 1 - (Wi-Fi traffic ratio/32)

## Reward
The reward can be customized in the [network_stats_to_reward](http://ns3db-v01.jf.intel.com:3000/docs/client/adapter#network_stats_to_reward) function.
By default, we compute a utility function: log(throughput) - log(one-way-delay) as the reward.
The goal of the utility function is to maximize the throughput and minimizing delay.

## Starting State
The position of the users is assigned a uniform random value in a 2D plane with (x, y) boundries. The starting velocity of the users can be configure in the JSON file.
More configurable parameters can be found in Network Environment Simulation parameters section of [Attributes](http://ns3db-v01.jf.intel.com:3000/docs/client/northbound_api#attributes).

## Episode Truncation
The episode truncates at L time steps, where L can be configured in the JSON file.

## Simulation End
The simulation stops at the starting time + L*E, where E is the number of episodes per simulation.

## Arguments
We use the JSON file to configure network environment paramters. See the parameter section in Network Environment Simulation parameters section of [Attributes](http://ns3db-v01.jf.intel.com:3000/docs/client/northbound_api#attributes) for more details.


## add one page for custom reward and custom observation.
## we can configre the steps per episode, and episode number per run.
