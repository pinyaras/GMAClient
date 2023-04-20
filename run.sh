#!/bin/bash

# python3 cleanrl_ddpg.py --use_case nqos_split 


python3 main_rl.py --use_case nqos_split --num_users=8 --lte_rb=50
python3 main_rl.py --use_case nqos_split --num_users=8 --lte_rb=75
python3 main_rl.py --use_case nqos_split --num_users=8 --lte_rb=100

# python3 main_rl.py --use_case nqos_split --num_users=16 --lte_rb=50
# python3 main_rl.py --use_case nqos_split --num_users=16 --lte_rb=75
# python3 main_rl.py --use_case nqos_split --num_users=16 --lte_rb=100

# python3 main_rl.py --use_case nqos_split --num_users=32 --lte_rb=50
# python3 main_rl.py --use_case nqos_split --num_users=32 --lte_rb=75
# python3 main_rl.py --use_case nqos_split --num_users=32 --lte_rb=100


#Test sum_rate
# python3 main_rl.py --use_case nqos_split --num_users=4 --lte_rb=50
# python3 main_rl.py --use_case nqos_split --num_users=8 --lte_rb=50
# python3 main_rl.py --use_case nqos_split --num_users=16 --lte_rb=50
# python3 main_rl.py --use_case nqos_split --num_users=24 --lte_rb=50
# python3 main_rl.py --use_case nqos_split --num_users=32 --lte_rb=50

# python3 main_rl.py --use_case nqos_split --num_users=4 --lte_rb=75
# python3 main_rl.py --use_case nqos_split --num_users=8 --lte_rb=75
# python3 main_rl.py --use_case nqos_split --num_users=16 --lte_rb=75
# python3 main_rl.py --use_case nqos_split --num_users=24 --lte_rb=75

# python3 main_rl.py --use_case nqos_split --num_users=4 --lte_rb=100
# python3 main_rl.py --use_case nqos_split --num_users=8 --lte_rb=100
# python3 main_rl.py --use_case nqos_split --num_users=16 --lte_rb=100
# python3 main_rl.py --use_case nqos_split --num_users=24 --lte_rb=100

# python3 main_rl.py --use_case nqos_split --num_users=32
# python3 main_rl.py --use_case nqos_split --num_users=40
# python3 main_rl.py --use_case nqos_split --num_users=48
# python3 main_rl.py --use_case nqos_split --num_users=56
# python3 main_rl.py --use_case nqos_split --num_users=64

