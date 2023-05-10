import torch
import numpy as np
import pandas as pd

def convert_owd(df_owd):

    # can you add a check what if Wi-Fi or LTE link does not have measurement....
    
    #print(qos_rate)
    df_pivot = df_owd.pivot_table(index="user", columns="cid", values="value", aggfunc="first")[["Wi-Fi", "LTE"]]
    # Rename the columns to "wi-fi" and "lte"
    df_pivot.columns = ["wi-fi", "lte"]
    # Compute the delay difference between 'Wi-Fi' and 'LTE' for each user
    delay_diffs = df_pivot['wi-fi'].subtract(df_pivot['lte'], axis=0)
    abs_delay_diffs = delay_diffs.abs()
    df_pivot["diff"] = abs_delay_diffs
    return df_pivot.values

# def convert_phy_max_rate(df_):
#     traffic ,rate_1, rate_2 = s_t[ueid]
#     ratio = rate_1 / (rate_1 + rate_2)

def verify_LTE_owd(df):
    """
    Verifies if every user in the input DataFrame has an entry with 'LTE' in the 'cid' column.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'user' and 'cid'.

    Returns:
        bool: True if every user has an entry with 'LTE' in the 'cid' column, False otherwise.
    """
    # Count the number of unique users
    num_users = len(df['user'].unique())

    # Count the number of entries with 'LTE' in the 'cid' column
    num_lte_entries = df.loc[df['cid'] == 'LTE', 'user'].nunique()

    print("***************************",num_lte_entries)

    # Check if the counts are equal
    if num_users == num_lte_entries:
        return True
    else:
        return False


def check_LTE_owd(df):

    # get the unique set of users
    users = df['user'].unique()

    # loop through each user and check if there is a 'LTE' entry in the 'cid' column
    # if not, add an entry with a value of 0
    for user in users:
        if 'LTE' not in df.loc[df['user'] == user, 'cid'].values:
            last_wifi_entry = df.loc[(df['user'] == user) & (df['cid'] == 'Wi-Fi')].tail(1)
            new_entry = last_wifi_entry.copy()
            new_entry['cid'] = 'LTE'
            new_entry['value'] = 0.0
            df = pd.concat([df, new_entry])

    return df

def rescale_ratio_tanh(arr):
    # Define the minimum and maximum values
    low = -1
    high = 1

    # Scale the values of the array to the range of [-1, 1]
    scaled_arr = ((2 * (arr - low)) / (high - low)) - 1
    print(scaled_arr)
    return scaled_arr

def phy_max_rate_ratio(local_state):
    lte_link = local_state[0]
    wifi_link = local_state[1]

    ratio = wifi_link / (wifi_link + lte_link)

    # if wifi_delay > lte_delay:
    #     split_ratio -= 1
    # elif lte_delay > wifi_delay:
    #     split_ratio -= 1

    print(ratio)
    
    return rescale_ratio_tanh(ratio)

def guided_exploration(action_mean, explore_action ,delay_list, delay_thresh):
    d1,d2, delay_diff = delay_list
    action_mean = action_mean.cpu().numpy()
    explore_action = explore_action.cpu().numpy()

    # if delay_diff < 2:
    #     ue_noise = 0.00

    # sample_action = action_mean + torch.normal(action_mean, ue_noise)
    print("action_mean :",action_mean, " sample action :" ,explore_action, " d1 :",d1, " d2 :",d2, "diff :",delay_diff)
    # original_action = action_mean
    guided_action = action_mean
    if delay_diff > delay_thresh:
        if d1 < d2:
            print("Increasing traffic to Link1")

            diff = explore_action-action_mean
            # if action_mean < 0:
            #     action_mean = 0.0
            offset = action_mean + abs(diff)
            # action = tf.clip_by_value(offset, action_mean, offset)
            guided_action = np.clip(offset, action_mean, offset)

        elif d1 > d2:

            print("Increasing traffic to Link2")
            diff = explore_action-action_mean
            # if action_mean > 0:
            #     action_mean = 0.0
            offset = action_mean - abs(diff)
            # action = tf.clip_by_value(offset, offset, action_mean)
            guided_action = np.clip(offset, action_mean, offset)

    else:
        guided_action = action_mean
    print("guieded action :",guided_action)
    return guided_action