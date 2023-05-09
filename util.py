import torch
import numpy as np
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

def guided_exploration(action_mean, delay_list, ue_noise, delay_diff):
    d1,d2 = delay_list
    if delay_diff < 2:
        ue_noise = 0.00

    sample_action = action_mean + torch.normal(action_mean, ue_noise)
    print("action_mean :",action_mean, " sample action :" ,sample_action, " d1 :",d1, " d2 :",d2, "diff :",delay_diff)
    original_action = action_mean
    guided_action = action_mean
    if delay_diff > 2:
        if d1 < d2:
            print("Increasing traffic to Link1")

            diff = sample_action-action_mean
            # if action_mean < 0:
            #     action_mean = 0.0
            offset = action_mean + abs(diff)
            # action = tf.clip_by_value(offset, action_mean, offset)
            guided_action = np.clip(offset, action_mean, offset)

        elif d1 > d2:

            print("Increasing traffic to Link2")
            diff = sample_action-action_mean
            # if action_mean > 0:
            #     action_mean = 0.0
            offset = action_mean - abs(diff)
            # action = tf.clip_by_value(offset, offset, action_mean)
            guided_action = np.clip(offset, action_mean, offset)

    else:
        guided_action = action_mean
        freeze = False
    print("guieded action :",guided_action)
    return guided_action, freeze