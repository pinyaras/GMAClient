def convert_owd(df_owd):

    # can you add a check what if Wi-Fi or LTE link does not have measurement....
    
    #print(qos_rate)
    df_pivot = df_owd.pivot_table(index="user", columns="cid", values="value", aggfunc="first")[["Wi-Fi", "LTE"]]
    # Rename the columns to "wi-fi" and "lte"
    df_pivot.columns = ["wi-fi", "lte"]
    # Compute the delay difference between 'Wi-Fi' and 'LTE' for each user
    # delay_diffs = df_pivot['wi-fi'].subtract(df_pivot['lte'], axis=0)
    # abs_delay_diffs = delay_diffs.abs()
    # print(abs_delay_diffs)
    # local_reward = 1/abs_delay_diffs*100
    # reward = abs_delay_diffs.mean()
    return df_pivot.values