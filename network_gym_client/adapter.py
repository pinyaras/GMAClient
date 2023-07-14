import sys

class adapter:
    """base class for data format adapter.

    This class is an data format "adapter" between the gymnasium environment and network_gym environment.
    Adapter transforms the network stats measurements to obs and reward.
    It also transforms the action to a policy that can be applied to the network.
    """
    def __init__(self, wandb):
        """Initialize.

        Args:
            wandb (database): WanDB database
        """
        self.config_json = None
        self.wandb_log_info = None
        self.wandb = wandb

    def set_config (self, config_json):
        """Set the configuration file.

        Args:
            config_json (JSON): the configuration file
        """
        if config_json['gmasim_config']['env'] != self.env:
            sys.exit("[ERROR] wrong environment helper. config file environment: " + str(config_json['gmasim_config']['env']) + " helper environment: " + str(self.env))
        self.config_json = config_json
    
    def wandb_log (self):
        """Send the log information to WanDB.
        """
        # send info to wandb
        #print(self.wandb_log_info)
        self.wandb.log(self.wandb_log_info)
        self.wandb_log_info = None

    def df_to_dict(self, df, description):
        """Transform datatype from pandas.dataframe to dictionary.

        Args:
            df (pandas.dataframe): a pandas.dataframe object
            description (string): a descritption for the data

        Returns:
            dictionary: converted data with dictionary format
        """
        df_cp = df.copy()
        df_cp['user'] = df_cp['user'].map(lambda u: f'UE{u}_'+description)
        # Set the index to the 'user' column
        df_cp = df_cp.set_index('user')
        # Convert the DataFrame to a dictionary
        data = df_cp['value'].to_dict()
        return data