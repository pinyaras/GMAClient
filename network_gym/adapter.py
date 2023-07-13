import sys

#base class for environment adapter
class adapter:
    def __init__(self, wandb):
        self.config_json = None
        self.wandb_log_info = None
        self.wandb = wandb

    def set_config (self, config_json):
        if config_json['gmasim_config']['env'] != self.env:
            sys.exit("[ERROR] wrong environment helper. config file environment: " + str(config_json['gmasim_config']['env']) + " helper environment: " + str(self.env))
        self.config_json = config_json
    
    def wandb_log (self):
        # send info to wandb
        #print(self.wandb_log_info)
        self.wandb.log(self.wandb_log_info)
        self.wandb_log_info = None

    def df_to_dict(self, df, description):
        df_cp = df.copy()
        df_cp['user'] = df_cp['user'].map(lambda u: f'UE{u}_'+description)
        # Set the index to the 'user' column
        df_cp = df_cp.set_index('user')
        # Convert the DataFrame to a dictionary
        data = df_cp['value'].to_dict()
        return data