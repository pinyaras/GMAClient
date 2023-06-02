import sys

#base class for use case helpers
class use_case_base_helper:
    def __init__(self):
        self.use_case = "base_class" # will be overwriten by child class
        self.config_json = None
        self.wandb_log_info = None

    def get_use_case(self):
        return self.use_case

    def set_config (self, config_json):
        if config_json['gmasim_config']['use_case'] != self.use_case:
            sys.exit("[ERROR] wrong use case helper. config file use case: " + str(config_json['gmasim_config']['use_case']) + " helper use case: " + str(self.use_case))
        self.config_json = config_json

    def set_wandb_log (self, wandb_log_info):
        self.wandb_log_info = wandb_log_info

    def get_config (self):
        return self.config_json