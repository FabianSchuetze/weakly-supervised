r"""Class that integrates both dataset specific and agnostic configuration"""

# from .dataset_config import conf as dataset_conf
from .agnostic_config import conf as agnostic_conf



def conf(dataset_name: str, command_ling_args=None):
    """Returns the configuration file"""
    config = agnostic_conf
    try:
        dataset = agnostic_conf[dataset_name]
    except KeyError:
        keys = agnostic_conf.keys()
        message = "The available keys are: %s, please define the dataset in"\
                " transferlearning.config.dataset_conf" %(keys)
        raise KeyError(message)
    for param in dataset:
        config[param] = dataset[param]
    if command_ling_args:
        to_dict = command_ling_args.__dict__
        for param in to_dict:
            config[param] = to_dict[param]
    return config
