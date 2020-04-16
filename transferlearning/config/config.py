r"""Class that integrates both dataset specific and agnostic configuration"""

from .dataset_config import conf as dataset_conf
from .agnostic_config import conf as agnostic_conf



def conf(dataset_name: str):
    """Returns the configuration file"""
    config = agnostic_conf
    dataset = dataset_conf[dataset_name]
    for param in dataset:
        config[param] = dataset[param]
    return config
