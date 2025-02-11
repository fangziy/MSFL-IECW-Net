import argparse
import yaml
import os 



def load_config_from_yaml(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def mk_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)