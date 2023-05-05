import argparse
import random
from dataclasses import fields

import numpy as np
import torch

from config import Config
from org_trainer import Trainer
from utils import set_seed


def get_args():
    parser = argparse.ArgumentParser()
    for field in fields(Config):
        name = field.name
        default = getattr(Config, name)
        parser.add_argument("--{}".format(name), default=default, type=type(default))
    args = parser.parse_args()
    return args


def main():
    seed = 28
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #set_seed()
    args = get_args()
    kwargs = {}
    for field in fields(Config):
        name = field.name
        kwargs[name] = getattr(args, name)
    config = Config(**kwargs)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
