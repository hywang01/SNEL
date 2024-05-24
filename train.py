import argparse
import time
import collections
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from rich import pretty, print
import os

from snel.trainer import build_trainer


wandb.login(key='')

def wandb_init(cfg: dict):
    wandb.init(
        project='my_proj',
        group=cfg.exp_group,
        notes=cfg.exp_desc+time.strftime("%Y_%m%d_%H%M%S"),
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )


@hydra.main(version_base=None, config_path="./configs",  config_name="default_config")
def main(cfgs: DictConfig) -> None:
    '''
    1. setup configs using Hydra and Omegaconf
    2. setup logger using Wandb
    3. build dataloaders for training
    4. build model, loss, optimizer, etc
    '''

    pretty.install()   
    print(OmegaConf.to_yaml(cfgs))

    cfgs.output_dir = os.path.join(cfgs.output_dir, cfgs.exp_name)
    trainer = build_trainer(cfgs)
    trainer.train()


if __name__ == '__main__':
    main()
