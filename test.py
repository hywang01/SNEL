import os
from omegaconf import DictConfig
import hydra
import wandb
from rich import print

from snel.trainer import build_trainer


@hydra.main(version_base=None, config_path="./configs",  config_name="testing_config")
def main(cfgs: DictConfig) -> None:

    # build trainer
    model_file_name = 'test_model'
    cfgs.exp_name = model_file_name + '_exp'
    
    cfgs.trainers.train_mode = False
    trainer = build_trainer(cfgs)

    model_epoch = None # cfgs.models.optim_max_epoch
    trainer.load_model(os.path.join(cfgs.output_dir, model_file_name), 
                       epoch=model_epoch)
    trainer.test()


if __name__ == '__main__':
    main()
