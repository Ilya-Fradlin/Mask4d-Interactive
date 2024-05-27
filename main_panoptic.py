import logging
import os
import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import ObjectSegmentation
from utils.utils import flatten_dict
from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.callbacks import ModelCheckpoint


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    cfg.general.gpus = torch.cuda.device_count()
    loggers = []

    if "debugging" in cfg.general.experiment_name:
        # Add debugging options + No logging
        cfg.callbacks = [cfg.callbacks[0]]
        cfg.trainer.detect_anomaly = True
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.log_every_n_steps = 1
        cfg.trainer.max_epochs = 30
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.trainer.limit_train_batches = 0.0002
        cfg.trainer.limit_val_batches = 0.0005

        if cfg.general.experiment_name == "debugging-with-logging":
            cfg.general.visualization_frequency = 1
            for log in cfg.logging:
                print(log)
                loggers.append(hydra.utils.instantiate(log))
                loggers[-1].log_hyperparams(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))

    else:
        if not os.path.exists(cfg.general.save_dir):
            os.makedirs(cfg.general.save_dir)
        else:
            print("EXPERIMENT ALREADY EXIST")
            cfg.general.ckpt_path = f"{cfg.general.save_dir}/last-epoch.ckpt"

        for log in cfg.logging:
            print(log)
            loggers.append(hydra.utils.instantiate(log))
            loggers[-1].log_hyperparams(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))

    model = ObjectSegmentation(cfg)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(config_path="conf", config_name="config_panoptic_4d.yaml")
def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    if cfg.callbacks is not None:
        for cb in cfg.callbacks:
            callbacks.append(hydra.utils.instantiate(cb))

    runner = Trainer(
        callbacks=callbacks,
        logger=loggers,
        **cfg.trainer,
    )
    runner.fit(model, ckpt_path=cfg.general.ckpt_path)


@hydra.main(config_path="conf", config_name="config_panoptic_4d.yaml")
def validate(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=1,
        default_root_dir=str(cfg.general.save_dir),
    )
    runner.validate(model=model, ckpt_path=cfg.general.ckpt_path)


@hydra.main(config_path="conf", config_name="config_panoptic_4d.yaml")
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=1,
        default_root_dir=str(cfg.general.save_dir),
    )
    runner.test(model=model, ckpt_path=cfg.general.ckpt_path)


@hydra.main(config_path="conf", config_name="config_panoptic_4d.yaml")
def main(cfg: DictConfig):
    if cfg["general"]["mode"] == "train":
        train(cfg)
    elif cfg["general"]["mode"] == "validate":
        validate(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
