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
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    runner = Trainer(
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=str(cfg.general.save_dir),
        profiler="simple",
        devices="auto",
        accelerator="gpu",
        strategy="ddp",
        **cfg.trainer,
        # debugging options
        # detect_anomaly=True,
        # num_sanity_val_steps=0,
        # log_every_n_steps=1,
        # max_epochs=30,
        # check_val_every_n_epoch=1,
        # limit_train_batches=0.0002,
        # limit_val_batches=0.0005,
    )
    # runner.fit(model)
    runner.fit(model, ckpt_path="/home/fradlin/Github/Mask4D-Interactive/saved/2024-05-12_092821/last.ckpt")


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
