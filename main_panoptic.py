import logging
import os
import argparse
import hydra
import torch
from datetime import datetime
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import ObjectSegmentation
from utils.utils import flatten_dict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models.metrics.utils import MemoryUsageLogger


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    cfg.general.gpus = torch.cuda.device_count()
    print(f"Number of gpus: {cfg.general.gpus}")

    loggers = []

    if "debugging" in cfg.general.experiment_name:
        os.environ["WANDB_MODE"] = "dryrun"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["GLOO_LOG_LEVEL"] = "DEBUG"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        # Add debugging options + No logging
        cfg.data.dataloader.voxel_size = 0.2
        cfg.data.dataloader.batch_size = 2
        cfg.data.dataloader.num_workers = 1
        cfg.trainer.detect_anomaly = True
        cfg.trainer.num_sanity_val_steps = 1
        cfg.trainer.log_every_n_steps = 1
        cfg.trainer.max_epochs = 30
        cfg.trainer.check_val_every_n_epoch = 5
        cfg.trainer.limit_train_batches = 2  # 0.0002
        cfg.trainer.limit_val_batches = 2
        cfg.general.save_dir = os.path.join("saved", cfg.general.experiment_name)

        if cfg.general.experiment_name == "debugging-with-logging":
            cfg.general.visualization_frequency = 1
            if not os.path.exists(cfg.general.save_dir):
                os.makedirs(cfg.general.save_dir, exist_ok=True)

            loggers.append(
                WandbLogger(
                    project=cfg.general.project_name,
                    name=cfg.general.experiment_name,
                    save_dir=cfg.general.save_dir,
                    id=cfg.general.experiment_name,
                    entity="rwth-data-science",
                )
            )
            loggers[-1].log_hyperparams(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))

    else:
        if not os.path.exists(cfg.general.save_dir):
            os.makedirs(cfg.general.save_dir, exist_ok=True)
            # else:
            #     print("EXPERIMENT ALREADY EXIST")
            #     cfg.general.ckpt_path = f"{cfg.general.save_dir}/last-epoch.ckpt"
        loggers.append(
            WandbLogger(
                project=cfg.general.project_name,
                name=cfg.general.experiment_name,
                save_dir=cfg.general.save_dir,
                id=cfg.general.experiment_name,
                entity="rwth-data-science",
            )
        )
        loggers[-1].log_hyperparams(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))

    model = ObjectSegmentation(cfg)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


def train(cfg: DictConfig):
    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            verbose=True,
            save_top_k=1,
            save_last=True,
            monitor="mIoU_epoch",
            mode="max",
            dirpath=cfg.general.save_dir,
            every_n_epochs=1,
            filename="{epoch:02d}-{mIoU_epoch:.3f}",
            save_on_train_epoch_end=True,
        )
    )
    callbacks.append(LearningRateMonitor())
    callbacks.append(MemoryUsageLogger())

    runner = Trainer(
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=cfg.general.save_dir,
        devices=cfg.trainer.num_devices,
        num_nodes=cfg.trainer.num_nodes,
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        detect_anomaly=cfg.trainer.detect_anomaly,
        strategy="ddp_find_unused_parameters_false",
        profiler="simple",
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    runner.fit(model, ckpt_path=cfg.general.ckpt_path)


def validate(cfg: DictConfig):
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        logger=loggers,
        default_root_dir=cfg.general.save_dir,
        devices=1,
        num_nodes=1,
        accelerator="gpu",
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        limit_val_batches=cfg.trainer.limit_train_batches,
    )

    runner.validate(model, ckpt_path=cfg.general.ckpt_path)


def test(cfg: DictConfig):
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=1,
        default_root_dir=str(cfg.general.save_dir),
    )
    runner.test(model=model, ckpt_path=cfg.general.ckpt_path)


def main(running_on_julich=False):
    # Load the configuration from the YAML file
    cfg = OmegaConf.load("config.yaml")
    # cfg = OmegaConf.load("config_validation.yaml")

    if running_on_julich:
        cfg.data.datasets.data_dir = "/p/scratch/objectsegvideo/ilya/code/preprocessing"
        cfg.trainer.num_devices = 4
        cfg.trainer.num_nodes = 4

    cfg.general.experiment_name = cfg.general.experiment_name.replace("now", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(resolved_cfg)
    cfg = OmegaConf.create(resolved_cfg)

    parser = argparse.ArgumentParser(description="Override config parameters.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    if args.debug:
        cfg.general.experiment_name = "debugging-with-logging"

    if cfg["general"]["mode"] == "train":
        train(cfg)
    elif cfg["general"]["mode"] == "validate":
        validate(cfg)
    elif cfg["general"]["mode"] == "test":
        test(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg['general']['mode']}")


if __name__ == "__main__":
    running_on_julich = False
    main(running_on_julich)
