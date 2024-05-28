import MinkowskiEngine as ME
import torch
import hydra
import pytorch_lightning
import numpy as np
from sklearn.cluster import DBSCAN
import utils.misc as utils
from utils.utils import generate_wandb_objects3d
from utils.seg import mean_iou, mean_iou_scene, cal_click_loss_weights, extend_clicks, get_simulated_clicks
from models.metrics.utils import IoU_at_numClicks, NumClicks_for_IoU
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    print("Imports completed")
    print("cuda is available: ", torch.cuda.is_available())


if __name__ == "__main__":
    main()
