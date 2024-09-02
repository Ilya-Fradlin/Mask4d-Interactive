# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------

from omegaconf import DictConfig, OmegaConf
import argparse
import torch
from interactive_tool.utils import *
from interactive_tool.interactive_segmentation_user import UserInteractiveSegmentationModel
from interactive_tool.dataloader import InteractiveDataLoader


def main(config: DictConfig):
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    dataloader_test = InteractiveDataLoader(config)
    inseg_model_class = UserInteractiveSegmentationModel(device, config, dataloader_test)
    print(f"Using {device}")
    inseg_model_class.run_segmentation()


if __name__ == "__main__":
    config = OmegaConf.load("interactive_tool/config_interactive.yaml")
    parser = argparse.ArgumentParser(description="Override config parameters.")
    parser.add_argument("--user_name", type=str, default="user_00")
    parser.add_argument("--point_type", type=str, default="pointcloud", help="choose between 'mesh' and 'pointcloud'. If not given, the type will be determined automatically")
    args = parser.parse_args()

    config.dataset_scenes = config.data.datasets.data_dir
    config.user_name = args.user_name
    config.point_type = args.point_type
    main(config)


# python run_UI.py --user_name="Fradlin"
