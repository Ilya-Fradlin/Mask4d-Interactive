import re
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
from fire import Fire
from loguru import logger
from natsort import natsorted
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from tqdm import tqdm

from datasets.utils import (
    load_yaml,
    merge_trainval,
    parse_calibration,
    parse_poses,
    points2image,
    save_database,
)

IMAGE_SHAPE = (900, 1600, 3)


class NuScenesPreprocessing:
    def __init__(
        self,
        data_dir: str = "/globalwork/yilmaz/data/raw/nuscenes",
        save_dir: str = "/globalwork/yilmaz/data/processed/nuscenes_new",
        generate_instances: bool = False,
        modes: tuple = ("train", "validation", "test"),
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.generate_instances = generate_instances
        self.modes = modes

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if generate_instances:
            self.instances_dir = self.save_dir / "instances"
            if not self.instances_dir.exists():
                self.instances_dir.mkdir(parents=True, exist_ok=True)

        self.nusc_trainval = NuScenes(version="v1.0-trainval", dataroot=data_dir)
        self.nusc_test = NuScenes(version="v1.0-test", dataroot=data_dir)
        scene_splits = create_splits_scenes()

        self.files = {}

        for mode in self.modes:
            self.files[mode] = []

        self.nusc = self.nusc_trainval
        for scene in self.nusc.scene:
            mode = "validation" if scene["name"] in scene_splits["val"] else "train"
            next_sample = scene["first_sample_token"]
            scan = 0
            while next_sample != "":
                sample = self.nusc.get("sample", next_sample)

                token = sample["data"]["LIDAR_TOP"]
                while token != "":
                    sample_data = self.nusc.get("sample_data", token)
                    token = sample_data["next"]
                    scan = scan + 1
                    print(sample_data["is_key_frame"])

                sensors = self.get_sensors(sample)

                label_filepath = self.nusc.get("panoptic", sample["data"]["LIDAR_TOP"])["filename"]

                sensors.update(
                    {
                        "sequence": scene["name"],
                        "scan": scan,
                        "label_filepath": label_filepath,
                    }
                )

                self.files[mode].append(sensors)

                next_sample = sample["next"]
                scan = scan + 1

        mode = "test"
        self.nusc = self.nusc_test
        for scene in self.nusc_test.scene:
            next_sample = scene["first_sample_token"]
            scan = 0
            while next_sample != "":
                sample = self.nusc.get("sample", next_sample)

                sensors = self.get_sensors(sample)

                sensors.update(
                    {
                        "sequence": scene["name"],
                        "scan": scan,
                    }
                )

                self.files[mode].append(sensors)

                next_sample = sample["next"]
                scan = scan + 1

    def preprocess(self):
        for mode in self.modes:
            if mode == "test":
                self.nusc = self.nusc_test
            else:
                self.nusc = self.nusc_trainval
            Path(self.save_dir / mode).mkdir(parents=True, exist_ok=True)
            database = []
            instance_database = {}
            for sensors in tqdm(self.files[mode], unit="file"):
                filebase = self.process_file(sensors, mode)
                database.append(filebase)
                if self.generate_instances and mode in ["train", "validation"]:
                    instances = self.extract_instance_from_file(filebase)
                    for instance in instances:
                        unique_identifier = f"{instance['sequence']}_{instance['panoptic_label']}"
                        if unique_identifier in instance_database:
                            instance_database[unique_identifier]["filepaths"].append(instance["instance_filepath"])
                        else:
                            instance_database[unique_identifier] = {
                                "semantic_label": instance["semantic_label"],
                                "filepaths": [instance["instance_filepath"]],
                            }
            save_database(database, mode, self.save_dir)
            if self.generate_instances and mode in ["train", "validation"]:
                save_database(list(instance_database.values()), f"{mode}_instances", self.save_dir)
        merge_trainval(self.save_dir, self.generate_instances)

    def process_file(self, sensors, mode):
        sequence, scan = sensors["sequence"], sensors["scan"]
        save_path = self.save_dir / mode / f"{sequence}_{scan}"
        filepath = self.data_dir / sensors["lidar"]["filename"]
        coords = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)[:, :3]

        lidar_pose = self.lidar_pose(sensors["lidar"])

        image_paths = []
        cam_calibs = []
        points_image_save_paths = []
        points_masks_save_paths = []
        for cam_id in range(6):
            image_paths.append(str(Path(self.data_dir) / sensors[f"cam_{cam_id}"]["filename"]))
            l2c, c_intrinsic = self.lidar_camera_calibration(sensors["lidar"], sensors[f"cam_{cam_id}"])
            cam_calibs.append(
                {
                    "distorted_img_K": c_intrinsic,
                    "D": [0, 0, 0, 0, 0],
                    "upper2cam": l2c,
                }
            )
            points_image, points_mask = points2image(coords, cam_calibs[-1], IMAGE_SHAPE, distort=True)

            points_image_save_path = f"{save_path}_points_image_camera_{cam_id}.npy"
            np.save(points_image_save_path, points_image.astype(np.float32))
            points_image_save_paths.append(points_image_save_path)

            points_masks_save_path = f"{save_path}_points_masks_camera_{cam_id}.npy"
            np.save(points_masks_save_path, points_mask)
            points_masks_save_paths.append(points_masks_save_path)

        filebase = {
            "filepath": str(filepath),
            "image_paths": image_paths,
            "points_image_paths": points_image_save_paths,
            "points_masks_paths": points_masks_save_paths,
            "sequence": sequence,
            "scan": scan,
            "cameras": cam_calibs,
            "pose": lidar_pose,
        }
        if mode in ["train", "validation"]:
            filebase["label_filepath"] = str(self.data_dir / sensors["label_filepath"])
        return filebase

    def lidar_camera_calibration(self, lidar_sensor, camera_sensor):
        lidar_calibration = self.nusc.get("calibrated_sensor", lidar_sensor["calibrated_sensor_token"])
        lidar2ego = self.calibration_to_transformation_matrix(lidar_calibration)

        lidar_ego_pose_calibration = self.nusc.get("ego_pose", lidar_sensor["ego_pose_token"])
        lidar_ego_pose = self.calibration_to_transformation_matrix(lidar_ego_pose_calibration)

        cam_ego_pose_calibration = self.nusc.get("ego_pose", camera_sensor["ego_pose_token"])
        cam_ego_pose_inv = self.calibration_to_transformation_matrix(cam_ego_pose_calibration, inverse=True)

        camera_calibration = self.nusc.get("calibrated_sensor", camera_sensor["calibrated_sensor_token"])
        camera2ego_inv = self.calibration_to_transformation_matrix(camera_calibration, inverse=True)

        lidar2camera = camera2ego_inv @ cam_ego_pose_inv @ lidar_ego_pose @ lidar2ego
        camera_intrinsic = np.array(camera_calibration["camera_intrinsic"])

        return lidar2camera.tolist(), camera_intrinsic.tolist()

    def lidar_pose(self, lidar_sensor):
        lidar_calibration = self.nusc.get("calibrated_sensor", lidar_sensor["calibrated_sensor_token"])
        lidar2ego = self.calibration_to_transformation_matrix(lidar_calibration)

        lidar_ego_pose_calibration = self.nusc.get("ego_pose", lidar_sensor["ego_pose_token"])
        lidar_ego_pose = self.calibration_to_transformation_matrix(lidar_ego_pose_calibration)

        lidar_pose = np.linalg.inv(lidar2ego) @ lidar_ego_pose @ lidar2ego

        return lidar_pose.tolist()

    def calibration_to_transformation_matrix(self, calibration, inverse=False):
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = Quaternion(calibration["rotation"]).rotation_matrix
        transformation_matrix[:3, 3] = calibration["translation"]
        if inverse:
            transformation_matrix = np.linalg.inv(transformation_matrix)
        return transformation_matrix

    def get_sensors(self, sample):
        lidar_sensor = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cam_front_sensor = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        cam_front_right_sensor = self.nusc.get("sample_data", sample["data"]["CAM_FRONT_RIGHT"])
        cam_back_right_sensor = self.nusc.get("sample_data", sample["data"]["CAM_BACK_RIGHT"])
        cam_back_sensor = self.nusc.get("sample_data", sample["data"]["CAM_BACK"])
        cam_back_left_sensor = self.nusc.get("sample_data", sample["data"]["CAM_BACK_LEFT"])
        cam_front_left_sensor = self.nusc.get("sample_data", sample["data"]["CAM_FRONT_LEFT"])

        return {
            "lidar": lidar_sensor,
            "cam_0": cam_front_sensor,
            "cam_1": cam_front_right_sensor,
            "cam_2": cam_back_right_sensor,
            "cam_3": cam_back_sensor,
            "cam_4": cam_back_left_sensor,
            "cam_5": cam_front_left_sensor,
        }

    # def extract_instance_from_file(self, filebase):
    #     sequence, scan = re.search(r"(\d{2}).*(\d{6})", filebase["filepath"]).group(
    #         1, 2
    #     )
    #     points = np.fromfile(filebase["filepath"], dtype=np.float32).reshape(-1, 4)
    #     pose = np.array(filebase["pose"]).T
    #     points[:, :3] = points[:, :3] @ pose[:3, :3] + pose[3, :3]
    #     labels = np.fromfile(filebase["label_filepath"], dtype=np.uint32)

    #     file_instances = []
    #     for panoptic_label in np.unique(labels):
    #         semantic_label = panoptic_label & 0xFFFF
    #         semantic_label = np.vectorize(self.config["learning_map"].__getitem__)(
    #             semantic_label
    #         )
    #         if not np.isin(semantic_label, self.config["stuff_cls_ids"]):
    #             instance_mask = labels == panoptic_label
    #             instance_points = points[instance_mask, :]
    #             filename = f"{sequence}_{panoptic_label:010d}_{scan}.npy"
    #             instance_filepath = self.instances_dir / filename
    #             instance = {
    #                 "sequence": sequence,
    #                 "panoptic_label": f"{panoptic_label:010d}",
    #                 "instance_filepath": str(instance_filepath),
    #                 "semantic_label": semantic_label.item(),
    #             }
    #             np.save(instance_filepath, instance_points.astype(np.float32))
    #             file_instances.append(instance)
    #     return file_instances


if __name__ == "__main__":
    nu = NuScenesPreprocessing()
    nu.preprocess()
