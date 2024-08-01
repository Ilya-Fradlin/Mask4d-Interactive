import os
import json
from pathlib import Path


import numpy as np
from loguru import logger
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from tqdm import tqdm

IMAGE_SHAPE = (900, 1600, 3)


class NuScenesPreprocessing:
    def __init__(
        self,
        data_dir: str = "/nodes/veltins/work/nuscenes",
        save_dir: str = "/home/fradlin/Github/Mask4D-Interactive/datasets/nuscenes",
        generate_instances: bool = False,
        modes: tuple = ("train", "validation"),
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
        # self.nusc_test = NuScenes(version="v1.0-test", dataroot=data_dir)
        scene_splits = create_splits_scenes()

        self.files = {}

        for mode in self.modes:
            self.files[mode] = []

        self.nusc = self.nusc_trainval
        for scene in self.nusc.scene:
            mode = "validation" if scene["name"] in scene_splits["val"] else "train"
            next_sample = scene["first_sample_token"]
            scan = 0
            # while next_sample != "":
            sample = self.nusc.get("sample", next_sample)
            token = sample["data"]["LIDAR_TOP"]
            while token != "":
                sample_data = self.nusc.get("sample_data", token)
                sensors = self.get_sensors(sample)
                label_filepath = self.nusc.get("panoptic", sample["data"]["LIDAR_TOP"])["filename"]
                sensors.update({"sequence": scene["name"], "scan": scan, "label_filepath": label_filepath, "is_key_frame": sample_data["is_key_frame"]})
                self.files[mode].append(sensors)
                next_sample = sample["next"]
                scan = scan + 1
                token = sample_data["next"]

        # mode = "test"
        # self.nusc = self.nusc_test
        # for scene in self.nusc_test.scene:
        #     next_sample = scene["first_sample_token"]
        #     scan = 0
        #     while next_sample != "":
        #         sample = self.nusc.get("sample", next_sample)

        #         sensors = self.get_sensors(sample)

        #         sensors.update(
        #             {
        #                 "sequence": scene["name"],
        #                 "scan": scan,
        #             }
        #         )

        #         self.files[mode].append(sensors)

        #         next_sample = sample["next"]
        #         scan = scan + 1

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
                # if self.generate_instances and mode in ["train", "validation"]:
                #     instances = self.extract_instance_from_file(filebase)
                #     for instance in instances:
                #         unique_identifier = f"{instance['sequence']}_{instance['panoptic_label']}"
                #         if unique_identifier in instance_database:
                #             instance_database[unique_identifier]["filepaths"].append(instance["instance_filepath"])
                #         else:
                #             instance_database[unique_identifier] = {
                #                 "semantic_label": instance["semantic_label"],
                #                 "filepaths": [instance["instance_filepath"]],
                #             }
            self.save_databases_as_json(database, mode, self.save_dir)
            if self.generate_instances and mode in ["train", "validation"]:
                self.save_databases_as_json(list(instance_database.values()), f"{mode}_instances", self.save_dir)

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
            # points_image, points_mask = points2image(coords, cam_calibs[-1], IMAGE_SHAPE, distort=True)

            # points_image_save_path = f"{save_path}_points_image_camera_{cam_id}.npy"
            # np.save(points_image_save_path, points_image.astype(np.float32))
            # points_image_save_paths.append(points_image_save_path)

            # points_masks_save_path = f"{save_path}_points_masks_camera_{cam_id}.npy"
            # np.save(points_masks_save_path, points_mask)
            # points_masks_save_paths.append(points_masks_save_path)

        filebase = {
            "filepath": str(filepath),
            "image_paths": image_paths,
            # "points_image_paths": points_image_save_paths,
            # "points_masks_paths": points_masks_save_paths,
            "sequence": sequence,
            "scan": scan,
            "cameras": cam_calibs,
            "pose": lidar_pose,
        }
        if mode in ["train", "validation"] and sensors["is_key_frame"]:
            filebase["label_filepath"] = str(self.data_dir / sensors["label_filepath"])
            filebase["is_key_frame"] = sensors["is_key_frame"]
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

    def save_databases_as_json(self):
        """
        Save the databases as JSON files.

        This method saves the data in the databases attribute as JSON files for each mode.
        The JSON files are named based on the mode and whether the data is subsampled or not.

        Returns:
            None
        """
        # train_json_file_name = "subsampled_train_list.json" if self.subsample_dataset else "full_train_list.json"
        val_json_file_name = "nuscenes_validation_list.json"

        # Save data as JSON for each mode
        for mode in self.modes:
            data = self.databases[mode]
            json_data = {}
            for item in data:
                # Construct the key based on the filepath
                scene_id = item["scene"]
                sample_token = item["sample_token"]
                key = f"{scene_id}_{sample_token}"
                json_data[key] = item

            # Determine file name based on mode
            # json_file_name = train_json_file_name if "train" in mode else val_json_file_name
            with open(os.path.join(self.save_dir, val_json_file_name), "w") as file:
                json_data = json.dumps(json_data, indent=2)
                file.write(json_data)


# def points2image(points, cam_calib, image_shape, distort=True):
#     H, W, _ = image_shape
#     cam_K = np.array(cam_calib["distorted_img_K"])
#     cam_D = np.array(cam_calib["D"])

#     # Transform points from upper lidar coordinates to camera coordinates
#     upper2cam = np.array(cam_calib["upper2cam"]).T
#     points_cam = points @ upper2cam[:3, :3] + upper2cam[3, :3]

#     # visible
#     depth = points_cam[:, 2]
#     visibility_mask = depth > 0

#     if distort:
#         points_image = points_cam[:, :2] / points_cam[:, 2:]
#         points_image = cam2img_distort(points_image, cam_K, cam_D)
#         else:
#             points_image = points_cam @ cam_K.T
#             points_image = points_image[:, :2] / points_image[:, 2:]

#             visibility_mask = np.logical_and(visibility_mask, np.all(points_image >= 0, axis=1))
#             visibility_mask = np.logical_and(visibility_mask, points_image[:, 0] < W)
#             visibility_mask = np.logical_and(visibility_mask, points_image[:, 1] < H)

#         return points_image[visibility_mask], visibility_mask

# def cam2img_distort(
#     points2d: np.ndarray,
#     cam_K: np.ndarray, # 3x3
#     cam_D: np.ndarray, # distortion vector
#     ) -> np.ndarray:
#     """
#     Apply radial distortion to the normalized points using the distortion coefficients.
#     https://github.com/JRDB-dataset/jrdb_toolkit/blob/main/tutorials/tutorials.ipynb
#     """
#     x, y = points2d[:, 0], points2d[:, 1]
#     k1, k2, k3, k4, k5 = cam_D
#     k6 = 0 # Assuming no 6th distortion coefficient.
#     r2 = x2 + y2
#     r4 = x4 + y4
#     r6 = x6 + y6
#     fx, _, cx = cam_K[0]
#     _, fy, cy = cam_K[1]
#     rad_dist = (1 + k1 r2 + k2 r4 + k3 r6) / (1 + k4 r2 + k5 r4 + k6 r6)
#     xd = fx rad_dist x + cx
#     yd = fy rad_dist y + cy

#     return np.stack([xd, yd], axis=1)


if __name__ == "__main__":
    nu = NuScenesPreprocessing()
    nu.preprocess()
