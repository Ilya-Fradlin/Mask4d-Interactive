import re
import random
import json
import os
import numpy as np
import yaml
from pathlib import Path
from natsort import natsorted
from loguru import logger
from tqdm import tqdm
from fire import Fire

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from scipy.spatial.transform import Rotation
from nuscenes.utils.data_io import load_bin_file

from pyquaternion import Quaternion


class SemanticKittiPreprocessing:
    def __init__(
        self,
        data_dir: str = "/globalwork/datasets/nuScenes-v1.0/",
        save_dir: str = "/home/fradlin/Github/Mask4D-Interactive/datasets",
        modes: tuple = ["validation"],  #  "train", "test"
        subsample_dataset: bool = False,
        version="v1.0-trainval",  # "v1.0-mini", "v1.0-test"
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.subsample_dataset = subsample_dataset
        self.modes = modes
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=self.data_dir, verbose=True)
        self.validation_scenes = create_splits_scenes()["val"]
        self.databases = {}

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError
        if self.save_dir.exists() is False:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        for mode in self.modes:
            self.databases[mode] = []
            for scene_name in tqdm(sorted(self.validation_scenes)):
                scene_token = self.nusc.field2token("scene", "name", scene_name)[0]
                scene_metadata = self.nusc.get("scene", scene_token)

                # Get the first and last sample tokens from the scene metadata
                # Then iterate through the samples using the first sample token and the 'next' field
                first_sample_token = scene_metadata["first_sample_token"]
                current_token = first_sample_token
                while current_token != "":
                    current_sample = self.nusc.get("sample", current_token)
                    lidar_data = self.nusc.get("sample_data", current_sample["data"]["LIDAR_TOP"])
                    calibration_token = lidar_data["calibrated_sensor_token"]
                    calibrated_sensor = self.nusc.get("calibrated_sensor", calibration_token)
                    # calibration = parse_calibration(calibrated_sensor)
                    ego_pose_token = lidar_data["ego_pose_token"]
                    ego_pose = self.nusc.get("ego_pose", ego_pose_token)
                    pcd_path = os.path.join(self.data_dir, lidar_data["filename"])
                    panoptic_data = self.nusc.get("panoptic", current_sample["data"]["LIDAR_TOP"])
                    panoptic_labels_path = panoptic_data["filename"]
                    label_path = os.path.join("/globalwork/datasets/nuScenes-v1.0/panoptic/nuScenes-panoptic-v1.0-all", panoptic_labels_path)

                    sample_dict = {}
                    sample_dict["sample_token"] = current_token
                    sample_dict["filepath"] = pcd_path
                    sample_dict["label_filepath"] = label_path
                    sample_dict["unique_panoptic_labels"] = read_labels(label_path)[1]
                    pose_matrix = transform_matrix(ego_pose["translation"], Quaternion(ego_pose["rotation"]), inverse=True)
                    sample_dict["pose"] = parse_poses(ego_pose, calibrated_sensor)
                    sample_dict["scene"] = scene_name.replace("-", "_")
                    self.databases[mode].append(sample_dict)

                    current_token = current_sample["next"]

    def preprocess(self):
        # logger.info(f"starting preprocessing...")
        # for mode in self.modes:
        #     logger.info(f"Initializing {mode} database...")
        #     self.databases[mode] = []
        #     database = []
        #     for filepath in tqdm(self.files[mode], unit="file"):
        #         filebase = self.process_file(filepath, mode)
        #         if filebase is None:
        #             continue
        #         database.append(filebase)
        #     self.databases[mode] = database

        self.select_validation_objects()
        self.save_databases_as_json()

    def select_validation_objects(self):
        data = self.databases["validation"]
        data_updated = []
        logger.info("starting validation subsampling...")
        for sample in tqdm(data):
            # pick a random number between 1 and num_of_obj
            max_num_obj = len(sample["unique_panoptic_labels"])
            num_of_obj_to_validate = max_num_obj
            # num_of_obj_to_validate = np.random.randint(1, min(10, max_num_obj) + 1)

            ########### Pre-determine objects to be segmented in each validation scene ###########
            chosen_objects = random.sample(sample["unique_panoptic_labels"], num_of_obj_to_validate)
            clicks = {}
            obj = {}
            for obj_idx, label in enumerate(chosen_objects):
                clicks[str(obj_idx + 1)] = []
                obj[str(obj_idx + 1)] = label

            clicks[str(0)] = []  # added click for background

            sample["clicks"] = clicks
            sample["obj"] = obj

            data_updated.append(sample)

        self.databases["validation"] = data_updated

    def subsample_data(self):
        logger.info("beginning subsampling...")
        for mode in self.databases:
            data = self.databases[mode]
            if mode == "train":
                number_of_scences = 1200  # Agile3D had 1200 validation samples
            if mode == "validation":
                number_of_scences = 350  # Agile3D had 313 validation samples
            subsampled_data = subsample_uniformly(data, number_of_scences)
            self.databases[mode] = subsampled_data
        logger.info("finished subsampling...")

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

    def make_instance_database(self):
        train_database = self._load_yaml(self.save_dir / "train_database.yaml")
        instance_database = {}
        for sample in tqdm(train_database):
            instances = self.extract_instance_from_file(sample)
            for instance in instances:
                scene = instance["scene"]
                panoptic_label = instance["panoptic_label"]
                unique_identifier = f"{scene}_{panoptic_label}"
                if unique_identifier in instance_database:
                    instance_database[unique_identifier]["filepaths"].append(instance["instance_filepath"])
                else:
                    instance_database[unique_identifier] = {
                        "semantic_label": instance["semantic_label"],
                        "filepaths": [instance["instance_filepath"]],
                    }
        self.save_database(list(instance_database.values()), "train_instances")

        validation_database = self._load_yaml(self.save_dir / "validation_database.yaml")
        for sample in tqdm(validation_database):
            instances = self.extract_instance_from_file(sample)
            for instance in instances:
                scene = instance["scene"]
                panoptic_label = instance["panoptic_label"]
                unique_identifier = f"{scene}_{panoptic_label}"
                if unique_identifier in instance_database:
                    instance_database[unique_identifier]["filepaths"].append(instance["instance_filepath"])
                else:
                    instance_database[unique_identifier] = {
                        "semantic_label": instance["semantic_label"],
                        "filepaths": [instance["instance_filepath"]],
                    }
        self.save_database(list(instance_database.values()), "trainval_instances")

    def extract_instance_from_file(self, sample):
        points = np.fromfile(sample["filepath"], dtype=np.float32).reshape(-1, 4)
        pose = np.array(sample["pose"]).T
        points[:, :3] = points[:, :3] @ pose[:3, :3] + pose[3, :3]
        label = np.fromfile(sample["label_filepath"], dtype=np.uint32)
        scene, sub_scene = re.search(r"(\d{2}).*(\d{6})", sample["filepath"]).group(1, 2)
        file_instances = []
        for panoptic_label in np.unique(label):
            semantic_label = panoptic_label & 0xFFFF
            semantic_label = np.vectorize(self.config["learning_map"].__getitem__)(semantic_label)
            if np.isin(semantic_label, range(1, 9)):
                instance_mask = label == panoptic_label
                instance_points = points[instance_mask, :]
                filename = f"{scene}_{panoptic_label:010d}_{sub_scene}.npy"
                instance_filepath = self.save_dir / "instances" / filename
                instance = {
                    "scene": scene,
                    "sub_scene": sub_scene,
                    "panoptic_label": f"{panoptic_label:010d}",
                    "instance_filepath": str(instance_filepath),
                    "semantic_label": semantic_label.item(),
                }
                if not instance_filepath.parent.exists():
                    instance_filepath.parent.mkdir(parents=True, exist_ok=True)
                np.save(instance_filepath, instance_points.astype(np.float32))
                file_instances.append(instance)
        return file_instances

    def save_database(self, database, mode):
        for element in database:
            self._dict_to_yaml(element)
        self._save_yaml(self.save_dir / (mode + "_database.yaml"), database)

    def joint_database(self, train_modes=["train", "validation"]):
        joint_db = []
        for mode in train_modes:
            joint_db.extend(self._load_yaml(self.save_dir / (mode + "_database.yaml")))
        self._save_yaml(self.save_dir / "trainval_database.yaml", joint_db)

    @classmethod
    def _save_yaml(cls, path, file):
        with open(path, "w") as f:
            yaml.safe_dump(file, f, default_style=None, default_flow_style=False)

    @classmethod
    def _dict_to_yaml(cls, dictionary):
        if not isinstance(dictionary, dict):
            return
        for k, v in dictionary.items():
            if isinstance(v, dict):
                cls._dict_to_yaml(v)
            if isinstance(v, np.ndarray):
                dictionary[k] = v.tolist()
            if isinstance(v, Path):
                dictionary[k] = str(v)

    @classmethod
    def _load_yaml(cls, filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file

    def create_label_database(self, config_file):
        if (self.save_dir / "label_database.yaml").exists():
            return self._load_yaml(self.save_dir / "label_database.yaml")
        config = self._load_yaml(config_file)
        label_database = {}
        for key, old_key in config["learning_map_inv"].items():
            label_database.update(
                {
                    key: {
                        "name": config["labels"][old_key],
                        "color": config["color_map"][old_key][::-1],
                        "validation": not config["learning_ignore"][key],
                    }
                }
            )

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def process_file(self, filepath, mode):
        scene, sub_scene = re.search(r"(\d{2}).*(\d{6})", filepath).group(1, 2)
        sample = {
            "filepath": filepath,
            "scene": int(scene),
            "pose": self.pose[mode][int(scene)][int(sub_scene)].tolist(),
        }

        if mode in ["train", "validation"]:
            # getting label info
            label_filepath = filepath.replace("velodyne", "labels").replace("bin", "label")
            sample["label_filepath"] = label_filepath
            config_path = self.data_dir / "semantic-kitti.yaml"
            _, unique_panoptic_labels, number_of_things, number_of_stuff = read_labels(label_filepath, self.config)
            if (number_of_things + number_of_stuff) == 0:
                return None
            sample["unique_panoptic_labels"] = unique_panoptic_labels
            # sample["number_of_things"] = number_of_things
            # sample["number_of_stuff"] = number_of_stuff
            # sample["number_of_objects"] = number_of_things + number_of_stuff

        return sample


###############################################################################
############################# Utility Functions ###############################
###############################################################################


def read_labels(file_path):
    """
    Read labels from a file.

    Args:
        file_path (str): The path to the file containing the labels.

    Returns:
        tuple: A tuple containing the following elements:
            - labels (numpy.ndarray): An array of labels.
            - number_of_things (int): The number of unique things in the labels.
            - unique_things_labels (list): A list of unique labels for things.

    """
    panoptic_labels = np.load(file_path)["data"]
    # semantic_label = panoptic_labels // 1000
    instance_label = panoptic_labels % 1000

    unique_things_labels = np.unique(panoptic_labels[instance_label != 0])
    unique_stuff_labels = np.unique(panoptic_labels[instance_label == 0])
    unique_stuff_labels = unique_stuff_labels[unique_stuff_labels != 0]  # Drop the 0 label
    unique_panoptic_labels = unique_things_labels.tolist() + unique_stuff_labels.tolist()
    number_of_stuff = len(unique_stuff_labels)
    number_of_things = len(unique_things_labels)

    return panoptic_labels, unique_panoptic_labels, number_of_things, number_of_stuff


def parse_calibration(calibrated_sensor):
    # NuScenes pose representation (rotation quaternion and translation vector)
    rotation_quaternion = calibrated_sensor["rotation"]
    translation_vector = calibrated_sensor["translation"]
    rotation_matrix = Rotation.from_quat(rotation_quaternion).as_matrix()
    # Construct a 4x4 transformation matrix
    calibration = np.eye(4)
    calibration[:3, :3] = rotation_matrix
    calibration[:3, 3] = translation_vector

    # Append a row of [0, 0, 0, 1] to make it homogeneous
    calibration[3] = [0, 0, 0, 1]

    return calibration


def parse_poses(ego_pose, ego_calibration):
    # NuScenes pose representation (rotation quaternion and translation vector)
    pose = transform_matrix(ego_pose["translation"], Quaternion(ego_pose["rotation"]), inverse=False)
    calibration = transform_matrix(ego_calibration["translation"], Quaternion(ego_calibration["rotation"]), inverse=False)
    calibration_inv = transform_matrix(ego_calibration["translation"], Quaternion(ego_calibration["rotation"]), inverse=True)

    calibrated_pose = np.matmul(calibration_inv, np.matmul(pose, calibration))
    return calibrated_pose.tolist()


def subsample_uniformly(data, desired_samples):
    total_items = len(data)
    step_size = max(1, int(total_items / desired_samples))
    logger.info(f"total items: {total_items}, step size: {step_size}")
    # Use list slicing to pick items at regular intervals
    subsampled_data = data[::step_size]
    # If we got more samples than desired due to rounding, trim the list
    if len(subsampled_data) > desired_samples:
        subsampled_data = subsampled_data[:desired_samples]
    return subsampled_data


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]), rotation: Quaternion = Quaternion([1, 0, 0, 0]), inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


if __name__ == "__main__":
    Fire(SemanticKittiPreprocessing)
