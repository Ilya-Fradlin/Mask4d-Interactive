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
import MinkowskiEngine as ME


class KITTI360Preprocessing:
    def __init__(
        self,
        data_dir: str = "/nodes/veltins/work/fradlin/KITTI-360-4D",
        save_dir: str = "/nodes/veltins/work/fradlin/jsons",
        modes: tuple = ["validation"],  # "test"
        subsample_dataset: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.subsample_dataset = subsample_dataset
        self.modes = modes
        self.databases = {}

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError
        if self.save_dir.exists() is False:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.files = {}
        for data_type in self.modes:
            self.files.update({data_type: []})

        for mode in self.modes:
            scene_mode = "valid" if mode == "validation" else mode
            for scene in sorted(os.listdir(self.data_dir)):
                pattern = f"{scene}/velodyne/*.bin"
                filepaths = list(self.data_dir.glob(pattern))
                # filepaths = list(self.data_dir.glob(f"*/{scene:02}/velodyne/*bin"))
                filepaths = [str(file) for file in filepaths]
                self.files[mode].extend(natsorted(filepaths))

    def preprocess(self):
        logger.info(f"starting preprocessing...")
        for mode in self.modes:
            logger.info(f"Initializing {mode} database...")
            self.databases[mode] = []
            database = []
            for filepath in tqdm(self.files[mode], unit="file"):
                if "extra_tile" in filepath:
                    # don't process extra tiles
                    continue
                filebase = self.process_file(filepath, mode)
                if filebase is None:
                    continue
                database.append(filebase)
            self.databases[mode] = database
        logger.info(f"Finished initializing")

        if self.subsample_dataset:
            self.subsample_data()

        self.select_validation_objects()
        self.save_databases_as_json()

    def select_validation_objects(self):
        data = self.databases["validation"]
        data_updated = []
        logger.info("starting validation subsampling...")
        for sample in tqdm(data):
            # pick a random number between 1 and num_of_obj
            max_num_obj = len(sample["unique_panoptic_labels"])
            num_of_obj_to_validate = np.random.randint(1, min(10, max_num_obj) + 1)

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
        val_json_file_name = "kitti360_4d.json"
        # val_json_file_name = "subsampled_validation_list.json" if self.subsample_dataset else "full_validation_list.json"

        # Save data as JSON for each mode
        for mode in self.modes:
            data = self.databases[mode]
            json_data = {}
            for item in data:
                # Construct the key based on the filepath
                filepath = item["filepath"]
                scene_id = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
                filename = os.path.splitext(os.path.basename(filepath))[0]
                key = f"scene_{scene_id}_{filename}"
                json_data[key] = item

            # Determine file name based on mode
            json_file_name = val_json_file_name
            with open(os.path.join(self.save_dir, json_file_name), "w") as file:
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

    def save_database(self, database, mode):
        for element in database:
            self._dict_to_yaml(element)
        self._save_yaml(self.save_dir / (mode + "_database.yaml"), database)

    def process_file(self, filepath, mode):
        scene, sub_scene = filepath.split("/")[-3], (filepath.split("/")[-1]).split(".")[0]
        sample = {
            "filepath": filepath,
            "scene": sub_scene,
            "tile": sub_scene,
        }

        if mode in ["train", "validation"]:
            # getting label info
            label_filepath = filepath.replace("velodyne", "labels").replace("bin", "label")
            sample["label_filepath"] = label_filepath
            panoptic_labels, unique_panoptic_labels, number_of_things, number_of_stuff, number_of_points = read_labels(label_filepath)
            number_of_objects = number_of_things + number_of_stuff
            # if (number_of_objects) < 2 or number_of_points < 10_000:
            #     return None
            sample["unique_panoptic_labels"] = unique_panoptic_labels
            sample["number_of_points"] = number_of_points
            point_cloud = np.fromfile(filepath, dtype=np.float32).reshape(-1, 8)
            quantized_coords, quantized_features, quantized_labels = quantize_point_cloud(point_cloud, panoptic_labels, voxel_size=0.1)
            voxel_count = len(quantized_coords)
            sample["number_of_voxels"] = int(voxel_count)
            sample["number_of_objects"] = number_of_objects

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
    panoptic_labels = np.fromfile(file_path, dtype=np.uint32)
    # Extract semantic labels
    semantic_labels = panoptic_labels // 1000
    instance_labels = panoptic_labels % 1000

    unique_things_labels = np.unique(panoptic_labels[instance_labels != 0])
    unique_stuff_labels = np.unique(panoptic_labels[instance_labels == 0])

    unique_stuff_labels = unique_stuff_labels[unique_stuff_labels != 0]  # Drop the 0 label
    unique_panoptic_labels = unique_things_labels.tolist() + unique_stuff_labels.tolist()
    number_of_stuff = len(unique_stuff_labels)
    number_of_things = len(unique_things_labels)
    number_of_points = panoptic_labels.shape[0]

    return panoptic_labels, unique_panoptic_labels, number_of_things, number_of_stuff, number_of_points


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


def quantize_point_cloud(point_cloud, labels, voxel_size=0.1):
    coordinates = point_cloud[:, :3]
    features = point_cloud[:, 4:]  # Corrected to select the features part properly
    quantized_coords, quantized_features, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=coordinates,
        features=features,
        return_index=True,
        return_inverse=True,
        quantization_size=voxel_size,
    )
    quantized_labels = labels[unique_map]
    return quantized_coords, quantized_features, quantized_labels


if __name__ == "__main__":
    Fire(KITTI360Preprocessing)
