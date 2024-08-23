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
        data_dir: str = "/nodes/anchorbrew/work/yilmaz/KITTI-360/data_3d_semantics/train/",
        label_dir: str = "/work/fradlin/KITTI360SingleScan",
        save_dir: str = "/nodes/veltins/work/fradlin/jsons",
        modes: tuple = ["validation"],  # "test"
        subsample_dataset: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.save_dir = Path(save_dir)
        self.validation_split_file = os.path.join(self.data_dir, "2013_05_28_drive_val.txt")
        self.subsample_dataset = subsample_dataset
        self.modes = modes
        self.databases = {}

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError
        if self.save_dir.exists() is False:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # poses_path = "/nodes/anchorbrew/work/yilmaz/KITTI-360/data_poses/"
        poses_path = str(self.data_dir).replace("data_3d_semantics/train", "data_poses")
        self.poses = {}
        self.timestamp = {}
        for scene in os.listdir(poses_path):
            self.poses[scene] = {}
            poses_file = os.path.join(poses_path, scene, "poses.txt")
            with open(poses_file, "r") as file:
                for line in file:
                    if line:
                        # elements = line.split()
                        # pose_elements = list(map(float, elements[1:]))
                        # pose = np.zeros((4, 4))
                        # pose[0, 0:4] = pose_elements[0:4]
                        # pose[1, 0:4] = pose_elements[4:8]
                        # pose[2, 0:4] = pose_elements[8:12]
                        # pose[3, 3] = 1.0
                        lineProcessed = line.rstrip("\n").split(" ")
                        transform = np.eye(4)
                        index = int(lineProcessed[0])
                        for i in range(12):
                            xi = i // 4
                            yi = i % 4
                            transform[xi, yi] = float(lineProcessed[i + 1])
                        self.poses[scene][int(line.split()[0])] = transform.tolist()

            self.timestamp[scene] = {}
            timestamp_file = f"/nodes/anchorbrew/work/yilmaz/KITTI-360/data_3d_raw/{scene}/velodyne_points/timestamps.txt"
            scan_counter = 0
            if not os.path.exists(timestamp_file):
                print(f"Timestamp file does not exist: {timestamp_file}")
                continue

            with open(timestamp_file, "r") as file:
                for line in file:
                    line = line.strip()
                    if line:
                        self.timestamp[scene][scan_counter] = line
                        scan_counter += 1
                    else:
                        raise ValueError("Empty line in timestamp file")

        with open(self.validation_split_file, "r") as file:
            self.validation_chunks = file.readlines()
            self.validation_chunks = [scene.strip() for scene in self.validation_chunks]

        self.files = {}
        for data_type in self.modes:
            self.files.update({data_type: []})

        mode = "validation"
        counter = 0
        for chunk in sorted(self.validation_chunks):
            current_scene = chunk.split("/")[2]
            velodyne_dir = f"/nodes/anchorbrew/work/yilmaz/KITTI-360/data_3d_raw/{current_scene}/velodyne_points/data/"
            single_label_dir = f"/work/fradlin/KITTI360SingleScan/{current_scene}/labels/"
            chunk_ranges = chunk.split("/")[4].split(".")[0]
            start_str, end_str = chunk_ranges.split("_")
            start_num = int(start_str)
            end_num = int(end_str)
            for num in range(start_num, end_num + 1):
                scan = f"{num:010d}"  # Format as a 10-digit number with leading zeros
                single_label_path = os.path.join(single_label_dir, f"{scan}.bin")
                if os.path.exists(single_label_path):
                    raw_scan_path = os.path.join(velodyne_dir, f"{scan}.bin")
                    assert os.path.exists(raw_scan_path), f"File does not exist: {raw_scan_path}"
                    if raw_scan_path not in self.files[mode]:
                        self.files[mode].append(raw_scan_path)
                    else:
                        print(f"File already exists in database: {raw_scan_path}")
                else:
                    counter += 1
                    print(f"File does not exist: {single_label_path}")

        print(f"Number of missing files: {counter}")

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

        self.save_databases_as_json()

    def save_databases_as_json(self):
        """
        Save the databases as JSON files.

        This method saves the data in the databases attribute as JSON files for each mode.
        The JSON files are named based on the mode and whether the data is subsampled or not.

        Returns:
            None
        """
        # Save data as JSON for each mode
        for mode in self.modes:
            data = self.databases[mode]
            json_data = {}
            for item in data:
                # Construct the key based on the filepath
                filepath = item["filepath"]
                scene_id = item["scene"]
                scan = item["scan"]
                key = f"scene_{scene_id}_{scan}"
                json_data[key] = item

            # Determine file name based on mode
            val_json_file_name = os.path.join(self.save_dir, "kitti360_single_scans_validation.json")
            with open(val_json_file_name, "w") as file:
                json_data = json.dumps(json_data, indent=2)
                file.write(json_data)

    def process_file(self, filepath, mode):
        scene, scan = filepath.split("/")[-4], (filepath.split("/")[-1]).split(".")[0]
        sample = {
            "filepath": filepath,
            "scene": scene,
            "scan": scan,
        }

        if mode in ["validation"]:
            # getting label info
            scene = filepath.split("/")[-4]
            scan = (filepath.split("/")[-1]).split(".")[0]
            label_filepath = os.path.join(self.label_dir, scene, "labels", f"{scan}.bin")
            assert os.path.exists(label_filepath), f"Label file does not exist: {label_filepath}"

            panoptic_labels, unique_panoptic_labels, number_of_objects, number_of_points = read_labels(label_filepath)
            sample["label_filepath"] = label_filepath
            sample["unique_panoptic_labels"] = [int(label) for label in unique_panoptic_labels]
            sample["number_of_points"] = number_of_points
            sample["number_of_objects"] = number_of_objects

            sample["poses"] = self.poses[scene][int(scan)]
            sample["timestamp"] = self.timestamp[scene][int(scan)]

            point_cloud = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
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
    panoptic_labels = np.fromfile(file_path, dtype=np.int64)
    # Extract semantic labels
    semantic_labels = panoptic_labels // 1000
    instance_labels = panoptic_labels % 1000

    unique_panoptic_labels = np.unique(panoptic_labels)
    number_of_points = panoptic_labels.shape[0]

    return panoptic_labels, unique_panoptic_labels, len(unique_panoptic_labels), number_of_points


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
