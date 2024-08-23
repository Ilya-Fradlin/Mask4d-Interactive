import numpy as np
import volumentations as V
import csv
import yaml
import json
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Union
from random import choice, uniform, sample as random_sample, random

from datasets.kitti360_info import map_labels_kitti360, is_thing_kitti360


class LidarDataset_Kitti360(Dataset):
    def __init__(
        self,
        data_dir: Optional[str] = "/globalwork/fradlin/mask4d-interactive/processed/semantic_kitti",
        mode: Optional[str] = "train",
        sample_choice: Optional[str] = "full",
        add_distance: Optional[bool] = False,
        ignore_label: Optional[Union[int, List[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        instance_population: Optional[int] = 0,
        sweep: Optional[int] = 1,
        segment_full_scene=True,
        obj_type="all",
        center_coordinates=False,
    ):
        super(LidarDataset_Kitti360, self).__init__()

        self.mode = mode
        self.data_dir = data_dir
        self.ignore_label = ignore_label
        self.add_distance = add_distance
        self.instance_population = instance_population
        self.sweep = 1
        self.segment_full_scene = segment_full_scene
        self.obj_type = obj_type
        self.center_coordinates = center_coordinates
        self.config = self._load_yaml("conf/semantic-kitti.yaml")
        if sweep >= 1:
            self.drop_outliers = True
        else:
            self.drop_outliers = False

        # loading database file
        database_file = Path("/nodes/veltins/work/fradlin/jsons/kitti360_single_scans_validation.json")
        if not database_file.exists():
            print(f"generate {database_file}")
            exit()
        with open(database_file) as json_file:
            self.data = json.load(json_file)

        # reformulating in sweeps
        data = [[]]
        scene_names = list(self.data.keys())
        last_scene = self.data[scene_names[0]]["scene"]
        for scene_name in scene_names:
            x = self.data[scene_name]  # get the actual sample from the dictionary
            if x["scene"] == last_scene:
                data[-1].append(x)
            else:
                last_scene = x["scene"]
                data.append([x])
        for i in range(len(data)):
            data[i] = list(self.chunks(data[i], sweep))
        self.data = [val for sublist in data for val in sublist]

        # augmentations
        self.volume_augmentations = V.NoOp()
        if volume_augmentations_path is not None:
            self.volume_augmentations = V.load(volume_augmentations_path, data_format="yaml")

    def chunks(self, lst, n):
        if "train" in self.mode or n == 1:
            for i in range(len(lst) - n + 1):
                yield lst[i : i + n]
        else:
            # Non-overlapping chunks
            for i in range(0, len(lst) - n, n):
                yield lst[i : i + n]
            # Ensure the last chunk is also of size n, taking the last n elements
            yield lst[-n:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        coordinates_list = []
        features_list = []
        labels_list = []
        acc_num_points = []
        num_obj_in_scene = []
        obj2label_maps_list = []
        file_paths = []

        # for debugging can specify idx = 1397 (for scene 1397)
        label2obj_map, obj2label_map, click_idx, max_instance_id = {}, {}, {}, 0
        for time, scan in enumerate(self.data[idx]):
            file_paths.append(scan["filepath"])
            # points = x , y, z, intensity, time, r, g, b
            points = np.fromfile(scan["filepath"], dtype=np.float32).reshape(-1, 8)

            coordinates = points[:, :3]
            features = points[:, 3:4]  # intensity,
            time_array = points[:, 4:5]  # time
            colors = points[:, 5:8]  # RGB
            features = np.hstack((time_array, features))  # (time, intensity)

            # labels
            if "test" in self.mode:
                labels = np.zeros_like(features).astype(np.int64)
                obj2label_maps_list.append({})
            else:
                # Convert the panoptic labels into object labels
                labels, obj2label_map, click_idx, max_instance_id, label2obj_map = self.generate_object_labels(scan, max_instance_id, label2obj_map, obj2label_map, click_idx)
                obj2label_maps_list.append(obj2label_map)
                unique_labels = np.unique(labels)
                unique_labels = unique_labels[unique_labels != 0]
                num_obj_in_scene.append(len(unique_labels))

            if self.drop_outliers:
                # Create a boolean mask where labels are not 0
                mask = labels != 0
                # Apply the mask to each array
                labels = labels[mask]
                coordinates = coordinates[mask]
                features = features[mask]

            acc_num_points.append(len(labels))

            labels_list.append(labels)
            coordinates_list.append(coordinates)
            features_list.append(features)

        coordinates = np.vstack(coordinates_list)
        if self.center_coordinates:
            coordinates -= coordinates.mean(0)
        features = np.vstack(features_list)
        labels = np.hstack(labels_list)

        # Enrich the features with the distance to the center
        if self.add_distance:
            center_coordinate = coordinates.mean(0)
            features = np.hstack(
                (
                    features,
                    np.linalg.norm(coordinates - center_coordinate, axis=1)[:, np.newaxis],
                )  #  now features include: (time, intensity, distance)
            )

        # volume and image augmentations for train
        if "train" in self.mode:
            coordinates -= coordinates.mean(0)
            if 0.5 > random():
                coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
            aug = self.volume_augmentations(points=coordinates)
            coordinates = aug["points"]

        features = np.hstack((coordinates, features))

        return {
            "sequence": file_paths,
            "num_points": acc_num_points,
            "num_obj": num_obj_in_scene,
            "coordinates": coordinates,
            "features": features,  # (coordinates, time, intensity, distance)
            "labels": labels,
            "click_idx": click_idx,
            "obj2label": obj2label_maps_list,
        }

    def generate_object_labels(self, scan, max_instance_id, label2obj_map, obj2label_map, click_idx):
        """
        panoptic_labels = semantic_labels*1000 + instance_labels
        semantic_labels = panoptic_labels // 1000
        instance_labels = panoptic_labels % 1000
        """
        panoptic_labels = np.fromfile(scan["label_filepath"], dtype=np.uint32)
        current_max_instance_id = np.amax(panoptic_labels % 1000)
        if current_max_instance_id > max_instance_id:
            max_instance_id = current_max_instance_id

        instance_labels = panoptic_labels % 1000
        # Extract and update semantic labels
        semantic_labels = panoptic_labels // 1000

        # updated_semantic_labels = np.vectorize(map_labels_kitti360)(semantic_labels)
        # updated_semantic_labels = np.vectorize(self.config["learning_map"].__getitem__)(semantic_labels)

        panoptic_labels = semantic_labels * 1000 + instance_labels

        if "validation" in self.mode and not self.segment_full_scene:
            if self.obj_type != "all":
                # we need to keep only the things or only the stuff
                scan["clicks"], scan["obj"] = self.select_only_desired_objects_subsampled(self.obj_type, scan["obj"], scan["clicks"])
            obj_labels = np.zeros(panoptic_labels.shape)
            for obj_idx, label in scan["obj"].items():
                obj_labels[panoptic_labels == label] = int(obj_idx)
                obj2label_map[str(int(obj_idx))] = int(label)
            click_idx = scan["clicks"]

        elif ("train" in self.mode) or ("validation" in self.mode and self.segment_full_scene):
            # no pre-defined object selected, choose random objects
            obj_labels = np.zeros(panoptic_labels.shape)
            unique_panoptic_labels = scan["unique_panoptic_labels"]
            if self.obj_type != "all":
                unique_panoptic_labels = self.select_only_desired_objects(self.obj_type, unique_panoptic_labels)

            max_num_obj = len(unique_panoptic_labels)
            if self.segment_full_scene:
                num_obj = max_num_obj
            else:
                num_obj = np.random.randint(1, min(10, max_num_obj) + 1)
            chosen_objects = random_sample(unique_panoptic_labels, num_obj)

            if label2obj_map == {}:
                # This is the first scene we are generating object labels
                for obj_idx, label in enumerate(chosen_objects):
                    obj_idx += 1  # 0 is background
                    obj_labels[panoptic_labels == label] = int(obj_idx)
                    obj2label_map[str(int(obj_idx))] = int(label)
                    label2obj_map[label] = int(obj_idx)
                    click_idx[str(obj_idx)] = []
                # Background
                click_idx["0"] = []
            else:
                # We have already generated object labels for previous scene in the sweep, now need to update for new object
                current_obj_idx = max(label2obj_map.values()) + 1  # In case there are new objects in the scene, add them as the following index
                for label in chosen_objects:
                    if label in label2obj_map.keys():
                        defined_obj_id = label2obj_map[label]
                        obj_labels[panoptic_labels == label] = int(defined_obj_id)
                    else:
                        # a new obj is introduced
                        obj2label_map[str(int(current_obj_idx))] = int(label)
                        label2obj_map[label] = int(current_obj_idx)
                        obj_labels[panoptic_labels == label] = int(current_obj_idx)
                        click_idx[str(current_obj_idx)] = []
                        current_obj_idx += 1

        else:
            raise ValueError(f"{self.mode} should not be used generate_object_labels")

        return obj_labels, obj2label_map, click_idx, max_instance_id, label2obj_map

    def select_only_desired_objects(self, obj_type, unique_panoptic_labels):
        things_targets = [1, 2, 3, 4, 5, 6, 7, 8]  # [1:car,  2:bicycle,  3:motorcycle,  4:truck,  5:other-vehicle,  6:person,  7:bicyclist,  8:motorcyclist ]
        stuff_targets = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  #  [9:road,  10:parking, 11:sidewalk ,12:other-ground ,13:building ,14:fence ,15:vegetation ,16:trunk ,17:terrain ,18:pole ,19:traffic-sign]
        things_labels = [label for label in unique_panoptic_labels if label & 0xFFFF in things_targets]
        stuff_labels = [label for label in unique_panoptic_labels if label & 0xFFFF in stuff_targets]
        desired_objects = {"things": things_labels, "stuff": stuff_labels}.get(obj_type)
        if desired_objects is None:
            raise ValueError(f"Unknown obj_type {obj_type}")
        return desired_objects

    def select_only_desired_objects_subsampled(self, obj_type, selected_obj):
        click_idx = {}
        updated_obj = {}
        obj_idx = 1  # 0 is background
        things_targets = [1, 2, 3, 4, 5, 6, 7, 8]  # [1:car,  2:bicycle,  3:motorcycle,  4:truck,  5:other-vehicle,  6:person,  7:bicyclist,  8:motorcyclist ]
        stuff_targets = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  #  [9:road,  10:parking, 11:sidewalk ,12:other-ground ,13:building ,14:fence ,15:vegetation ,16:trunk ,17:terrain ,18:pole ,19:traffic-sign]
        for label in selected_obj.values():
            semantic_id = label & 0xFFFF
            if (obj_type == "things" and semantic_id in things_targets) or (obj_type == "stuff" and semantic_id in stuff_targets):
                updated_obj[str(obj_idx)] = label
                click_idx[str(obj_idx)] = []
                obj_idx += 1
        # Background
        click_idx["0"] = []

        return click_idx, updated_obj

    def augment(self, point_cloud):
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:, 0] = -1 * point_cloud[:, 0]

        if np.random.random() > 0.5:
            # Flipping along the XZ plane
            point_cloud[:, 1] = -1 * point_cloud[:, 1]

        # Rotation along up-axis/Z-axis
        rot_angle_pre = np.random.choice([0, np.pi / 2, np.pi, np.pi / 2 * 3])
        rot_mat_pre = self.rotz(rot_angle_pre)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat_pre))

        rot_angle = (np.random.random() * 2 * np.pi) - np.pi  # -180 ~ +180 degree
        rot_mat = self.rotz(rot_angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))

        return point_cloud

    def rotz(self, t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file

    def _select_correct_labels(self, learning_ignore):
        count = 0
        label_info = dict()
        for k, v in learning_ignore.items():
            if v:
                label_info[k] = self.ignore_label
            else:
                label_info[k] = count
                count += 1
        return label_info
