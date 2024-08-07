import numpy as np
import volumentations as V
import yaml
import json
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Union
from random import choice, uniform, sample as random_sample, random


class LidarDatasetNuscenes(Dataset):
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
        dataset_type="nuScenes_general",
    ):
        super(LidarDatasetNuscenes, self).__init__()

        self.mode = mode
        self.data_dir = "datasets/preprocessing"
        self.ignore_label = ignore_label
        self.add_distance = add_distance
        self.instance_population = instance_population
        self.sweep = sweep
        self.segment_full_scene = segment_full_scene
        self.obj_type = obj_type
        self.center_coordinates = center_coordinates
        self.dataset_type = dataset_type
        if sweep > 1:
            self.drop_outliers = True
        else:
            self.drop_outliers = False
        # loading database file
        database_path = Path(self.data_dir)
        database_file = database_path.joinpath(f"nuscenes_validation_list.json")
        if not database_file.exists():
            print(f"generate {database_file}")
            exit()
        with open(database_file) as json_file:
            self.data = json.load(json_file)

        # Id evaluating thing only- remove scenes with no things
        if obj_type == "things":
            # TODO: Implement this
            raise NotImplementedError("This feature is not implemented yet")

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
        if instance_population > 0:
            # self.instance_data = self._load_yaml(database_path / f"{mode}_instances_database.yaml")
            self.instance_data = self._load_yaml("/globalwork/fradlin/data/processed/semantic_kitti/trainval_instances_database.yaml")

    def chunks(self, lst, n):
        if "train" in self.mode or n == 1:
            for i in range(len(lst) - n + 1):
                yield lst[i : i + n]
        else:
            for i in range(0, len(lst) - n + 1, n - 1):
                yield lst[i : i + n]
            if i != len(lst) - n:
                yield lst[i + n - 1 :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        coordinates_list = []
        features_list = []
        labels_list = []
        acc_num_points = [0]
        obj2label_maps_list = []

        # for debugging can specify idx = 1397 (for scene 1397)
        label2obj_map, obj2label_map, click_idx, max_instance_id = {}, {}, {}, 0
        for time, scan in enumerate(self.data[idx]):
            # points = np.fromfile(scan["filepath"], dtype=np.float32).reshape(-1, 4)
            points, features = read_bin_point_cloud_nuscene(scan["filepath"])
            coordinates = points[:, :3]
            # rotate and translate
            pose = np.array(scan["pose"]).T
            coordinates = coordinates @ pose[:3, :3] + pose[3, :3]
            acc_num_points.append(acc_num_points[-1] + len(coordinates))

            # features
            # features = points[:, 3:4]  # intensity
            time_array = np.ones((features.shape[0], 1)) * time
            features = np.hstack((time_array, features))  # (time, intensity)

            # labels
            if "test" in self.mode:
                labels = np.zeros_like(features).astype(np.int64)
                obj2label_maps_list.append({})
            else:
                # Convert the panoptic labels into object labels
                labels, obj2label_map, click_idx, max_instance_id, label2obj_map = self.generate_object_labels(scan, max_instance_id, label2obj_map, obj2label_map, click_idx)
                obj2label_maps_list.append(obj2label_map)

            if self.drop_outliers:
                # Create a boolean mask where labels are not 0
                mask = labels != 0
                # Apply the mask to each array
                labels = labels[mask]
                coordinates = coordinates[mask]
                features = features[mask]
            coordinates_list.append(coordinates)
            features_list.append(features)
            labels_list.append(labels)

        coordinates = np.vstack(coordinates_list)
        if self.center_coordinates:
            coordinates -= coordinates.mean(0)
        features = np.vstack(features_list)
        labels = np.hstack(labels_list)
        # TODO handle how click_idx modification when sweep > 1

        # Populate the instances if required
        if "train" in self.mode and self.instance_population > 0:
            pc_center = coordinates.mean(axis=0)
            instance_c, instance_f, instance_l = self.populate_instances(max_instance_id, pc_center, self.instance_population)
            coordinates = np.vstack((coordinates, instance_c))
            features = np.vstack((features, instance_f))
            labels, obj2label_maps_list, click_idx = self.convert_instance_labels_to_obj_id(labels, instance_l, obj2label_maps_list, click_idx)

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
            "sequence": scan["filepath"],
            "num_points": acc_num_points,
            "coordinates": coordinates,
            "features": features,  # (coordinates, time, intensity, distance)
            "labels": labels,
            "click_idx": click_idx,
            "obj2label": obj2label_maps_list,
        }

    def generate_object_labels(self, scan, max_instance_id, label2obj_map, obj2label_map, click_idx):
        # panoptic_label are defined as category_id * 1000 + instance_id
        file_path = scan["label_filepath"]
        panoptic_labels = np.load(file_path)["data"]
        if self.dataset_type == "nuScenes_challenge":
            panoptic_labels = self.update_challenge_labels(panoptic_labels)
        # semantic_labels = panoptic_labels // 1000  # general class index
        instance_ids = panoptic_labels % 1000

        current_max_instance_id = np.amax(instance_ids)
        if current_max_instance_id > max_instance_id:
            max_instance_id = current_max_instance_id
        # Extract semantic labels

        if "validation" in self.mode and not self.segment_full_scene:
            # TODO: Handle the case when sweep > 1
            if self.obj_type != "all":
                # we need to keep only the things or only the stuff
                scan["clicks"], scan["obj"] = self.select_only_desired_objects_subsampled(self.obj_type, scan["obj"], scan["clicks"])
            obj_labels = np.zeros(panoptic_labels.shape)
            for obj_idx, label in scan["obj"].items():
                obj_labels[panoptic_labels == label] = int(obj_idx)
                obj2label_map[str(int(obj_idx))] = int(label)
            click_idx = scan["clicks"]

        elif "validation" in self.mode and self.segment_full_scene:
            # no pre-defined object selected, choose random objects
            obj_labels = np.zeros(panoptic_labels.shape)
            # unique_panoptic_labels = scan["unique_panoptic_labels"]
            unique_panoptic_labels = list(np.unique(panoptic_labels))
            if 0 in unique_panoptic_labels:
                unique_panoptic_labels.remove(0)
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

    def populate_instances(self, max_instance_id, pc_center, instance_population):
        coordinates_list = []
        features_list = []
        labels_list = []
        for _ in range(instance_population):
            instance_dict = choice(self.instance_data)
            idx = np.random.randint(len(instance_dict["filepaths"]))
            instance_list = []
            for time in range(self.sweep):
                if idx < len(instance_dict["filepaths"]):
                    filepath = instance_dict["filepaths"][idx]
                    instance = np.load(filepath)
                    time_array = np.ones((instance.shape[0], 1)) * time
                    instance = np.hstack((instance[:, :3], time_array, instance[:, 3:4]))
                    instance_list.append(instance)
                    idx = idx + 1
            instances = np.vstack(instance_list)
            coordinates = instances[:, :3] - instances[:, :3].mean(0)
            coordinates += pc_center + np.array([uniform(-10, 10), uniform(-10, 10), uniform(-1, 1)])
            features = instances[:, 3:]
            semantic_label = instance_dict["semantic_label"]
            labels = np.zeros_like(features, dtype=np.int64)
            labels[:, 0] = semantic_label
            max_instance_id = max_instance_id + 1
            labels[:, 1] = max_instance_id
            aug = self.volume_augmentations(points=coordinates)
            coordinates = aug["points"]
            coordinates_list.append(coordinates)
            features_list.append(features)
            labels_list.append(labels)
        return np.vstack(coordinates_list), np.vstack(features_list), np.vstack(labels_list)

    def convert_instance_labels_to_obj_id(self, labels, instance_l, obj2label_maps_list, click_idx):
        # Get unique rows from instance_l
        unique_instance_l = np.unique(instance_l, axis=0)
        new_instance2obj_id_map = {}
        # Start object ID from the maximum label value plus one
        cur_obj_id = labels.max() + 1
        # Iterate over the unique label pairs
        for unique_label_pair in unique_instance_l:
            # Ensure both labels are uint32 and combine the labels into a single uint32 value
            semantic_label = np.uint32(unique_label_pair[0])
            instance_label = np.uint32(unique_label_pair[1])
            combined_label = np.uint32((instance_label << 16) | semantic_label)
            # Convert label_pair to a tuple to make it hashable
            unique_label_pair = tuple(unique_label_pair)
            # Map the unique label pair to a new object ID and update the obj2label for later statistics with the panoptic label (combined_label)
            new_instance2obj_id_map[unique_label_pair] = cur_obj_id
            obj2label_maps_list[0][str(int(cur_obj_id))] = int(combined_label)
            click_idx[str(int(cur_obj_id))] = []
            # Increment the current object
            cur_obj_id += 1

        new_instance_labels = np.zeros(instance_l.shape[0], dtype=np.uint32)
        for i, (semantic_label, instance_label) in enumerate(instance_l):
            new_instance_labels[i] = new_instance2obj_id_map[(semantic_label, instance_label)]

        labels = np.hstack((labels, new_instance_labels))
        return labels, obj2label_maps_list, click_idx

    def update_challenge_labels(self, panoptic_labels):
        general2challenge_mapping = {1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 19: 0, 20: 0, 0: 0, 29: 0, 31: 0, 9: 1, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 21: 6, 2: 7, 3: 7, 4: 7, 6: 7, 12: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 30: 16}

        updated_labels = np.zeros_like(panoptic_labels)
        for i in range(panoptic_labels.shape[0]):
            # Extract category_id and instance_id
            label = panoptic_labels[i]
            category_id = label // 1000
            instance_id = label % 1000

            # Update category_id using the mapping
            if category_id in general2challenge_mapping:
                new_category_id = general2challenge_mapping[category_id]
                # Combine the new category_id and instance_id
                new_label = new_category_id * 1000 + instance_id
                updated_labels[i] = new_label
            else:
                # If category_id is not in mapping, keep the original label
                updated_labels[i] = label

        return updated_labels


def make_scan_transforms(split):

    if split == "train":
        return True
    else:
        return False

    raise ValueError(f"unknown {split}")


def read_bin_point_cloud_nuscene(file_path):
    """
    Loads a .bin file containing the lidarseg or lidar panoptic labels.
    :param bin_path: Path to the .bin file.
    :param type: semantic type, 'lidarseg': stored in 8-bit format, 'panoptic': store in 32-bit format.
    :return: An array containing the labels, with dtype of np.uint8 for lidarseg and np.int32 for panoptic.
    """
    scan = np.fromfile(file_path, dtype=np.float32)
    scan_data = scan.reshape((-1, 5))[:, :4]  #  (x, y, z, intensity)
    points = scan_data[:, :3]  # x, y, z
    intensities = scan_data[:, 3].reshape(-1, 1)  # Intensity
    intensities /= 255  # Normalise to [0, 1]

    return points, intensities
