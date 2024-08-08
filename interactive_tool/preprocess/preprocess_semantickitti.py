import numpy as np
import yaml
import os

from ply import *


def generate_object_labels(label_filepath, max_instance_id, label2obj_map, obj2label_map, click_idx):
    panoptic_labels = np.fromfile(label_filepath, dtype=np.uint32)
    current_max_instance_id = np.amax(panoptic_labels >> 16)
    if current_max_instance_id > max_instance_id:
        max_instance_id = current_max_instance_id
    # Extract semantic labels
    semantic_labels = panoptic_labels & 0xFFFF
    learning_map = yaml.safe_load(open("/home/fradlin/Github/Mask4D-Interactive/conf/semantic-kitti.yaml"))["learning_map"]
    updated_semantic_labels = np.vectorize(learning_map.__getitem__)(semantic_labels)
    # Update semantic labels
    panoptic_labels &= np.array(~0xFFFF).astype(np.uint32)  # Clear lower 16 bits
    panoptic_labels |= updated_semantic_labels.astype(np.uint32)  # Set lower 16 bits with updated semantic labels
    # drop outlier points
    mask = panoptic_labels != 0
    panoptic_labels = panoptic_labels[mask]
    obj_labels = np.zeros(panoptic_labels.shape)
    unique_panoptic_labels = np.unique(panoptic_labels)
    chosen_objects = unique_panoptic_labels

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

    return obj_labels, obj2label_map, click_idx, max_instance_id, label2obj_map, mask


def get_colors(labels, obj2label_map):
    learning_map_inv = yaml.safe_load(open("/home/fradlin/Github/Mask4D-Interactive/conf/semantic-kitti.yaml"))["learning_map_inv"]
    color_map = yaml.safe_load(open("/home/fradlin/Github/Mask4D-Interactive/conf/semantic-kitti.yaml"))["color_map"]
    # update the color map with the learning map
    color_map = {key: color_map[value] for key, value in learning_map_inv.items()}
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for obj_idx, panoptic_label in obj2label_map.items():
        semantic_lables = panoptic_label & 0xFFFF
        colors[labels == int(obj_idx)] = color_map[semantic_lables]
    # return colors / 255
    return colors


# low_object_scans = [236, 237, 238, 239, 243, 244, 246, 247, 248, 249, 250, 1567]
low_object_scans = [
    223,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    258,
    262,
    266,
    267,
    268,
    269,
    270,
    271,
    272,
    299,
    300,
    301,
    302,
    303,
    1563,
    1564,
    1565,
    1566,
    1567,
    1568,
    1569,
    1570,
    1575,
    1591,
    1594,
    1595,
    1604,
    1605,
    1606,
    1607,
    1608,
    1609,
    1625,
    1637,
    1638,
    1639,
    1640,
    1641,
    1642,
    1643,
]
low_object_scans = [238]

for scan in low_object_scans:
    scan_name = str(scan).zfill(6)
    scan_filepath = f"/globalwork/data/SemanticKITTI/dataset/sequences/08/velodyne/{scan_name}.bin"
    label_filepath = f"/globalwork/data/SemanticKITTI/dataset/sequences/08/labels/{scan_name}.label"
    points = np.fromfile(scan_filepath, dtype=np.float32).reshape(-1, 4)
    coordinates = points[:, :3]
    center_coordinate = coordinates.mean(0)
    coordinates -= center_coordinate
    features = points[:, 3:4]  # intensity
    time = 1  # Preparation for 4D data
    time_array = np.ones((features.shape[0], 1)) * time
    features = np.hstack((time_array, features))  # (time, intensity)
    features = np.hstack((features, np.linalg.norm(coordinates - center_coordinate, axis=1)[:, np.newaxis]))  #  now features include: (time, intensity, distance)

    max_instance_id = 0
    label2obj_map = {}
    obj2label_map = {}
    click_idx = {}
    labels, obj2label_map, click_idx, max_instance_id, label2obj_map, mask = generate_object_labels(label_filepath, max_instance_id, label2obj_map, obj2label_map, click_idx)
    coordinates = coordinates[mask]
    features = features[mask]

    colors = get_colors(labels, obj2label_map)

    dir_path = f"/work/fradlin/Interactive_dataset/interactive4d_data/scene_01_SemanticKITTI_08_{scan_name}"
    os.makedirs(dir_path, exist_ok=True)
    # save obj2label_map
    with open(os.path.join(dir_path, "obj2label.yaml"), "w") as file:
        yaml.dump(obj2label_map, file)

    field_names = ["x", "y", "z", "red", "green", "blue", "time", "intensity", "distance", "label"]
    ply_path = os.path.join(dir_path, f"scan_{scan}.ply")
    write_ply(ply_path, [coordinates, colors, features, labels], field_names)
