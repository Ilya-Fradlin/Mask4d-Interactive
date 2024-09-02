import os
import json
import yaml
import numpy as np
import colorsys
import random
import rerun as rr
from pathlib import Path


classname_to_color = {  # RGB.
    "void / ignore": (0, 0, 0),  # Black.
    "barrier (thing)": (70, 130, 180),  # Steelblue
    "bicycle (thing)": (0, 0, 230),  # Blue
    "bus (thing)": (135, 206, 235),  # Skyblue,
    "car (thing)": (100, 149, 237),  # Cornflowerblue
    "construction_vehicle (thing)": (219, 112, 147),  # Palevioletred
    "motorcycle (thing)": (0, 0, 128),  # Navy,
    "pedestrian (thing)": (240, 128, 128),  # Lightcoral
    "traffic_cone (thing)": (138, 43, 226),  # Blueviolet
    "trailer (thing)": (112, 128, 144),  # Slategrey
    "truck (thing)": (210, 105, 30),  # Chocolate
    "driveable_surface (stuff)": (105, 105, 105),  # Dimgrey
    "other_flat (stuff)": (47, 79, 79),  # Darkslategrey
    "sidewalk (stuff)": (188, 143, 143),  # Rosybrown
    "terrain (stuff)": (220, 20, 60),  # Crimson
    "manmade (stuff)": (255, 127, 80),  # Coral
    "vegetation (stuff)": (255, 69, 0),  # Orangered
}

label_mapping = {
    0: "void / ignore",
    1: "barrier (thing)",
    2: "bicycle (thing)",
    3: "bus (thing)",
    4: "car (thing)",
    5: "construction_vehicle (thing)",
    6: "motorcycle (thing)",
    7: "pedestrian (thing)",
    8: "traffic_cone (thing)",
    9: "trailer (thing)",
    10: "truck (thing)",
    11: "driveable_surface (stuff)",
    12: "other_flat (stuff)",
    13: "sidewalk (stuff)",
    14: "terrain (stuff)",
    15: "manmade (stuff)",
    16: "vegetation (stuff)",
}

challenge2general_mapping = {0: 1, 1: 9, 2: 14, 3: 15, 4: 17, 5: 18, 6: 21, 7: 2, 8: 12, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 30}


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


def load_predictions(prediction_path):
    predictions = np.fromfile(prediction_path, dtype=np.uint32)
    return predictions


def generate_object_labels(panoptic_labels):
    # drop outlier points
    obj_labels = np.zeros(panoptic_labels.shape)
    unique_panoptic_labels = np.unique(panoptic_labels)
    chosen_objects = unique_panoptic_labels

    obj2label_map = {"1": 4004, "2": 4009, "3": 4025, "4": 4021, "5": 4011, "6": 10028, "7": 10016, "8": 15000, "9": 11000}
    # This is the first scene we are generating object labels
    for obj_idx, label in obj2label_map.items():
        obj_id = int(obj_idx)
        obj_labels[panoptic_labels == label] = int(obj_id)

    return obj_labels, obj2label_map


def update_challenge_labels(panoptic_labels):
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


def drop_unneeded(combined_panoptic_labels):
    # Drop the void / ignore class
    mask = combined_panoptic_labels != 0
    # Drop the barrier class
    mask &= combined_panoptic_labels // 1000 != 1
    # Drop the sidewalk class
    mask &= combined_panoptic_labels // 1000 != 13
    # Drop pedestrian class
    mask &= combined_panoptic_labels // 1000 != 7
    # Drop the specific instances
    mask &= combined_panoptic_labels // 1000 != 8
    # drop car / truuk instances
    mask &= combined_panoptic_labels != 4020
    mask &= combined_panoptic_labels != 4017
    mask &= combined_panoptic_labels != 4018
    mask &= combined_panoptic_labels != 10022

    # [4004, 4009, 4011, 4017, 4018, 4020, 4021, 4025, 8039, 10016, 10022, 10028, 11000, 15000]
    len(np.unique(combined_panoptic_labels[mask]))
    return mask


validation_data_file = Path("/nodes/veltins/work/fradlin/jsons/nuscenes_validation_list.json")
with open(validation_data_file, "r") as f:
    validation_data = json.load(f)

# lists_of_interest = ["scene_0092_5fe961cfd2b444b0898c3abb7c969e77", "scene_0770_41e55a08b2dd4bb0836017b43cd181b3", "scene_0913_ce9df8d818354073b48c2b0e5e8ff7b5"]
lists_of_interest_1 = ["scene_0092_38c0c7b30e1142e4bc0833c2d489c493", "scene_0092_47f40eb09dcb40b9888d81fb4b872ac4", "scene_0092_47d74378d3e04fa6bb3132928c0f9fa4", "scene_0092_5fe961cfd2b444b0898c3abb7c969e77"]
# lists_of_interest_2 = ["scene_0770_e0286a70489f45a5b3469ee1b71b7667", "scene_0770_b380e87cf0df4f129ec3f6892ba7f6be", "scene_0770_7a2d2320af574587a515bd325ad86e36", "scene_0770_41e55a08b2dd4bb0836017b43cd181b3"]
# lists_of_interest_3 = ["scene_0913_04068be1e7374a34b7c9303002492d5c", "scene_0913_51bbacdcf4c94b8c977ab36f1f0c9e1b", "scene_0913_84228dcff485464a9011c592e6dd397a", "scene_0913_ce9df8d818354073b48c2b0e5e8ff7b5"]
combined_list = [lists_of_interest_1]
# Start a re-run visualization session
rr.init("nuScenes Validation", spawn=True)

# rr.log(
#     "",
#     rr.AnnotationContext(
#         [rr.AnnotationInfo(id=label_id, label=label_name, color=classname_to_color[label_name]) for label_id, label_name in label_mapping.items()],
#     ),
#     static=True,
# )

# Initiali
# ze variables for accumulating point clouds and labels
accumulated_point_cloud = []
accumulated_panoptic_labels = []
batch_size = 4  # Number of scans to accumulate

i = 0  # Initialize counter for batch processing
previous_scan = None

for batch_scans in combined_list:
    for key_scan in batch_scans:
        item = validation_data[key_scan]
        label_path = item["label_filepath"]
        scan_path = item["filepath"]
        scan_name = scan_path.split("/")[-1].split(".")[0]
        sequence = item["scene"]

        point_cloud, features = read_bin_point_cloud_nuscene(scan_path)
        coordinates = point_cloud[:, :3]
        pose = np.array(item["pose"]).T
        coordinates = coordinates @ pose[:3, :3] + pose[3, :3]

        label_file_path = item["label_filepath"]
        panoptic_labels = np.load(label_file_path)["data"]

        # point_cloud, semantic_labels = drop_unlabelled(semantic_labels, point_cloud)

        # Accumulate point clouds and labels
        accumulated_point_cloud.append(coordinates)
        accumulated_panoptic_labels.append(panoptic_labels)

    # Concatenate point clouds and labels
    combined_point_cloud = np.vstack(accumulated_point_cloud)
    combined_point_cloud -= np.mean(combined_point_cloud, axis=0)
    combined_panoptic_labels = np.hstack(accumulated_panoptic_labels)

    # drop point more than 30m away
    mask = np.linalg.norm(combined_point_cloud, axis=1) < 40
    combined_point_cloud = combined_point_cloud[mask]
    combined_panoptic_labels = combined_panoptic_labels[mask]
    #  Update the challenge labels
    combined_panoptic_labels = update_challenge_labels(combined_panoptic_labels)

    mask = drop_unneeded(combined_panoptic_labels)

    combined_panoptic_labels = combined_panoptic_labels[mask]
    combined_point_cloud = combined_point_cloud[mask]
    combined_semantic_labels = combined_panoptic_labels // 1000

    obj_labels_numbered, obj2label_map = generate_object_labels(combined_panoptic_labels)

    obj_color = {1: [1, 211, 211], 2: [233, 138, 0], 3: [41, 207, 2], 4: [244, 0, 128], 5: [194, 193, 3], 6: [121, 59, 50], 7: [254, 180, 214], 8: [239, 1, 51], 9: [85, 85, 85], 10: [229, 14, 241]}
    colors = np.array([obj_color[obj_idx] for obj_idx in obj_labels_numbered]).astype(np.uint8)

    rr.set_time_seconds("nuscenes_timestamp", i)
    rr.log(
        f"{sequence}/combined_points_batch_{i}",
        rr.Points3D(combined_point_cloud, radii=0.02, class_ids=combined_semantic_labels, colors=colors),
    )

    number_of_objects = len(np.unique(combined_panoptic_labels))
    print(f"for batch {i}, the number of objects is {number_of_objects}")
    print(f"the objects are:")
    for obj in np.unique(combined_semantic_labels):
        print(f"{label_mapping[obj]} with number of instances: {np.sum(combined_semantic_labels == obj)}")

    # Clear the accumulated lists for the next batch
    accumulated_point_cloud = []
    accumulated_semantic_labels = []

    # Update the previous scan (this will clear the combined batch in the next iteration)
    if previous_scan is not None:
        rr.log(previous_scan, rr.Clear(recursive=False))  # or `rr.Clear.flat()`
    previous_scan = f"{sequence}/combined_points_batch_{i}"
    i += 1
