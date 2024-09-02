import os
import json
import yaml
import numpy as np
import colorsys
import random
import rerun as rr


def load_point_cloud(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))  # The point cloud data is stored in a Nx4 format (x, y, z, intensity)
    points = scan[:, :3]  # Extracting the (x, y, z) coordinates
    features = scan[:, 3].reshape(-1, 1)  # Extracting the intensity values
    return points, features


def load_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32)  # Labels are stored as unsigned 32-bit integers
    return labels


def load_predictions(prediction_path):
    predictions = np.fromfile(prediction_path, dtype=np.uint32)
    return predictions


def update_challenge_labels(panoptic_labels):
    learning_map = yaml.safe_load(open("/home/fradlin/Github/Mask4D-Interactive/conf/semantic-kitti.yaml"))["learning_map"]
    # Extract category_id and instance_id
    semantic_labels = panoptic_labels & 0xFFFF
    updated_semantic_labels = np.vectorize(learning_map.__getitem__)(semantic_labels)
    # Combine the new category_id and instance_id
    panoptic_labels &= np.array(~0xFFFF).astype(np.uint32)  # Clear lower 16 bits
    panoptic_labels |= updated_semantic_labels.astype(np.uint32)  # Set lower 16 bits with updated semantic labels

    return panoptic_labels


def drop_unneeded(combined_panoptic_labels):
    # Drop the void / ignore class
    mask = combined_panoptic_labels & 0xFFFF != 0
    # Drop the barrier class
    mask &= combined_panoptic_labels & 0xFFFF != 14
    # drop car / truuk instances
    # mask &= combined_panoptic_labels != 4020

    # [4004, 4009, 4011, 4017, 4018, 4020, 4021, 4025, 8039, 10016, 10022, 10028, 11000, 15000]
    len(np.unique(combined_panoptic_labels[mask]))
    return mask


def generate_object_labels(panoptic_labels):
    # drop outlier points
    obj_labels = np.zeros(panoptic_labels.shape)
    unique_panoptic_labels = np.unique(panoptic_labels)
    chosen_objects = unique_panoptic_labels

    obj2label_map = {"1": 458759, "2": 17, "3": 15, "4": 18, "5": 17694721, "6": 16, "7": 458753, "8": 11, "9": 9, "10": 13}
    # This is the first scene we are generating object labels
    for obj_idx, label in obj2label_map.items():
        obj_id = int(obj_idx)
        obj_labels[panoptic_labels == label] = int(obj_id)

    return obj_labels, obj2label_map


# Start a re-run visualization session
rr.init("SemanticKITTI Scene 08", spawn=True)


val_json_path = "/nodes/veltins/work/fradlin/jsons/semantickitti_full_validation_list.json"

with open(val_json_path, "r") as file:
    val_list = json.load(file)
    low_object_scans = [236, 237, 238, 239]

accumulated_point_cloud = []
accumulated_panoptic_labels = []
accumulated_features = []

for time, scan in enumerate(low_object_scans):
    scan_name = str(scan).zfill(6)
    val_key = f"scene_08_{scan_name}"
    scan_entry = val_list[val_key]
    scan_filepath = scan_entry["filepath"]
    label_filepath = scan_entry["label_filepath"]

    point_cloud, features = load_point_cloud(scan_filepath)
    pose = np.array(scan_entry["pose"]).T
    coordinates = point_cloud[:, :3]
    coordinates = coordinates @ pose[:3, :3] + pose[3, :3]

    labels = load_labels(label_filepath)

    time_array = np.ones((features.shape[0], 1)) * time
    features = np.hstack((time_array, features))  # (time, intensity)

    accumulated_panoptic_labels.append(labels)
    accumulated_point_cloud.append(coordinates)
    accumulated_features.append(features)

# Concatenate point clouds and labels
combined_point_cloud = np.vstack(accumulated_point_cloud)
combined_point_cloud -= np.mean(combined_point_cloud, axis=0)

combined_features = np.vstack(accumulated_features)
center_coordinate = combined_point_cloud.mean(0)
combined_features = np.hstack(
    (
        combined_features,
        np.linalg.norm(combined_point_cloud - center_coordinate, axis=1)[:, np.newaxis],
    )
)  # now features include: (time, intensity, distance)

combined_panoptic_labels = np.hstack(accumulated_panoptic_labels)

# drop point more than 40m away
mask = np.linalg.norm(combined_point_cloud, axis=1) < 30
combined_point_cloud = combined_point_cloud[mask]
combined_panoptic_labels = combined_panoptic_labels[mask]
combined_features = combined_features[mask]
#  Update the challenge labels
combined_panoptic_labels = update_challenge_labels(combined_panoptic_labels)
mask = drop_unneeded(combined_panoptic_labels)
combined_panoptic_labels = combined_panoptic_labels[mask]
combined_point_cloud = combined_point_cloud[mask]
combined_features = combined_features[mask]

obj_labels_numbered, obj2label_map = generate_object_labels(combined_panoptic_labels)

obj_color = {1: [1, 211, 211], 2: [233, 138, 0], 3: [41, 207, 2], 4: [244, 0, 128], 5: [194, 193, 3], 6: [121, 59, 50], 7: [254, 180, 214], 8: [239, 1, 51], 9: [85, 85, 85], 10: [229, 14, 241]}
colors = np.array([obj_color[obj_idx] for obj_idx in obj_labels_numbered]).astype(np.uint8)


rr.set_time_seconds("timestamp", 1)
rr.log(
    f"08/{scan_name}/points",
    rr.Points3D(combined_point_cloud, radii=0.02, class_ids=labels, colors=colors),
)
