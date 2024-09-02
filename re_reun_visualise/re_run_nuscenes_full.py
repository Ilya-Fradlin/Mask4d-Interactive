import os
import json
import yaml
import numpy as np
import colorsys
import random
import rerun as rr
from pathlib import Path


classname_to_color = {  # RGB.
    "noise": (0, 0, 0),  # Black.
    "animal": (70, 130, 180),  # Steelblue
    "human.pedestrian.adult": (0, 0, 230),  # Blue
    "human.pedestrian.child": (135, 206, 235),  # Skyblue,
    "human.pedestrian.construction_worker": (100, 149, 237),  # Cornflowerblue
    "human.pedestrian.personal_mobility": (219, 112, 147),  # Palevioletred
    "human.pedestrian.police_officer": (0, 0, 128),  # Navy,
    "human.pedestrian.stroller": (240, 128, 128),  # Lightcoral
    "human.pedestrian.wheelchair": (138, 43, 226),  # Blueviolet
    "movable_object.barrier": (112, 128, 144),  # Slategrey
    "movable_object.debris": (210, 105, 30),  # Chocolate
    "movable_object.pushable_pullable": (105, 105, 105),  # Dimgrey
    "movable_object.trafficcone": (47, 79, 79),  # Darkslategrey
    "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
    "vehicle.bicycle": (220, 20, 60),  # Crimson
    "vehicle.bus.bendy": (255, 127, 80),  # Coral
    "vehicle.bus.rigid": (255, 69, 0),  # Orangered
    "vehicle.car": (255, 158, 0),  # Orange
    "vehicle.construction": (233, 150, 70),  # Darksalmon
    "vehicle.emergency.ambulance": (255, 83, 0),
    "vehicle.emergency.police": (255, 215, 0),  # Gold
    "vehicle.motorcycle": (255, 61, 99),  # Red
    "vehicle.trailer": (255, 140, 0),  # Darkorange
    "vehicle.truck": (255, 99, 71),  # Tomato
    "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
    "flat.other": (175, 0, 75),
    "flat.sidewalk": (75, 0, 75),
    "flat.terrain": (112, 180, 60),
    "static.manmade": (222, 184, 135),  # Burlywood
    "static.other": (255, 228, 196),  # Bisque
    "static.vegetation": (0, 175, 0),  # Green
    "vehicle.ego": (255, 240, 245),
}

label_mapping = {
    0: "noise",
    1: "animal",
    2: "human.pedestrian.adult",
    3: "human.pedestrian.child",
    4: "human.pedestrian.construction_worker",
    5: "human.pedestrian.personal_mobility",
    6: "human.pedestrian.police_officer",
    7: "human.pedestrian.stroller",
    8: "human.pedestrian.wheelchair",
    9: "movable_object.barrier",
    10: "movable_object.debris",
    11: "movable_object.pushable_pullable",
    12: "movable_object.trafficcone",
    13: "static_object.bicycle_rack",
    14: "vehicle.bicycle",
    15: "vehicle.bus.bendy",
    16: "vehicle.bus.rigid",
    17: "vehicle.car",
    18: "vehicle.construction",
    19: "vehicle.emergency.ambulance",
    20: "vehicle.emergency.police",
    21: "vehicle.motorcycle",
    22: "vehicle.trailer",
    23: "vehicle.truck",
    24: "flat.driveable_surface",
    25: "flat.other",
    26: "flat.sidewalk",
    27: "flat.terrain",
    28: "static.manmade",
    29: "static.other",
    30: "static.vegetation",
    31: "vehicle.ego",
}


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


validation_data_file = Path("/nodes/veltins/work/fradlin/jsons/nuscenes_validation_list.json")
with open(validation_data_file, "r") as f:
    validation_data = json.load(f)


# Start a re-run visualization session
rr.init("nuScenes Validation", spawn=True)

rr.log(
    "",
    rr.AnnotationContext(
        [rr.AnnotationInfo(id=label_id, label=label_name, color=classname_to_color[label_name]) for label_id, label_name in label_mapping.items()],
    ),
    static=True,
)

# Initiali
# ze variables for accumulating point clouds and labels
accumulated_point_cloud = []
accumulated_semantic_labels = []
batch_size = 4  # Number of scans to accumulate

i = 0  # Initialize counter for batch processing
previous_scan = None

for key, item in validation_data.items():
    label_path = item["label_filepath"]
    scan_path = item["filepath"]
    scan_name = scan_path.split("/")[-1].split(".")[0]
    sequence = item["scene"]
    number_of_objects = len(item["unique_panoptic_labels"])

    point_cloud, features = read_bin_point_cloud_nuscene(scan_path)
    coordinates = point_cloud[:, :3]
    pose = np.array(item["pose"]).T
    coordinates = coordinates @ pose[:3, :3] + pose[3, :3]

    label_file_path = item["label_filepath"]
    panoptic_labels = np.load(label_file_path)["data"]
    semantic_labels = panoptic_labels // 1000

    # point_cloud, semantic_labels = drop_unlabelled(semantic_labels, point_cloud)

    # Accumulate point clouds and labels
    accumulated_point_cloud.append(coordinates)
    accumulated_semantic_labels.append(semantic_labels)

    if len(accumulated_point_cloud) == batch_size:
        # Concatenate point clouds and labels
        combined_point_cloud = np.vstack(accumulated_point_cloud)
        combined_point_cloud -= np.mean(combined_point_cloud, axis=0)
        combined_semantic_labels = np.hstack(accumulated_semantic_labels)
        rr.set_time_seconds("nuscenes_timestamp", i)
        rr.log(
            f"{sequence}/combined_points_batch_{i}_{scan_name}",
            rr.Points3D(combined_point_cloud, radii=0.02, class_ids=combined_semantic_labels),
        )

        # Clear the accumulated lists for the next batch
        accumulated_point_cloud = []
        accumulated_semantic_labels = []

        # Update the previous scan (this will clear the combined batch in the next iteration)
        if previous_scan is not None:
            rr.log(previous_scan, rr.Clear(recursive=False))  # or `rr.Clear.flat()`
        previous_scan = f"{sequence}/combined_points_batch_{i}_{scan_name}"
        i += 1
