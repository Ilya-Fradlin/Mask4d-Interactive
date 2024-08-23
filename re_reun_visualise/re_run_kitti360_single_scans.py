import os
from datetime import datetime
import sys
import numpy as np
import colorsys
import random
import re
import json
import rerun as rr
import rerun.blueprint as rrb

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from datasets.kitti360_info import id_to_color_map, label2name

IGNORE_LABEL = -1


def get_evenly_distributed_colors(count: int):
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),
            HSV_tuples,
        )
    )


def load_predictions(prediction_path):
    predictions = np.fromfile(prediction_path, dtype=np.uint32)
    return predictions


# Define a function that checks if the value exists in the map, and returns the original if not
def map_label(label, learning_map):
    return learning_map.get(label, label)


# Function to extract the numeric part of the filename
def extract_number(filename):
    match = re.search(r"\d+", filename)  # Find the first sequence of digits
    return int(match.group()) if match else float("inf")  # Return as integer, or inf if no number is found


def load_point_cloud(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))  # The point cloud data is stored in a Nx4 format (x, y, z, intensity)
    points = scan[:, :3]  # Extracting the (x, y, z) coordinates
    # colors, intensities = scan[:, 5:], scan[:, 3]  # Extracting the RGB values
    return points


def load_labels(label_path):
    instance_label = np.fromfile(label_path, dtype=np.int64)  # Labels are stored as unsigned 32-bit integers
    return instance_label


def drop_unlabelled(semantic_labels, point_cloud):
    mask = np.logical_and(semantic_labels != -1, semantic_labels != 0)
    print(f"number of unlabelled points: {np.sum(~mask)}")
    return point_cloud[mask], semantic_labels[mask]


def loadTransform(filename):

    transform = np.eye(4)

    try:
        infile = open(filename).readline().rstrip("\n").split(" ")
    except:
        print("Failed to open transforms " + filename)
        return transform, False

    for i in range(12):
        xi = i // 4
        yi = i % 4
        transform[xi, yi] = float(infile[i])

    return transform, True


def transform_points(points, pose):
    # Convert points to homogeneous coordinates (Nx4)

    points = points @ pose[:3, :3].T + pose[:3, 3]
    return points
    # points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    # # Apply the transformation matrix
    # transformed_points = (pose_matrix @ points_homogeneous.T).T
    # # Return the transformed points in 3D (dropping the homogeneous coordinate)
    # return transformed_points[:, :3]


def loadCamPose(filename):
    poses = [None for _ in range(4)]

    try:
        infile = open(filename)
    except:
        print("Failed to open camera poses " + filename)
        return poses, False

    for line in infile:
        lineProcessed = line.rstrip("\n").split(" ")
        if any("image_0" in x for x in lineProcessed):
            transform = np.eye(4)
            index = int(lineProcessed[0][7])
            for i in range(12):
                xi = i // 4
                yi = i % 4
                transform[xi, yi] = float(lineProcessed[i + 1])
            poses[index] = transform

    infile.close()
    return poses, True


rr.init("KITTI360", spawn=True)
# rr.send_blueprint(blueprint, make_active=True)

rr.log(
    "",
    rr.AnnotationContext(
        [rr.AnnotationInfo(id=key, label=label2name[key], color=id_to_color_map[key]) for key, value in id_to_color_map.items()],
    ),
    static=False,
)


validation_json_path = "/nodes/veltins/work/fradlin/jsons/kitti360_single_scans_validation.json"
with open(validation_json_path, "r") as file:
    validation_data = json.load(file)

# Initialize variables for accumulating point clouds and labels
accumulated_point_cloud = []
accumulated_semantic_labels = []
batch_size = 10  # Number of scans to accumulate

i = 0  # Initialize counter for batch processing
previous_scan = None

for key, item in validation_data.items():
    label_path = item["label_filepath"]
    scan_path = item["filepath"]
    scan_name = item["scan"]
    sequence = item["scene"]
    timestamp = item["timestamp"][:26]
    timestamp_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
    seconds = timestamp_dt.timestamp()

    pose_matrix = np.array(item["poses"])

    extrinsicName = "/nodes/anchorbrew/work/yilmaz/KITTI-360/calibration/" + "/calib_cam_to_velo.txt"
    cam2poseName = "/nodes/anchorbrew/work/yilmaz/KITTI-360/calibration/" + "/calib_cam_to_pose.txt"
    Tr_cam_velo, success = loadTransform(extrinsicName)
    Tr_cam_pose, success = loadCamPose(cam2poseName)

    Tr_velo_pose = Tr_cam_pose[0] @ np.linalg.inv(Tr_cam_velo)
    velo_to_world = pose_matrix @ Tr_velo_pose

    point_cloud = load_point_cloud(scan_path)
    point_cloud = transform_points(point_cloud, velo_to_world)

    labels = load_labels(label_path)
    semantic_labels = labels // 1000

    point_cloud, semantic_labels = drop_unlabelled(semantic_labels, point_cloud)

    # Accumulate point clouds and labels
    accumulated_point_cloud.append(point_cloud)
    accumulated_semantic_labels.append(semantic_labels)

    if len(accumulated_point_cloud) == batch_size:
        # Concatenate point clouds and labels
        combined_point_cloud = np.vstack(accumulated_point_cloud)
        combined_point_cloud -= np.mean(combined_point_cloud, axis=0)
        combined_semantic_labels = np.hstack(accumulated_semantic_labels)
        rr.set_time_seconds("kitti360_timestamp", seconds)
        rr.log(
            f"{sequence}/combined_points_batch_{i}",
            rr.Points3D(combined_point_cloud, radii=0.02, class_ids=combined_semantic_labels),
        )

        # Clear the accumulated lists for the next batch
        accumulated_point_cloud = []
        accumulated_semantic_labels = []

        # Update the previous scan (this will clear the combined batch in the next iteration)
        if previous_scan is not None:
            rr.log(previous_scan, rr.Clear(recursive=False))  # or `rr.Clear.flat()`
        previous_scan = f"{sequence}/combined_points_batch_{i}"
        i += 1


# # Create a TimeSeries View
# my_blueprint = rrb.Blueprint(
#     rrb.TimeSeriesView(
#         origin="/kitti360",
#         # Set time different time ranges for different timelines.
#         time_ranges=[
#             # Sliding window depending on the time cursor for the first timeline.
#             rrb.VisibleTimeRange(
#                 "kitti360_timestamp",
#                 start=rrb.TimeRangeBoundary.cursor_relative(seq=-0.5),
#                 end=rrb.TimeRangeBoundary.cursor_relative(),
#             ),
#         ],
#     ),
#     collapse_panels=False,
# )


# blueprint = rrb.Vertical(
#     rrb.Spatial3DView(name="3D", origin="KITTI360"),
# )

# sequence = "2013_05_28_drive_0003_sync"
# velodyne_dir = f"/nodes/anchorbrew/work/yilmaz/KITTI-360/data_3d_raw/{sequence}/velodyne_points/data/"
# label_dir = f"/work/fradlin/KITTI360SingleScan/{sequence}/labels/"

# previous_scan = None
# for available_label in sorted(os.listdir(label_dir)):
#     if available_label.endswith(".bin"):

#         label_path = os.path.join(label_dir, available_label)
#         scan_path = os.path.join(velodyne_dir, available_label)
#         scan_name = scan_path.split("/")[-1]

#         point_cloud = load_point_cloud(scan_path)
#         labels = load_labels(label_path)
#         semantic_labels = labels // 1000

#         point_cloud, semantic_labels = drop_unlabelled(semantic_labels, point_cloud)

#         # semantic_labels = np.vectorize(config["learning_map"].__getitem__)(semantic_labels)
#         # semantic_labels = np.vectorize(map_label)(semantic_labels, config["learning_map"])

#         rr.set_time_seconds("kitti360_timestamp", i)
#         rr.log(
#             f"{sequence}/{available_label}/points",
#             rr.Points3D(point_cloud, radii=0.02, class_ids=semantic_labels),
#         )
#         if previous_scan is not None:
#             rr.log(previous_scan, rr.Clear(recursive=False))  # or `rr.Clear.flat()`

#         # rr.log(
#         #     f"{sequence}/{available_label}/semantic",
#         #     rr.Points3D(point_cloud, radii=0.02, colors=colors.astype(np.uint8)),
#         # )
#         previous_scan = f"{sequence}/{available_label}/points"
#         i += 1
