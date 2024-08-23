import os
import yaml
import numpy as np
import colorsys
import random
import re
import torch
import rerun as rr
import rerun.blueprint as rrb


def get_evenly_distributed_colors(count: int):
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),
            HSV_tuples,
        )
    )


def select_correct_labels(learning_ignore):
    count = 0
    label_info = dict()
    for k, v in learning_ignore.items():
        if v:
            label_info[k] = IGNORE_LABEL
        else:
            label_info[k] = count
            count += 1
    return label_info


def log_images(images: torch.Tensor, sequence: int):
    for i, image in enumerate(images):
        rr.log(f"world/{sequence}/camera/{i}/rgb", rr.Image(image))


def load_point_cloud(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32)
    scan = scan.reshape((-1, 8))  # The point cloud data is stored in a Nx4 format (x, y, z, intensity, timestamp, r,g,b)
    points = scan[:, :3]  # Extracting the (x, y, z) coordinates
    colors, intensities = scan[:, 5:], scan[:, 3]  # Extracting the RGB values
    return points, colors


def load_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32)  # Labels are stored as unsigned 32-bit integers
    return labels


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


# Create a TimeSeries View
my_blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin="/kitti360",
        # Set a custom Y axis.
        axis_y=rrb.ScalarAxis(range=(-1.0, 1.0), zoom_lock=True),
        # Configure the legend.
        plot_legend=rrb.PlotLegend(visible=False),
        # Set time different time ranges for different timelines.
        time_ranges=[
            # Sliding window depending on the time cursor for the first timeline.
            rrb.VisibleTimeRange(
                "timeline0",
                start=rrb.TimeRangeBoundary.cursor_relative(seq=-100),
                end=rrb.TimeRangeBoundary.cursor_relative(),
            ),
            # Time range from some point to the end of the timeline for the second timeline.
            rrb.VisibleTimeRange(
                "timeline1",
                start=rrb.TimeRangeBoundary.absolute(seconds=300.0),
                end=rrb.TimeRangeBoundary.infinite(),
            ),
        ],
    ),
    collapse_panels=True,
)


rr.init("KITTI360", spawn=True, default_blueprint=my_blueprint)


config_path = "conf/semantic-kitti.yaml"
config = yaml.safe_load(open(config_path, "r"))
rr.log(
    "",
    rr.AnnotationContext(
        [rr.AnnotationInfo(id=key, label=value, color=config["color_map"][key]) for key, value in config["labels"].items()],
    ),
    static=True,
)

kitti_360_4d_dir = "/nodes/veltins/work/fradlin/KITTI-360-4D"

i = 0
for sequence in sorted(os.listdir(kitti_360_4d_dir)):
    sequence_dir = os.path.join(kitti_360_4d_dir, sequence)
    velodyne_dir = os.path.join(sequence_dir, "velodyne")
    label_dir = os.path.join(sequence_dir, "labels")

    for file_name in sorted(os.listdir(velodyne_dir), key=extract_number):
        if i % 5 == 0:
            print(f"Processing {sequence}/{file_name}")
        if file_name.endswith(".bin"):
            scan_path = os.path.join(velodyne_dir, file_name)
            scan_name = scan_path.split("/")[-1]
            label_path = os.path.join(label_dir, file_name.replace(".bin", ".label"))

            point_cloud, colors = load_point_cloud(scan_path)
            labels = load_labels(label_path)
            semantic_labels = labels // 1000

            # semantic_labels = np.vectorize(config["learning_map"].__getitem__)(semantic_labels)
            semantic_labels = np.vectorize(map_label)(semantic_labels, config["learning_map"])

            rr.set_time_seconds("kitti360_timestamp", i)
            rr.log(
                f"{sequence}/{file_name}/points",
                rr.Points3D(point_cloud, radii=0.02, class_ids=labels),
            )

            rr.log(
                f"{sequence}/{file_name}/semantic",
                rr.Points3D(point_cloud, radii=0.02, colors=colors.astype(np.uint8)),
            )

            i += 0.1
