import os
import yaml
import numpy as np
import colorsys
import random
import rerun as rr


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
    scan = scan.reshape((-1, 4))  # The point cloud data is stored in a Nx4 format (x, y, z, intensity)
    points = scan[:, :3]  # Extracting the (x, y, z) coordinates
    return points


def load_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32)  # Labels are stored as unsigned 32-bit integers
    return labels


def load_predictions(prediction_path):
    predictions = np.fromfile(prediction_path, dtype=np.uint32)
    return predictions


velodyne_dir = "/globalwork/data/SemanticKITTI/dataset/sequences/08/velodyne/"
label_dir = "/globalwork/data/SemanticKITTI/dataset/sequences/08/labels/"
prediction_dir = "/nodes/veltins/work/fradlin/predictions/2024-08-21_153206-Interactive4d_validation3/average_10_clicks/sequences/08/predictions/"

# Start a re-run visualization session
rr.init("SemanticKITTI Scene 08", spawn=True)

config_path = "conf/semantic-kitti.yaml"
config = yaml.safe_load(open(config_path, "r"))
rr.log(
    "",
    rr.AnnotationContext(
        [rr.AnnotationInfo(id=key, label=value, color=config["color_map"][key]) for key, value in config["labels"].items()],
    ),
    static=True,
)

for file_name in sorted(os.listdir(velodyne_dir)):
    if file_name.endswith(".bin"):
        time_int = int(file_name.split(".")[0])
        scan_path = os.path.join(velodyne_dir, file_name)
        scan_name = scan_path.split("/")[-1]
        label_path = os.path.join(label_dir, file_name.replace(".bin", ".label"))
        prediction_path = os.path.join(prediction_dir, file_name.replace(".bin", ".label"))

        point_cloud = load_point_cloud(scan_path)
        labels = load_labels(label_path)
        predictions = load_predictions(prediction_path)

        rr.set_time_seconds("rio_timestamp", time_int * 0.1)
        rr.log(
            f"08/{scan_name}/points",
            rr.Points3D(point_cloud, radii=0.02, class_ids=labels),
        )
