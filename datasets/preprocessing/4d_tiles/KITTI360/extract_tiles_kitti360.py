import os
import json
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from ply import read_ply
from pathlib import Path
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define the learning map
learning_map = {0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5}


def load_point_cloud(file_path):
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud


def load_labels(file_path):
    labels = np.fromfile(file_path, dtype=np.uint32).reshape(-1)
    return labels


def load_poses_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    poses = {}
    for entry in data.values():
        filepath = entry["filepath"]
        pose_matrix = entry["pose"]
        poses[filepath] = pose_matrix

    return poses


def save_data_item(data, velodyne_path, label_path):
    # Combine coordinates and features
    combined = np.hstack((data["coordinates"], data["features"]))
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(velodyne_path), exist_ok=True)
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    # Save velodyne data
    combined.astype(np.float32).tofile(velodyne_path)
    # Save label data
    data["labels"].astype(np.uint32).tofile(label_path)


def generate_distinct_colors_kmeans(n):
    """
    Generate `n` distinct colors using k-means clustering.

    Args:
        n (int): Number of colors to generate.

    Returns:
        list of tuples: List of RGB color tuples.
    """
    # Sample a large number of colors in RGB space
    np.random.seed(0)
    large_sample = np.random.randint(0, 256, (10000, 3))

    # Apply k-means clustering to find n clusters
    kmeans = KMeans(n_clusters=n, n_init=10).fit(large_sample)
    colors = kmeans.cluster_centers_.astype(int)

    return [tuple(color) for color in colors]


def labels_to_colors(labels):
    """
    Convert labels to colors using a k-means generated colormap.

    Args:
        labels (numpy.ndarray): Array of labels.

    Returns:
        numpy.ndarray: Array of RGB colors.
    """
    # Determine the number of unique labels
    unique_labels = np.unique(labels)
    num_colors = len(unique_labels)

    # Generate distinct colors
    colors_list = generate_distinct_colors_kmeans(num_colors)

    # Create a mapping from labels to colors
    label_to_color = {label: colors_list[i] for i, label in enumerate(unique_labels)}

    # Apply the color map
    colors = np.array([label_to_color[label] for label in labels])
    return colors


def save_pcd(data, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data["coordinates"])
    colors = labels_to_colors(data["labels"])[:, :3] / 255
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)


def quantize_point_cloud(point_cloud, labels, features, voxel_size=0.05):
    # Quantize the point cloud
    # coordinates = point_cloud[:, :3]
    # features = point_cloud[:, 3:]
    _quantized_coords, _quantized_features, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=point_cloud,
        features=features,
        return_index=True,
        return_inverse=True,
        quantization_size=voxel_size,
    )
    quantized_labels = labels[unique_map]
    quantized_coords = point_cloud[unique_map]
    quantized_features = features[unique_map]
    return quantized_coords, quantized_features, quantized_labels


def apply_pose(point_cloud, pose):
    coordinates = point_cloud[:, :3]
    coordinates = coordinates @ pose[:3, :3] + pose[3, :3]
    point_cloud[:, :3] = coordinates
    return point_cloud


def drop_invalid_points(point_cloud, labels):
    semantic_labels = labels & 0xFFFF
    updated_semantic_labels = np.vectorize(learning_map.__getitem__)(semantic_labels)
    valid_mask = updated_semantic_labels != 0
    return point_cloud[valid_mask], labels[valid_mask]


def process_scene(validation_scans, scene_name, output_base_path, tile_size=100, voxel_size=0.1):

    output_sequence_path = os.path.join(output_base_path, scene_name)

    Path(output_sequence_path).mkdir(parents=True, exist_ok=True)

    scan_files = sorted([f for f in validation_scans if f.endswith(".ply")])

    combined_point_cloud_list = []

    time_feature = 0
    for scan_path in tqdm(scan_files):
        static_pcd = read_ply(scan_path)  # x, y, z, red, green, blue, semanticID, instanceID, isVisible, confidence
        dynamic_pcd = read_ply(scan_path.replace("static", "dynamic"))  # x, y, z, red, green, blue, semantic, instance, isVisible, timestamp

        # Convert structured arrays to regular 2D array
        static_pcd_array = np.column_stack([static_pcd[name] for name in static_pcd.dtype.names])
        static_pcd_with_timestamp = np.hstack([static_pcd_array, np.zeros((static_pcd_array.shape[0], 1))])
        dynamic_pcd_array = np.column_stack([dynamic_pcd[name] for name in dynamic_pcd.dtype.names])

        point_cloud = np.vstack([static_pcd_with_timestamp, dynamic_pcd_array])
        # combined_pcd = drop_invalid_points(combined_pcd)
        point_cloud_tensor = torch.from_numpy(point_cloud).float().to("cuda")

        combined_point_cloud_list.append(point_cloud_tensor)

    # the pcd has the following structure: x, y, z, red, green, blue, semanticID, instanceID, isVisible, confidence, timestamp
    combined_point_cloud = torch.cat(combined_point_cloud_list, dim=0)

    # min_x, min_y = np.min(combined_point_cloud[:, :2], axis=0)
    # max_x, max_y = np.max(combined_point_cloud[:, :2], axis=0)

    min_x, min_y = torch.min(combined_point_cloud[:, :2], dim=0).values
    max_x, max_y = torch.max(combined_point_cloud[:, :2], dim=0).values

    tile_index = 0
    x_range = np.arange(min_x.item(), max_x.item(), tile_size)
    y_range = np.arange(min_y.item(), max_y.item(), tile_size)

    # Generate the covering tiles
    for k, x in tqdm(enumerate(x_range)):
        for l, y in enumerate(y_range):
            # Adjust the tile to ensure it fits within the scan
            if x + tile_size > max_x:
                x = max_x - tile_size
            if y + tile_size > max_y:
                y = max_y - tile_size

            mask = (combined_point_cloud[:, 0] >= x) & (combined_point_cloud[:, 0] < x + tile_size) & (combined_point_cloud[:, 1] >= y) & (combined_point_cloud[:, 1] < y + tile_size)

            tile_point_cloud = combined_point_cloud[mask]

            if tile_point_cloud.shape[0] > 0:

                coordinates = tile_point_cloud[:, :3].cpu().numpy()
                features = np.hstack((np.ones((tile_point_cloud.shape[0], 1)), tile_point_cloud[:, 10].unsqueeze(1).cpu().numpy(), tile_point_cloud[:, 3:6].cpu().numpy()))  # intensity, timestamp, red, green, blue
                # instanceID = semanticID*1000 + classInstanceID
                label = tile_point_cloud[:, 7].cpu().numpy()

                # quantized_coords, quantized_features, quantized_labels = quantize_point_cloud(coordinates, label, features, voxel_size=voxel_size)

                tile_filename = f"tile_{tile_index}"

                velodyne_dir = os.path.join(output_sequence_path, "velodyne")
                labels_dir = os.path.join(output_sequence_path, "labels")
                velodyne_filename = os.path.join(velodyne_dir, f"{tile_filename}.bin")
                label_filename = os.path.join(labels_dir, f"{tile_filename}.label")

                data_item = {"coordinates": coordinates, "features": features, "labels": label}

                save_data_item(data_item, velodyne_filename, label_filename)

                tile_index += 1

    print("number of tiles: ", tile_index)
    # interval_index += 1


def process_sequences(validation_scenes, output_path, tile_size=50, voxel_size=0.1):
    # Process training set
    for key, value in tqdm(validation_scenes.items()):
        print(f"Starting pre-processing scene {key}")
        process_scene(value, key, output_path, tile_size=tile_size, voxel_size=voxel_size)
        print(f"Finished pre-processing scene {key}")


def main():
    # Path to KITTI-360 dataset
    dataset_path = "/nodes/anchorbrew/work/yilmaz/KITTI-360"
    validation_split_path = os.path.join(dataset_path, "data_3d_semantics/train/2013_05_28_drive_val.txt")

    validation_scenes = {}
    with open(validation_split_path, "r") as f:
        validation_scans = f.readlines()

    last_scene_name = validation_scans[0].split("/")[2]
    validation_scenes[last_scene_name] = []
    for line in validation_scans:
        scene_name = line.split("/")[2]
        if scene_name != last_scene_name:
            last_scene_name = scene_name
            validation_scenes[scene_name] = []
        scan_path = os.path.join(dataset_path, line.strip())
        validation_scenes[scene_name].append(scan_path)

    # Path to save the processed data
    output_path = "/nodes/veltins/work/fradlin/KITTI-360-4D"

    process_sequences(validation_scenes, output_path, tile_size=30, voxel_size=0.1)

    print(f"Data processing complete. Processed data saved to {output_path}")


if __name__ == "__main__":
    main()
