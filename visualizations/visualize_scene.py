import os
import sys
import open3d as o3d
import numpy as np
import torch


def save_labeled_point_cloud(raw_coords_qv, labels_qv, output_file):
    """
    Save a labeled point cloud in PCD format.

    Args:
    - raw_coords_qv (numpy.ndarray): Raw coordinates of the point cloud.
    - labels_qv (numpy.ndarray): Labels corresponding to each point.
    - output_file (str): Path to save the output PCD file.
    """
    # Create an Open3D PointCloud object
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(raw_coords_qv, torch.Tensor):
        raw_coords_qv = raw_coords_qv.cpu().numpy()
    if isinstance(labels_qv, torch.Tensor):
        labels_qv = labels_qv.cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(raw_coords_qv)

    # Convert labels to RGB colors
    unique_labels = np.unique(labels_qv)
    label_colors = np.random.rand(len(unique_labels), 3)  # Random colors for each unique label
    label_to_color = dict(zip(unique_labels, label_colors))
    colors = [label_to_color[label] for label in labels_qv]

    # Set point cloud colors
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Save the labeled point cloud to a PCD file
    o3d.io.write_point_cloud(output_file, point_cloud)


# Example usage:
# Assuming `raw_coords_qv` and `labels_qv` are your numpy arrays
# and `output_file` is the path where you want to save the PCD file
# save_labeled_point_cloud(raw_coords_qv, labels_qv, 'labeled_point_cloud.pcd')


def save_scene(scene_path):
    pass


def visualize_scene(pcd_features, labels):
    # Load the scene
    pcd = open3d.io.read_point_cloud(scene_path)
    # Visualize the scene
    open3d.visualization.draw_geometries([pcd])
