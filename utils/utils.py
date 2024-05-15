import torch
import sys
import wandb
import numpy as np
from scipy.optimize import linear_sum_assignment


if sys.version_info[:2] >= (3, 8):
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping


def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def associate_instances(previous_ins_label, current_ins_label):
    previous_instance_ids, c_p = np.unique(previous_ins_label[previous_ins_label != 0], return_counts=True)
    current_instance_ids, c_c = np.unique(current_ins_label[current_ins_label != 0], return_counts=True)

    associations = {0: 0}

    large_previous_instance_ids = []
    large_current_instance_ids = []
    for id, count in zip(previous_instance_ids, c_p):
        if count > 25:
            large_previous_instance_ids.append(id)
    for id, count in zip(current_instance_ids, c_c):
        if count > 50:
            large_current_instance_ids.append(id)

    p_n = len(large_previous_instance_ids)
    c_n = len(large_current_instance_ids)

    association_costs = torch.zeros(p_n, c_n)
    for i, p_id in enumerate(large_previous_instance_ids):
        for j, c_id in enumerate(large_current_instance_ids):
            intersection = np.sum((previous_ins_label == p_id) & (current_ins_label == c_id))
            union = np.sum(previous_ins_label == p_id) + np.sum(current_ins_label == c_id) - intersection
            iou = intersection / union
            cost = 1 - iou
            association_costs[i, j] = cost

    idxes_1, idxes_2 = linear_sum_assignment(association_costs)

    for i1, i2 in zip(idxes_1, idxes_2):
        if association_costs[i1][i2] < 1.0:
            associations[large_current_instance_ids[i2]] = large_previous_instance_ids[i1]
    return associations


# Function to calculate all 8 corners of the bounding box
def calculate_bounding_box_corners(min_x, max_x, min_y, max_y, min_z, max_z):
    corners = []
    for x in (min_x, max_x):
        for y in (min_y, max_y):
            for z in (min_z, max_z):
                corners.append([x, y, z])
    return corners


def generate_wandb_objects3d(raw_coords, labels, pred, click_idx):
    # Add a new dimension to labels
    labels = torch.unsqueeze(labels, dim=1)
    pred = torch.unsqueeze(pred, dim=1)
    # Stack the tensors along the second dimension
    pcd_gt = torch.cat((raw_coords, labels), dim=1).cpu().numpy()
    pcd_pred = torch.cat((raw_coords, pred), dim=1).cpu().numpy()
    # Convert the 4th column to integers and add +1 for wandb class range
    pcd_gt[:, 3] = pcd_gt[:, 3].astype(int) + 1
    pcd_pred[:, 3] = pcd_pred[:, 3].astype(int) + 1

    ####################################################################################
    ############### Get the Predicted and clicks as small Bounding Boxes ###############
    ####################################################################################
    boxes_array = []
    # Iterate over each object in sample_click_idx
    for obj, click_indices in click_idx.items():
        # Extract click points from numpy_array
        click_points = pcd_pred[click_indices]
        # Calculate bounding box coordinates
        for click in click_points:
            min_x, max_x = click[0] - 0.1, click[0] + 0.1
            min_y, max_y = click[1] - 0.1, click[1] + 0.1
            min_z, max_z = click[2] - 0.1, click[2] + 0.1

            current_box_click = {
                "corners": calculate_bounding_box_corners(min_x, max_x, min_y, max_y, min_z, max_z),
                "label": f"obj_{obj}",
                "color": [123, 321, 111],
            }
            boxes_array.append(current_box_click)

    gt_scene = wandb.Object3D({"type": "lidar/beta", "points": pcd_gt})
    pred_scene = wandb.Object3D({"type": "lidar/beta", "points": pcd_pred, "boxes": np.array(boxes_array)})

    return gt_scene, pred_scene
