# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------
import torch
import numpy as np
import random


def mean_iou_single(pred, labels):
    """Calculate the mean IoU for a single object"""
    truepositive = pred * labels
    intersection = torch.sum(truepositive == 1)
    uni = torch.sum(pred == 1) + torch.sum(labels == 1) - intersection

    iou = intersection / uni
    return iou


def mean_iou(pred, labels, obj2label):
    """Calculate the mean IoU for a batch"""

    assert len(pred) == len(labels)
    bs = len(pred)
    iou_batch = 0.0
    label_mapping = {
        0: "unlabeled",
        1: "car",
        2: "bicycle",
        3: "motorcycle",
        4: "truck",
        5: "other-vehicle",
        6: "person",
        7: "bicyclist",
        8: "motorcyclist",
        9: "road",
        10: "parking",
        11: "sidewalk",
        12: "other-ground",
        13: "building",
        14: "fence",
        15: "vegetation",
        16: "trunk",
        17: "terrain",
        18: "pole",
        19: "traffic-sign",
    }
    iou_per_label = {}  # Initialize IoU for the entire batch
    for label_name in label_mapping.values():
        iou_per_label[label_name] = []
    for b in range(bs):
        pred_sample = pred[b]
        labels_sample = labels[b]
        obj_ids = torch.unique(labels_sample)
        obj_ids = obj_ids[obj_ids != 0]
        obj_num = len(obj_ids)
        iou_sample = 0.0
        for obj_id in obj_ids:
            original_label = label_mapping[obj2label[b][str(int(obj_id.item()))] & 0xFFFF]
            obj_iou = mean_iou_single(pred_sample == obj_id, labels_sample == obj_id)
            iou_sample += obj_iou
            # Accumulate IoU for each original label
            iou_per_label[original_label].append(obj_iou)

        iou_sample /= obj_num
        iou_batch += iou_sample

    iou_batch /= bs

    # Aggregate iou_per_label across batches
    average_iou_per_label = {label_name: sum(iou_list) / len(iou_list) if iou_list else None for label_name, iou_list in iou_per_label.items()}
    average_iou_per_label = {"miou_class/" + k: v for k, v in average_iou_per_label.items() if v is not None}

    return iou_batch, average_iou_per_label


def mean_iou_validation(pred, labels, obj2label):
    """Calculate the mean IoU for a batch"""
    assert len(pred) == len(labels)
    bs = len(pred)
    iou_batch = 0.0
    label_mapping = {
        0: "unlabeled",
        1: "car",
        2: "bicycle",
        3: "motorcycle",
        4: "truck",
        5: "other-vehicle",
        6: "person",
        7: "bicyclist",
        8: "motorcyclist",
        9: "road",
        10: "parking",
        11: "sidewalk",
        12: "other-ground",
        13: "building",
        14: "fence",
        15: "vegetation",
        16: "trunk",
        17: "terrain",
        18: "pole",
        19: "traffic-sign",
    }
    iou_per_label = {}  # Initialize IoU for the entire batch
    objects_info = {}  # Initialize IoU for the entire batch
    for label_name in label_mapping.values():
        iou_per_label[label_name] = []
    for obj_id, panoptic_label in obj2label[0].items():
        objects_info[obj_id] = {}
        objects_info[obj_id]["class"] = label_mapping[obj2label[0][obj_id] & 0xFFFF]

    for b in range(bs):
        pred_sample = pred[b]
        labels_sample = labels[b]
        obj_ids = torch.unique(labels_sample)
        obj_ids = obj_ids[obj_ids != 0]
        obj_num = len(obj_ids)
        iou_sample = 0.0
        for obj_id in obj_ids:
            original_label = label_mapping[obj2label[b][str(int(obj_id.item()))] & 0xFFFF]
            obj_iou = mean_iou_single(pred_sample == obj_id, labels_sample == obj_id)
            objects_info[str(int(obj_id.item()))]["miou"] = obj_iou.item()
            iou_sample += obj_iou
            # Accumulate IoU for each original label
            iou_per_label[original_label].append(obj_iou)

        iou_sample /= obj_num
        iou_batch += iou_sample

    iou_batch /= bs

    # Aggregate iou_per_label across batches
    average_iou_per_label = {label_name: sum(iou_list) / len(iou_list) if iou_list else None for label_name, iou_list in iou_per_label.items()}
    average_iou_per_label = {"miou_class/" + k: v for k, v in average_iou_per_label.items() if v is not None}

    return iou_batch, average_iou_per_label, objects_info


def mean_iou_scene(pred, labels):
    """Calculate the mean IoU for all target objects in the scene"""
    obj_ids = torch.unique(labels)
    obj_ids = obj_ids[obj_ids != 0]
    obj_num = len(obj_ids)
    iou_sample = 0.0
    iou_dict = {}
    for obj_id in obj_ids:
        obj_iou = mean_iou_single(pred == obj_id, labels == obj_id)
        iou_dict[int(obj_id)] = float(obj_iou)
        iou_sample += obj_iou

    iou_sample /= obj_num

    return iou_sample, iou_dict


def loss_weights(points, clicks, tita, alpha, beta):
    """Points closer to clicks have bigger weights. Vice versa."""
    pairwise_distances = torch.cdist(points, clicks)
    pairwise_distances, _ = torch.min(pairwise_distances, dim=1)

    weights = alpha + (beta - alpha) * (1 - torch.clamp(pairwise_distances, max=tita) / tita)

    return weights


def cal_click_loss_weights(batch_idx, raw_coords, labels, click_idx, alpha=0.8, beta=2.0, tita=0.3):
    """Calculate the loss weights for each point in the point cloud."""
    weights = []

    bs = batch_idx.max() + 1
    for i in range(bs):

        click_idx_sample = click_idx[i]
        sample_mask = batch_idx == i
        raw_coords_sample = raw_coords[sample_mask]
        all_click_idx = [np.array(v) for k, v in click_idx_sample.items()]
        all_click_idx = np.hstack(all_click_idx).astype(np.int64).tolist()
        click_points_sample = raw_coords_sample[all_click_idx]
        weights_sample = loss_weights(raw_coords_sample, click_points_sample, tita, alpha, beta)
        weights.append(weights_sample)

    return weights


def get_next_click_coo_torch(discrete_coords, unique_labels, gt, pred, pairwise_distances):
    """Sample the next click from the center of the error region"""
    zero_indices = unique_labels == 0
    one_indices = unique_labels == 1
    if zero_indices.sum() == 0 or one_indices.sum() == 0:
        return None, None, None, -1, None, None

    # point furthest from border
    center_id = torch.where(pairwise_distances == torch.max(pairwise_distances, dim=0)[0])
    center_coo = discrete_coords[one_indices, :][center_id[0][0]]
    center_label = gt[one_indices][center_id[0][0]]
    center_pred = pred[one_indices][center_id[0][0]]

    local_mask = torch.zeros(pairwise_distances.shape[0], device=discrete_coords.device)
    global_id_mask = torch.zeros(discrete_coords.shape[0], device=discrete_coords.device)
    local_mask[center_id] = 1
    global_id_mask[one_indices] = local_mask
    center_global_id = torch.argwhere(global_id_mask)[0][0]

    candidates = discrete_coords[one_indices, :]

    max_dist = torch.max(pairwise_distances)

    return center_global_id, center_coo, center_label, max_dist, candidates


def get_next_click_random(discrete_coords, unique_labels, gt, pred, pairwise_distances):
    """Sample the next click from the center of the error region"""
    zero_indices = unique_labels == 0
    one_indices = unique_labels == 1
    if zero_indices.sum() == 0 or one_indices.sum() == 0:
        return None, None, None, -1, None, None

    # point furthest from border
    center_id = random.randint(0, pairwise_distances.shape[0] - 1)
    center_coo = discrete_coords[one_indices, :][center_id]
    center_label = gt[one_indices][center_id]

    local_mask = torch.zeros(pairwise_distances.shape[0], device=discrete_coords.device)
    global_id_mask = torch.zeros(discrete_coords.shape[0], device=discrete_coords.device)
    local_mask[center_id] = 1
    global_id_mask[one_indices] = local_mask
    center_global_id = torch.argwhere(global_id_mask)[0][0]

    candidates = discrete_coords[one_indices, :]

    max_dist = torch.max(pairwise_distances)

    return center_global_id, center_coo, center_label, max_dist, candidates


def get_next_simulated_click_multi(error_cluster_ids, error_cluster_ids_mask, pred_qv, labels_qv, coords_qv, error_distances):
    """Sample the next clicks for each error region"""

    click_dict = {}
    new_click_pos = {}
    click_time_dict = {}
    click_order = 0

    random.shuffle(error_cluster_ids)

    for cluster_id in error_cluster_ids:

        error = error_cluster_ids_mask == cluster_id

        pair_distances = error_distances[cluster_id]

        # get next click candidate
        center_id, center_coo, center_gt, max_dist, candidates = get_next_click_coo_torch(coords_qv, error, labels_qv, pred_qv, pair_distances)

        if click_dict.get(str(int(center_gt))) == None:
            click_dict[str(int(center_gt))] = [int(center_id)]
            new_click_pos[str(int(center_gt))] = [center_coo]
            click_time_dict[str(int(center_gt))] = [click_order]
        else:
            click_dict[str(int(center_gt))].append(int(center_id))
            new_click_pos[str(int(center_gt))].append(center_coo)
            click_time_dict[str(int(center_gt))].append(click_order)

        click_order += 1

    click_num = len(error_cluster_ids)

    return click_dict, click_num, new_click_pos, click_time_dict


def measure_error_size(discrete_coords, unique_labels):
    """Measure error size in 3D space"""
    torch.cuda.empty_cache()

    zero_indices = unique_labels == 0  # background
    one_indices = unique_labels == 1  # foreground
    if zero_indices.sum() == 0 or one_indices.sum() == 0:
        return None, None, None, -1, None, None

    # All distances from foreground points to background points
    pairwise_distances = torch.cdist(discrete_coords[zero_indices, :], discrete_coords[one_indices, :])
    # Bg points on the border
    pairwise_distances, _ = torch.min(pairwise_distances, dim=0)

    return pairwise_distances


def get_simulated_clicks(pred_qv, labels_qv, coords_qv, current_num_clicks=None, current_click_idx=None, training=True):
    """Sample simulated clicks.
    The simulation samples next clicks from the top biggest error regions in the current iteration.
    """
    labels_qv = labels_qv.float()
    pred_label = pred_qv.float()

    # Do not generate new clicks for obj that has been clicked more than the threshold
    if current_click_idx is not None:
        for obj_id, click_ids in current_click_idx.items():
            if len(click_ids) >= 10:  # TODO: inject this as a click_threshold parameter
                # Artificially set the pred_label to labels_qv for this object (as it received the threshold number of clicks)
                pred_label[labels_qv == int(obj_id)] = int(obj_id)

    error_mask = torch.abs(pred_label - labels_qv) > 0

    if error_mask.sum() == 0:
        return None, None, None, None

    cluster_ids = labels_qv * 96 + pred_label * 11

    # error_region = coords_qv[error_mask]

    num_obj = (torch.unique(labels_qv) != 0).sum()

    error_clusters = cluster_ids[error_mask]
    error_cluster_ids = torch.unique(error_clusters)
    num_error_cluster = len(error_cluster_ids)

    error_cluster_ids_mask = torch.ones(coords_qv.shape[0], device=error_mask.device) * -1
    error_cluster_ids_mask[error_mask] = error_clusters

    ### measure the size of each error cluster and store the distance
    error_sizes = {}
    error_distances = {}

    for cluster_id in error_cluster_ids:
        error = error_cluster_ids_mask == cluster_id
        pairwise_distances = measure_error_size(coords_qv, error)

        error_distances[int(cluster_id)] = pairwise_distances
        #  TODO: why do we take the max error_distance instead of the sum?
        error_sizes[int(cluster_id)] = torch.max(pairwise_distances).tolist()

    error_cluster_ids_sorted = sorted(error_sizes, key=error_sizes.get, reverse=True)

    if training:
        if num_error_cluster >= num_obj:
            selected_error_cluster_ids = error_cluster_ids_sorted[:num_obj]
        else:
            selected_error_cluster_ids = error_cluster_ids_sorted
    else:
        if current_num_clicks == 0:
            selected_error_cluster_ids = error_cluster_ids_sorted
        else:
            selected_error_cluster_ids = error_cluster_ids_sorted[:1]

    new_clicks, new_click_num, new_click_pos, new_click_time = get_next_simulated_click_multi(selected_error_cluster_ids, error_cluster_ids_mask, pred_qv, labels_qv, coords_qv, error_distances)

    return new_clicks, new_click_num, new_click_pos, new_click_time


def extend_clicks(current_clicks, current_clicks_time, new_clicks, new_click_time):
    """Append new click to existing clicks"""

    current_click_num = sum([len(c) for c in current_clicks_time.values()])

    for obj_id, click_ids in new_clicks.items():
        current_clicks[obj_id].extend(click_ids)
        current_clicks_time[obj_id].extend([t + current_click_num for t in new_click_time[obj_id]])

    return current_clicks, current_clicks_time
