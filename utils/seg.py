# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------
import torch
import numpy as np
import random
from scipy.spatial import ConvexHull


def mean_iou_single(pred, labels):
    """Calculate the mean IoU for a single object"""
    truepositive = pred * labels
    intersection = torch.sum(truepositive == 1)
    uni = torch.sum(pred == 1) + torch.sum(labels == 1) - intersection

    iou = intersection / uni
    return iou


def mean_iou(pred, labels, obj2label, dataset_type="semantickitti"):
    """Calculate the mean IoU for a batch"""

    assert len(pred) == len(labels)
    bs = len(pred)
    iou_batch = 0.0
    label_mapping = get_label_mapping(dataset_type)[0]
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
            if dataset_type == "semantickitti":
                original_label = label_mapping[obj2label[b][str(int(obj_id.item()))] & 0xFFFF]
            elif "nuScenes" in dataset_type:
                original_label = label_mapping[obj2label[b][str(int(obj_id.item()))] // 1000]
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


def mean_iou_validation(pred, labels, obj2label, dataset_type="semantickitti"):
    """Calculate the mean IoU for a batch"""
    assert len(pred) == len(labels)
    bs = len(pred)
    iou_batch = 0.0
    label_mapping = get_label_mapping(dataset_type)[0]
    iou_per_label = {}  # Initialize IoU for the entire batch
    objects_info = {}  # Initialize IoU for the entire batch
    for label_name in label_mapping.values():
        iou_per_label[label_name] = []
    for obj_id, panoptic_label in obj2label[0].items():
        objects_info[obj_id] = {}
        if dataset_type == "semantickitti":
            objects_info[obj_id]["class"] = label_mapping[obj2label[0][obj_id] & 0xFFFF]
        elif "nuScenes" in dataset_type:
            objects_info[obj_id]["class"] = label_mapping[obj2label[0][obj_id] // 1000]

    for b in range(bs):
        pred_sample = pred[b]
        labels_sample = labels[b]
        obj_ids = torch.unique(labels_sample)
        obj_ids = obj_ids[obj_ids != 0]
        obj_num = len(obj_ids)
        iou_sample = 0.0
        for obj_id in obj_ids:
            if dataset_type == "semantickitti":
                original_label = label_mapping[obj2label[b][str(int(obj_id.item()))] & 0xFFFF]
            elif "nuScenes" in dataset_type:
                original_label = label_mapping[obj2label[b][str(int(obj_id.item()))] // 1000]
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


def get_objects_iou(pred, labels):
    """Calculate the mean IoU for a batch"""
    assert len(pred) == len(labels)
    bs = len(pred)
    objects_info = []  # Initialize IoU for the entire batch

    for b in range(bs):
        objects_info.append({})
        pred_sample = pred[b]
        labels_sample = labels[b]
        obj_ids = torch.unique(labels_sample)
        # obj_ids = obj_ids[obj_ids != 0]
        # obj_num = len(obj_ids)
        iou_sample = 0.0
        for obj_id in obj_ids:
            obj_iou = mean_iou_single(pred_sample == obj_id, labels_sample == obj_id)
            objects_info[b][int(obj_id)] = obj_iou.item()
            iou_sample += obj_iou

    return objects_info


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


def get_next_click_random(coords, error):
    # Extract the coordinates of the wrongly classified points
    wrong_points = coords[error]
    # Select random point from the error region
    # centroid = torch.mean(wrong_points, dim=0)
    # closest_point_index = torch.argmin(distances)
    selected_point_index = random.randint(0, wrong_points.shape[0] - 1)
    selected_point = wrong_points[selected_point_index]
    # Find the point closest to the selected_point
    distance = 0

    return distance, selected_point, selected_point_index


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
            if obj_id != "0":  # background can receive as many clicks as needed
                if len(click_ids) >= 20:  # TODO: inject this as a click_threshold parameter
                    # Artificially set the pred_label to labels_qv for this object (as it received the threshold number of clicks)
                    pred_label[labels_qv == int(obj_id)] = int(obj_id)

    error_mask = torch.abs(pred_label - labels_qv) > 0

    if error_mask.sum() == 0:
        return None, None, None, None

    cluster_ids = labels_qv * 9973 + pred_label * 11

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
    center_ids, center_coos, center_gts = {}, {}, {}

    for cluster_id in error_cluster_ids:
        error = error_cluster_ids_mask == cluster_id

        #### Original implementation
        # pairwise_distances = measure_error_size(coords_qv, error)
        # error_distances[int(cluster_id)] = pairwise_distances
        # error_sizes[int(cluster_id)] = torch.max(pairwise_distances).tolist()

        #### Compute the AABB (Axis-Aligned Bounding Box) for the wrongly classified points
        clusters_error_distance, furthest_point, furthest_point_index = find_closest_point_to_centroid(coords_qv, error)
        original_indices = torch.nonzero(error).squeeze()  # Find the index of the furthest point in the original coords_qv
        if original_indices.dim() == 0:  # There is only one point in the error region
            furthest_point_original_index = original_indices.item()
        else:
            furthest_point_original_index = original_indices[furthest_point_index]
        error_sizes[int(cluster_id)], center_ids[int(cluster_id)], center_coos[int(cluster_id)], center_gts[int(cluster_id)] = clusters_error_distance, furthest_point_original_index, furthest_point, labels_qv[furthest_point_original_index]

        #### Compute the OBB (Oriented bounding box) for the wrongly classified points
        # error_points = coords_qv[error]
        # furthest_point = find_furthest_point_hull(error_points)

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

    # new_clicks, new_click_num, new_click_pos, new_click_time = get_next_simulated_click_multi(selected_error_cluster_ids, error_cluster_ids_mask, pred_qv, labels_qv, coords_qv, error_distances)
    new_clicks, new_click_num, new_click_pos, new_click_time = get_next_simulated_click_multi_bbox(selected_error_cluster_ids, error_cluster_ids_mask, center_ids, center_coos, center_gts)
    return new_clicks, new_click_num, new_click_pos, new_click_time


def get_iou_based_simulated_clicks(pred_qv, labels_qv, coords_qv, current_num_clicks=None, current_click_idx=None, training=True, objects_info={}, cluster_dict={}):
    labels_qv = labels_qv.float()
    pred_label = pred_qv.float()

    # Do not generate new clicks for obj that has been clicked more than the threshold
    if current_click_idx is not None:
        for obj_id, click_ids in current_click_idx.items():
            if obj_id != "0":  # background can receive as many clicks as needed
                if len(click_ids) >= 20:  # TODO: inject this as a click_threshold parameter
                    # Artificially set the pred_label to labels_qv for this object (as it received the threshold number of clicks)
                    pred_label[labels_qv == int(obj_id)] = int(obj_id)

    error_mask = torch.abs(pred_label - labels_qv) > 0

    if error_mask.sum() == 0:
        return None, None, None, None, None

    cluster_ids = labels_qv * 9973 + pred_label * 11
    # cluster_ids = labels_qv

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
    center_ids, center_coos, center_gts = {}, {}, {}

    for cluster_id in error_cluster_ids:
        error = error_cluster_ids_mask == cluster_id
        if int(cluster_id) not in cluster_dict.keys():
            cluster_dict[int(cluster_id)] = 1
        else:
            cluster_dict[int(cluster_id)] += 1

        #### Original implementation
        # pairwise_distances = measure_error_size(coords_qv, error)
        # error_distances[int(cluster_id)] = pairwise_distances
        # error_sizes[int(cluster_id)] = torch.max(pairwise_distances).tolist()

        #### Compute the AABB (Axis-Aligned Bounding Box) for the wrongly classified points
        if cluster_dict[int(cluster_id)] < 4:
            clusters_error_distance, furthest_point, furthest_point_index = find_closest_point_to_centroid(coords_qv, error)
        else:  # centroid click was selected already 3 times, now select the furthest point from the centroid
            clusters_error_distance, furthest_point, furthest_point_index = get_next_click_random(coords_qv, error)
        original_indices = torch.nonzero(error).squeeze()
        if original_indices.dim() == 0:  # There is only one point in the error region
            furthest_point_original_index = original_indices.item()
        else:
            furthest_point_original_index = original_indices[furthest_point_index]
        center_ids[int(cluster_id)], center_coos[int(cluster_id)], center_gts[int(cluster_id)] = furthest_point_original_index, furthest_point, labels_qv[furthest_point_original_index]

        correct_label, _ = decode_cluster_ids(cluster_id)
        error_region_percetage = torch.sum(error) / (labels_qv == correct_label).count_nonzero()
        # error_sizes[int(cluster_id)] = error_point_portion / objects_info[int(correct_label)]  # Divide by iou
        if objects_info[int(correct_label)] == 0:
            error_sizes[int(cluster_id)] = np.inf
        else:
            error_sizes[int(cluster_id)] = error_region_percetage / (objects_info[int(correct_label)])

        #### Compute the OBB (Oriented bounding box) for the wrongly classified points
        # error_points = coords_qv[error]
        # furthest_point = find_furthest_point_hull(error_points)

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

    # new_clicks, new_click_num, new_click_pos, new_click_time = get_next_simulated_click_multi(selected_error_cluster_ids, error_cluster_ids_mask, pred_qv, labels_qv, coords_qv, error_distances)
    new_clicks, new_click_num, new_click_pos, new_click_time = get_next_simulated_click_multi_bbox(selected_error_cluster_ids, error_cluster_ids_mask, center_ids, center_coos, center_gts)
    return new_clicks, new_click_num, new_click_pos, new_click_time, cluster_dict


def extend_clicks(current_clicks, current_clicks_time, new_clicks, new_click_time):
    """Append new click to existing clicks"""

    current_click_num = sum([len(c) for c in current_clicks_time.values()])

    for obj_id, click_ids in new_clicks.items():
        current_clicks[obj_id].extend(click_ids)
        current_clicks_time[obj_id].extend([t + current_click_num for t in new_click_time[obj_id]])

    return current_clicks, current_clicks_time


def find_furthest_point_hull(error_points):
    # Compute the convex hull
    error_points = error_points.cpu().numpy()
    hull = ConvexHull(error_points)

    # Calculate distances from each point to each hull face
    max_min_dist = 0
    furthest_point = None

    for point in error_points:
        min_dist_to_hull = np.inf
        for simplex in hull.simplices:
            # Get the vertices of this face
            vertices = error_points[simplex]
            # Create a plane equation from the vertices
            v0, v1, v2 = vertices
            normal = np.cross(v1 - v0, v2 - v0)
            normal /= np.linalg.norm(normal)  # Normalize the normal vector
            d = -np.dot(normal, v0)  # Plane equation: ax + by + cz + d = 0
            # Distance from point to this plane
            distance = abs(np.dot(normal, point) + d) / np.linalg.norm(normal)
            # Update the minimum distance for this point
            if distance < min_dist_to_hull:
                min_dist_to_hull = distance
        # Update the global maximum of minimum distances
        if min_dist_to_hull > max_min_dist:
            max_min_dist = min_dist_to_hull
            furthest_point = point

    return furthest_point


def find_furthest_point_bbox(coords, error):
    # AABB: Axis-Aligned Bounding Box
    wrong_points = coords[error]
    # Compute the AABB for the wrongly classified points
    min_coords = torch.min(wrong_points, dim=0).values
    max_coords = torch.max(wrong_points, dim=0).values
    # Compute the distance of each point from the nearest border of the AABB
    distances = torch.min(wrong_points - min_coords, max_coords - wrong_points)
    min_distances = torch.min(distances, dim=1).values
    # Find the point furthest away from any of the borders
    furthest_point_index = torch.argmax(min_distances)
    distance = torch.max(min_distances)
    furthest_point = wrong_points[furthest_point_index]

    return distance, furthest_point, furthest_point_index


def find_closest_point_to_centroid(coords, error):
    # Extract the coordinates of the wrongly classified points
    wrong_points = coords[error]
    # Compute the centroid of the wrongly classified points
    centroid = torch.mean(wrong_points, dim=0)
    # Compute the Euclidean distance of each point from the centroid
    distances = torch.norm(wrong_points - centroid, dim=1)
    # Find the point closest to the centroid
    closest_point_index = torch.argmin(distances)
    distance = torch.min(distances)
    closest_point = wrong_points[closest_point_index]

    return distance, closest_point, closest_point_index


def get_next_simulated_click_multi_bbox(error_cluster_ids, error_cluster_ids_mask, center_ids, center_coos, center_gts):
    """Sample the next clicks for each error region"""

    click_dict = {}
    new_click_pos = {}
    click_time_dict = {}
    click_order = 0

    random.shuffle(error_cluster_ids)

    for cluster_id in error_cluster_ids:
        center_id, center_coo, center_gt = center_ids[int(cluster_id)], center_coos[int(cluster_id)], center_gts[int(cluster_id)]
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


def decode_cluster_ids(cluster_ids, prime1=9973, prime2=11):
    # Recover pred_label using modular arithmetic
    pred_label = (cluster_ids % prime1) // prime2

    # Recover labels_qv by isolating it
    labels_qv = (cluster_ids - pred_label * prime2) // prime1

    return labels_qv, pred_label


def get_label_mapping(dataset_type):
    if dataset_type == "semantickitti":
        label_mapping = {0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck", 5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist", 9: "road", 10: "parking", 11: "sidewalk", 12: "other-ground", 13: "building", 14: "fence", 15: "vegetation", 16: "trunk", 17: "terrain", 18: "pole", 19: "traffic-sign"}
        things_labels = [1, 2, 3, 4, 5, 6, 7, 8]  # [1:car,  2:bicycle,  3:motorcycle,  4:truck,  5:other-vehicle,  6:person,  7:bicyclist,  8:motorcyclist ]
        stuff_labels = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  #  [9:road,  10:parking, 11:sidewalk ,12:other-ground ,13:building ,14:fence ,15:vegetation ,16:trunk ,17:terrain ,18:pole ,19:traffic-sign]
    elif dataset_type == "nuScenes_challenge":
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
        things_labels = [9, 14, 15, 16, 17, 18, 21, 2, 3, 4, 6, 12, 22, 23]
        stuff_labels = [24, 25, 26, 27, 28, 30]
        ignore_labels = [1, 5, 7, 8, 10, 11, 13, 19, 20, 0, 29, 31]
    elif dataset_type == "nuScenes_general":
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
        things_labels = []
        stuff_labels = []
    return label_mapping, things_labels
