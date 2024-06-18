import MinkowskiEngine as ME
import numpy as np
import torch


class VoxelizeCollate:
    def __init__(
        self,
        mode="train",  # train, validation, or test
        ignore_label=255,
        voxel_size=0.05,
    ):
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode

    def __call__(self, batch):

        (
            scene_names,
            coordinates,
            full_coordinates,
            features,
            labels,
            original_labels,
            inverse_maps,
            unique_maps,
            num_points,
            sequences,
            click_idxs,
            obj2labels,
        ) = ([], [], [], [], [], [], [], [], [], [], [], [])

        for sample in batch:
            scene_names.append(sample["sequence"])
            full_coordinates.append(sample["features"][:, :3])
            click_idxs.append(sample["click_idx"])
            obj2labels.append(sample["obj2label"])
            original_labels.append(sample["labels"])
            num_points.append(sample["num_points"])
            sequences.append(sample["sequence"])
            sample_c, sample_f, sample_l, inverse_map, unique_map = voxelize(
                sample["coordinates"],
                sample["features"],
                sample["labels"],
                self.voxel_size,
            )
            inverse_maps.append(inverse_map)
            unique_maps.append(unique_map)
            coordinates.append(sample_c)
            features.append(sample_f)
            labels.append(sample_l)

        # Concatenate all lists
        target = generate_target(features, labels, original_labels, self.ignore_label, inverse_maps, unique_maps, obj2labels)
        coordinates, features = ME.utils.sparse_collate(coordinates, features)
        # TODO: why do we need here then the time i.e. features are: [original_x, original_y, original_z , time, intensity, distance]
        raw_coordinates = features[:, :3]  # [original_x, original_y, original_z , time]
        features = features[:, 4:]  # [intensity, distance]
        collated_data = generate_collated_data(
            mode=self.mode,
            scene_names=scene_names,
            coordinates=coordinates,
            full_coordinates=full_coordinates,
            features=features,
            raw_coordinates=raw_coordinates,
            num_points=num_points,
            sequences=sequences,
            click_idx=click_idxs,
        )

        return (
            collated_data,  # collated_data
            target,  # labels
        )


def voxelize(coordinates, features, labels, voxel_size):
    if coordinates.shape[1] == 4:
        voxel_size = np.array([voxel_size, voxel_size, voxel_size, 1])
    sample_c, sample_f, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=coordinates,
        features=features,
        return_index=True,
        return_inverse=True,
        quantization_size=voxel_size,
    )

    sample_f = torch.from_numpy(sample_f).float()
    sample_l = torch.from_numpy(labels[unique_map])
    return sample_c, sample_f, sample_l, inverse_map, unique_map


def generate_target(features, labels, original_labels, ignore_label, inverse_maps, unique_maps, obj2labels):
    # TODO do we really need the entire original_labels?
    target = {}
    target["labels"] = labels
    target["labels_full"] = original_labels
    target["inverse_maps"] = inverse_maps
    target["unique_maps"] = unique_maps
    target["obj2labels"] = obj2labels

    # Prepare bounding boxes
    batch_bboxs = []
    for feat, lb in zip(features, labels):
        scene_bboxs = {}
        obj_in_scene = lb.unique()
        raw_coords = feat[:, :3]
        raw_coords = (raw_coords - raw_coords.min(0)[0]) / (raw_coords.max(0)[0] - raw_coords.min(0)[0])
        for obj in obj_in_scene:
            if obj == 0:
                continue
            mask = lb == obj
            mask_coords = raw_coords[mask, :]
            obj_bbox = torch.hstack((mask_coords.mean(0), mask_coords.max(0)[0] - mask_coords.min(0)[0]))
            scene_bboxs[int(obj)] = obj_bbox
        batch_bboxs.append(scene_bboxs)

    target["bboxs"] = batch_bboxs

    return target


def generate_collated_data(
    mode,
    scene_names,
    coordinates,
    features,
    raw_coordinates,
    full_coordinates,
    num_points=None,
    sequences=None,
    click_idx=None,
):
    collated_data = {}
    collated_data["scene_names"] = scene_names
    collated_data["coordinates"] = coordinates
    collated_data["features"] = features
    collated_data["raw_coordinates"] = raw_coordinates
    collated_data["num_points"] = num_points
    collated_data["sequences"] = sequences
    collated_data["click_idx"] = click_idx
    if mode == "validation":
        collated_data["full_coordinates"] = full_coordinates

    return collated_data


class NoGpu:
    def __init__(
        self,
        coordinates,
        features,
        raw_coordinates,
        inverse_maps=None,
        unique_maps=None,
        num_points=None,
        sequences=None,
        click_idx=None,
        obj2label=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.raw_coordinates = raw_coordinates
        self.inverse_maps = inverse_maps
        self.unique_maps = unique_maps
        self.num_points = num_points
        self.sequences = sequences
        self.click_idx = click_idx
        self.obj2label = obj2label
