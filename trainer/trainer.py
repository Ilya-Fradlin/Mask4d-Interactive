import wandb
import math
import hydra
import copy
import random
import warnings
import MinkowskiEngine as ME
import pytorch_lightning as pl
import torch
from contextlib import nullcontext
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import utils.misc as utils
from utils.utils import generate_wandb_objects3d
from utils.seg import mean_iou, mean_iou_validation, mean_iou_scene, cal_click_loss_weights, extend_clicks, get_simulated_clicks, get_iou_based_simulated_clicks, get_objects_iou
from datasets.utils import VoxelizeCollate
from datasets.lidar import LidarDataset
from datasets.lidar_nuscenes import LidarDatasetNuscenes
from models import Interactive4D
from models.criterion import SetCriterion
from models.metrics.utils import IoU_at_numClicks, NumClicks_for_IoU, mIoU_per_class_metric, mIoU_metric, losses_metric


class ObjectSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # model
        self.interactive4d = Interactive4D(
            num_heads=8, num_decoders=3, num_levels=1, hidden_dim=128, dim_feedforward=1024, shared_decoder=False, num_bg_queries=10, dropout=0.0, pre_norm=False, aux=True, voxel_size=config.data.voxel_size, sample_sizes=[4000, 8000, 16000, 32000], use_objid_attention=config.general.use_objid_attention
        )
        self.optional_freeze = nullcontext
        self.dataset_type = self.config.general.dataset

        weight_dict = {
            # self.config.model.num_queries
            "loss_bce": self.config.loss.bce_loss_coef,
            "loss_dice": self.config.loss.dice_loss_coef,
            "loss_bbox": self.config.loss.bbox_loss_coef,
        }

        # TODO: check this aux loss
        if config.loss.aux:
            aux_weight_dict = {}
            for i in range(self.interactive4d.num_decoders * self.interactive4d.num_levels):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(losses=["bce", "dice", "bbox"], weight_dict=weight_dict)

        # Initiatie the monitoring metric
        self.log("mIoU_monitor", 0, sync_dist=True, logger=False)
        # Training metrics
        self.losses_metric = losses_metric()
        self.mIoU_metric = mIoU_metric()
        self.mIoU_per_class_metric = mIoU_per_class_metric(training=True)
        # Validation metrics
        # self.mIoU_per_class_metric_validation = mIoU_per_class_metric(training=False)
        self.iou_at_numClicks = IoU_at_numClicks(num_clicks=self.config.general.clicks_of_interest)
        self.numClicks_for_IoU = NumClicks_for_IoU(iou_thresholds=self.config.general.iou_targets)
        self.validation_metric_logger = utils.MetricLogger(delimiter="  ")

        self.save_hyperparameters()

    def forward(self, x, raw_coordinates=None, feats=None, click_idx=None, is_eval=False):
        with self.optional_freeze():
            x = self.interactive4d(x, raw_coordinates=raw_coordinates, feats=feats, is_eval=is_eval)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch

        coords = data["coordinates"]
        raw_coords = data["raw_coordinates"]
        feats = data["features"]
        labels = target["labels"]
        click_idx = data["click_idx"]
        obj2label = [mapping[0] for mapping in target["obj2labels"]]
        batch_indicators = coords[:, 0]
        batch_size = batch_indicators.max() + 1

        click_idx, obj2label, labels, bboxs_target = self.verify_labels_post_quantization(labels, target["bboxs"], click_idx, obj2label, batch_size)

        # Check if there is more than just the background in the scene
        for idx in range(batch_size):
            if len(labels[idx].unique()) < 2:
                # If there is only the background in the scene, skip the scene
                print("after quantization, only background in the scene")
                print(f"From Rank: {self.global_rank}, The corrupted scene is: '{data['scene_names'][idx]}'!")
                return None

        data = ME.SparseTensor(coordinates=coords, features=feats, device=self.device)
        pcd_features, aux, coordinates, pos_encodings_pcd = self.interactive4d.forward_backbone(data, raw_coordinates=raw_coords)

        click_time_idx = copy.deepcopy(click_idx)

        #########  1. pre interactive sampling  #########
        click_idx, click_time_idx = self.pre_interactive_sampling(
            pcd_features=pcd_features,
            aux=aux,
            coordinates=coordinates,
            raw_coords=raw_coords,
            batch_indicators=batch_indicators,
            pos_encodings_pcd=pos_encodings_pcd,
            labels=labels,
            click_idx=click_idx,
            click_time_idx=click_time_idx,
        )
        self.interactive4d.train()

        #########  2. real forward pass  #########
        outputs = self.interactive4d.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=click_idx, click_time_idx=click_time_idx)

        ######### 3. loss back propagation #########
        click_weights = cal_click_loss_weights(batch_indicators, raw_coords, torch.cat(labels), click_idx)
        loss_dict = self.criterion(outputs, labels, bboxs_target, obj2label, click_weights)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            exit(1)

        with torch.no_grad():
            pred_logits = outputs["pred_masks"]
            pred = [p.argmax(-1) for p in pred_logits]
            general_miou, label_miou_dict = mean_iou(pred, labels, obj2label, self.dataset_type)
            label_miou_dict = {"trainer/" + k: v for k, v in label_miou_dict.items()}

        self.log("trainer/loss", losses, prog_bar=True, on_step=True)
        self.log("trainer/mIoU", general_miou, prog_bar=True, on_step=True)
        self.losses_metric.update(losses)
        self.mIoU_metric.update(general_miou)
        self.mIoU_per_class_metric.update(label_miou_dict)

        return losses

    def on_train_epoch_end(self):
        miou_epoch = self.mIoU_metric.compute()
        miou_per_class_epoch = self.mIoU_per_class_metric.compute()
        losses_epoch = self.losses_metric.compute()
        print(f"End Epoch mIoU: {miou_epoch},  loss: {losses_epoch}, class_mIoU: {miou_per_class_epoch}", flush=True)

        self.log_dict(miou_per_class_epoch)
        self.log("mIoU_epoch", miou_epoch)
        self.log("loss_epoch", losses_epoch)

        self.mIoU_metric.reset()
        self.mIoU_per_class_metric.reset()
        self.losses_metric.reset()

    def validation_step(self, batch, batch_idx):

        data, target = batch
        scene_names = data["scene_names"]
        coords = data["coordinates"]
        raw_coords = data["raw_coordinates"]
        full_raw_coords = data["full_coordinates"]
        feats = data["features"]
        labels = target["labels"]
        labels_full = [torch.from_numpy(l).to(coords) for l in target["labels_full"]]
        click_idx = data["click_idx"]
        inverse_maps = target["inverse_maps"]
        cluster_dict = {}

        obj2label = [mapping[0] for mapping in target["obj2labels"]]
        batch_indicators = coords[:, 0]
        num_obj = [len(mapping.keys()) for mapping in obj2label]
        batch_size = batch_indicators.max() + 1
        current_num_clicks = 0

        # Remove objects which are not in the scene (due to quantization)
        click_idx, obj2label, labels, bboxs_target = self.verify_labels_post_quantization(labels, target["bboxs"], click_idx, obj2label, batch_size)
        click_time_idx = copy.deepcopy(click_idx)

        # Check if there is more than just the background in the scene
        for idx in range(batch_size):
            if len(labels[idx].unique()) < 2:
                # If there is only the background in the scene, skip the scene
                print("after quantization, only background in the scene")
                return

        ###### interactive evaluation ######
        data = ME.SparseTensor(coordinates=coords, features=feats, device=self.device)
        pcd_features, aux, coordinates, pos_encodings_pcd = self.interactive4d.forward_backbone(data, raw_coordinates=raw_coords)

        # TODO: check how is the max_num_clicks update more than 1 batch size
        iou_targets = self.config.general.iou_targets
        if iou_targets[-1] != 9999:
            iou_targets.append(9999)  # serving as a stop condition
        max_num_clicks = num_obj[0] * self.config.general.max_num_clicks
        next_iou_target_indices = {idx: 0 for idx in range(batch_size)}
        while current_num_clicks <= max_num_clicks:
            if current_num_clicks == 0:
                pred = [torch.zeros(l.shape).to(coords) for l in labels]
            else:
                outputs = self.interactive4d.forward_mask(
                    pcd_features,
                    aux,
                    coordinates,
                    pos_encodings_pcd,
                    click_idx=click_idx,
                    click_time_idx=click_time_idx,
                )
                pred_logits = outputs["pred_masks"]
                pred = [p.argmax(-1) for p in pred_logits]
                # pred = labels

            if current_num_clicks != 0:
                click_weights = cal_click_loss_weights(batch_indicators, raw_coords, torch.cat(labels), click_idx)
                loss_dict = self.criterion(outputs, labels, bboxs_target, obj2label, click_weights)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
                loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}

            updated_pred = []

            for idx in range(batch_size):
                sample_mask = batch_indicators == idx
                sample_pred = pred[idx]

                if current_num_clicks != 0:
                    # update prediction with sparse gt
                    for obj_id, cids in click_idx[idx].items():
                        sample_pred[cids] = int(obj_id)
                    updated_pred.append(sample_pred)

                sample_labels = labels[idx]
                sample_raw_coords = raw_coords[sample_mask]
                sample_pred_full = sample_pred[inverse_maps[idx]]

                sample_labels_full = labels_full[idx]
                # mean_iou_scene here is calculated for the entire scene (with all the points! not just the ones responsible for the quantized values)
                sample_iou, _ = mean_iou_scene(sample_pred_full, sample_labels_full)

                # Logging IoU@1, IoU@2, IoU@3, IoU@4, IoU@5
                average_clicks_per_obj = current_num_clicks / num_obj[idx]
                if average_clicks_per_obj in self.config.general.clicks_of_interest:
                    self.iou_at_numClicks.update(iou=sample_iou.item(), noc=average_clicks_per_obj)

                # Logging NoC@50, NoC@65, NoC@80, NoC@85, NoC@90
                if iou_targets[next_iou_target_indices[idx]] < sample_iou:
                    while iou_targets[next_iou_target_indices[idx]] < sample_iou:
                        self.numClicks_for_IoU.update(iou=iou_targets[next_iou_target_indices[idx]], noc=average_clicks_per_obj)
                        next_iou_target_indices[idx] += 1
                        if next_iou_target_indices[idx] == len(iou_targets) - 1:
                            break

                if updated_pred == []:
                    objects_info = get_objects_iou(pred, labels)
                else:
                    objects_info = get_objects_iou(updated_pred, labels)
                new_clicks, new_clicks_num, new_click_pos, new_click_time, cluster_dict = get_iou_based_simulated_clicks(
                    sample_pred,
                    sample_labels,
                    sample_raw_coords,
                    current_num_clicks,
                    objects_info=objects_info[idx],
                    current_click_idx=click_idx[idx],
                    training=False,
                    cluster_dict=cluster_dict,
                )

                ### add new clicks ###
                if new_clicks is not None:
                    click_idx[idx], click_time_idx[idx] = extend_clicks(click_idx[idx], click_time_idx[idx], new_clicks, new_click_time)

            if current_num_clicks != 0:
                # mean_iou here is calculated for just with the points responsible for the quantized values!
                general_miou, label_miou_dict, objects_info = mean_iou_validation(updated_pred, labels, obj2label, self.dataset_type)

                label_miou_dict = {"validation/" + k: v for k, v in label_miou_dict.items()}
                self.log("validation/full_mIoU", sample_iou, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=True)
                self.log("validation/quantized_mIoU", general_miou, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=True)
                self.log_dict(label_miou_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.validation_metric_logger.update(mIoU_quantized=general_miou)
                self.validation_metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

            if current_num_clicks == 0:
                new_clicks_num = num_obj[idx]
            else:
                new_clicks_num = 1
            current_num_clicks += new_clicks_num

        pred = 0

        # Update NoC@target for all the targets that have not been reached:
        for idx in range(batch_size):
            while next_iou_target_indices[idx] != len(iou_targets) - 1:
                self.numClicks_for_IoU.update(iou=iou_targets[next_iou_target_indices[idx]], noc=max_num_clicks)
                next_iou_target_indices[idx] += 1

        if (batch_idx % self.config.general.visualization_frequency == 0) or (general_miou < 0.4):  # Condition for visualization logging
            # choose a random scene to visualize from the batch
            chosen_scene_idx = random.randint(0, batch_size - 1)
            scene_name = scene_names[chosen_scene_idx]
            scene_name = scene_name.split("/")[-1].split(".")[0]
            sample_raw_coords_full = full_raw_coords[chosen_scene_idx]
            sample_mask = batch_indicators == chosen_scene_idx
            sample_raw_coords = raw_coords[sample_mask]
            sample_pred = updated_pred[chosen_scene_idx]
            sample_labels = labels[chosen_scene_idx]
            sample_click_idx = click_idx[chosen_scene_idx]

            # general_miou, label_miou_dict, obj2label
            # gt_scene, pred_scene = generate_wandb_objects3d(sample_raw_coords, sample_labels, sample_pred, sample_click_idx, objects_info)
            gt_scene, gt_full_scene, pred_scene, pred_full_scene = generate_wandb_objects3d(sample_raw_coords, sample_raw_coords_full, sample_labels, sample_labels_full, sample_pred, sample_pred_full, sample_click_idx, objects_info)
            if sample_iou < 0.4:
                print("Logging visualization data for low mIoU scene")
            wandb.log({f"point_scene/ground_truth_ quantized_{scene_name}": gt_scene})
            wandb.log({f"point_scene/ground_truth_full_{scene_name}": gt_full_scene})
            wandb.log({f"point_scene/prediction_{scene_name}_full_sample_iou_{sample_iou:.2f}": pred_full_scene})
            wandb.log({f"point_scene/prediction_{scene_name}_quantized_iou_{general_miou:.2f}": pred_scene})

        # self.config.general.visualization_dir,
        # self.validation_step_outputs.append(pred)

    def on_validation_epoch_end(self):
        print("\n")
        print("--------- Evaluating Validation Performance  -----------")
        warnings.filterwarnings("ignore", message="The ``compute`` method of metric NumClicks_for_IoU was called before the ``update`` method", category=UserWarning)
        results_dict = {}
        results_dict["mIoU_quantized"] = self.validation_metric_logger.meters["mIoU_quantized"].global_avg
        # Evaluate the NoC@IoU Metric
        metrics_dictionary, iou_thresholds = self.numClicks_for_IoU.compute()
        for iou in iou_thresholds:
            noc = metrics_dictionary[iou]["noc"]
            count = metrics_dictionary[iou]["count"]
            results_dict[f"scenes_reached_{iou}_iou"] = count.item()
            if count == 0:
                results_dict[f"NoC@{iou}"] = 0  # or return a default value or raise an error
            else:
                results_dict[f"NoC@{iou}"] = (noc / count).item()

        # Evaluate the IoU@NoC Metric
        metrics_dictionary, evaluated_num_clicks = self.iou_at_numClicks.compute()
        for noc in evaluated_num_clicks:
            iou = metrics_dictionary[noc]["iou"]
            count = metrics_dictionary[noc]["count"]
            if count == 0:
                results_dict[f"IoU@{noc}"] = 0  # or return a default value or raise an error
            else:
                results_dict[f"IoU@{noc}"] = (iou / count).item()
        print("\n")
        print(results_dict)
        stats = {k: meter.global_avg for k, meter in self.validation_metric_logger.meters.items()}
        stats.update(results_dict)
        self.log_dict(
            {
                "validation/epoch": self.current_epoch,
                "validation/loss_epoch": stats["loss"],
                "validation/loss_bce_epoch": stats["loss_bce"],
                "validation/loss_dice_epoch": stats["loss_dice"],
                "validation/mIoU_quantized_epoch": stats["mIoU_quantized"],
                "validation/Interactive_metrics/NoC_50": stats["NoC@50"],
                "validation/Interactive_metrics/scenes_reached_50_iou": stats["scenes_reached_50_iou"],
                "validation/Interactive_metrics/NoC_65": stats["NoC@65"],
                "validation/Interactive_metrics/scenes_reached_65_iou": stats["scenes_reached_65_iou"],
                "validation/Interactive_metrics/NoC_80": stats["NoC@80"],
                "validation/Interactive_metrics/scenes_reached_80_iou": stats["scenes_reached_80_iou"],
                "validation/Interactive_metrics/NoC_85": stats["NoC@85"],
                "validation/Interactive_metrics/scenes_reached_85_iou": stats["scenes_reached_85_iou"],
                "validation/Interactive_metrics/NoC_90": stats["NoC@90"],
                "validation/Interactive_metrics/scenes_reached_90_iou": stats["scenes_reached_90_iou"],
                "validation/Interactive_metrics/IoU_1": stats["IoU@1"],
                "validation/Interactive_metrics/IoU_2": stats["IoU@2"],
                "validation/Interactive_metrics/IoU_3": stats["IoU@3"],
                "validation/Interactive_metrics/IoU_4": stats["IoU@4"],
                "validation/Interactive_metrics/IoU_5": stats["IoU@5"],
                "validation/Interactive_metrics/IoU_5": stats["IoU@6"],
                "validation/Interactive_metrics/IoU_5": stats["IoU@7"],
                "validation/Interactive_metrics/IoU_5": stats["IoU@8"],
                "validation/Interactive_metrics/IoU_5": stats["IoU@9"],
                "validation/Interactive_metrics/IoU_5": stats["IoU@10"],
            }
        )
        # self.validation_step_outputs.clear()  # free memory
        self.validation_metric_logger = utils.MetricLogger(delimiter="  ")  # reset metric

    def test_step(self, batch, batch_idx):
        # TODO
        pass

    def test_epoch_end(self, outputs):
        # TODO
        return {}

    def configure_optimizers(self):
        # Adjust the learning rate based on the number of GPUs
        # Optimizer:
        # self.config.optimizer.lr = self.config.optimizer.lr * math.sqrt(self.config.trainer.num_nodes)
        self.config.optimizer.lr = self.config.optimizer.lr * self.config.trainer.num_nodes
        optimizer = CustomAdamW(params=self.parameters(), lr=self.config.optimizer.lr)

        # Scehduler:
        steps_per_epoch = math.ceil(len(self.train_dataloader()) / (self.config.trainer.num_devices * self.config.trainer.num_nodes))
        lr_scheduler = OneCycleLR(max_lr=self.config.optimizer.lr, epochs=self.config.trainer.max_epochs, steps_per_epoch=steps_per_epoch, optimizer=optimizer)
        scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}

        print(f"optimizer_lr: {self.config.optimizer.lr}, scheduler_steps_per_epoch:{steps_per_epoch}")
        return [optimizer], [scheduler_config]

    def setup(self, stage):
        self.train_dataset = LidarDataset(
            data_dir=self.config.data.datasets.data_dir,
            add_distance=self.config.data.datasets.add_distance,
            sweep=self.config.data.datasets.sweep,
            segment_full_scene=self.config.data.datasets.segment_full_scene,
            obj_type=self.config.data.datasets.obj_type,
            sample_choice=self.config.data.datasets.sample_choice_train,
            ignore_label=self.config.data.ignore_label,
            volume_augmentations_path=self.config.data.datasets.volume_augmentations_path,
            mode="train",
            instance_population=self.config.data.datasets.instance_population,
            center_coordinates=self.config.data.datasets.center_coordinates,
        )
        if self.config.general.dataset == "semantickitti":
            self.validation_dataset = LidarDataset(
                data_dir=self.config.data.datasets.data_dir,
                add_distance=self.config.data.datasets.add_distance,
                sweep=self.config.data.datasets.sweep,
                segment_full_scene=self.config.data.datasets.segment_full_scene,
                obj_type=self.config.data.datasets.obj_type,
                sample_choice=self.config.data.datasets.sample_choice_validation,
                ignore_label=self.config.data.ignore_label,
                mode="validation",
                instance_population=0,  # self.config.data.datasets.instance_population
                volume_augmentations_path=None,
                center_coordinates=self.config.data.datasets.center_coordinates,
            )
        elif "nuScenes" in self.config.general.dataset:
            self.validation_dataset = LidarDatasetNuscenes(
                data_dir=self.config.data.datasets.data_dir,
                add_distance=self.config.data.datasets.add_distance,
                sweep=self.config.data.datasets.sweep,
                segment_full_scene=self.config.data.datasets.segment_full_scene,
                obj_type=self.config.data.datasets.obj_type,
                sample_choice=self.config.data.datasets.sample_choice_validation,
                ignore_label=self.config.data.ignore_label,
                mode="validation",
                instance_population=0,  # self.config.data.datasets.instance_population
                volume_augmentations_path=None,
                center_coordinates=self.config.data.datasets.center_coordinates,
                dataset_type=self.config.general.dataset,
            )
        # self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)

    def train_dataloader(self):
        effective_batch_size = self.config.data.dataloader.batch_size * self.config.trainer.num_devices * self.config.trainer.num_nodes
        print(f"num devices (self.trainer.num_devices): {self.trainer.num_devices}")
        print(f"train_dataloader - batch_size: {self.config.data.dataloader.batch_size}, effective_batch_size: {effective_batch_size}, num_workers: {self.config.data.dataloader.num_workers}")
        c_fn = VoxelizeCollate(mode="train", ignore_label=self.config.data.ignore_label, voxel_size=self.config.data.dataloader.voxel_size)
        return DataLoader(self.train_dataset, shuffle=True, pin_memory=self.config.data.dataloader.pin_memory, num_workers=self.config.data.dataloader.num_workers, batch_size=self.config.data.dataloader.batch_size, collate_fn=c_fn)

    def val_dataloader(self):
        effective_batch_size = self.config.data.dataloader.test_batch_size * self.config.trainer.num_devices * self.config.trainer.num_nodes
        print(f"val_dataloader - batch_size: {self.config.data.dataloader.test_batch_size}, effective_batch_size: {effective_batch_size}, num_workers: {self.config.data.dataloader.num_workers}")
        c_fn = VoxelizeCollate(mode="validation", ignore_label=self.config.data.ignore_label, voxel_size=self.config.data.dataloader.voxel_size)
        return DataLoader(self.validation_dataset, shuffle=False, pin_memory=self.config.data.dataloader.pin_memory, num_workers=self.config.data.dataloader.num_workers, batch_size=self.config.data.dataloader.test_batch_size, collate_fn=c_fn)

    def test_dataloader(self):
        return None

    def pre_interactive_sampling(
        self,
        pcd_features,
        aux,
        coordinates,
        raw_coords,
        batch_indicators,
        pos_encodings_pcd,
        labels,
        click_idx,
        click_time_idx,
    ):
        batch_size = batch_indicators.max() + 1
        current_num_iter = 0
        cluster_dict = {}
        num_forward_iters = random.randint(0, self.config.general.max_num_clicks - 1)
        # self.log(
        #     "trainer/forward_iterations", num_forward_iters, prog_bar=True, on_epoch=False, on_step=True, batch_size=batch_size
        # )

        with torch.no_grad():
            self.interactive4d.eval()
            eval_model = self.interactive4d
            while current_num_iter <= num_forward_iters:
                if current_num_iter == 0:
                    pred = [torch.zeros(l.shape).to(raw_coords) for l in labels]
                else:
                    outputs = eval_model.forward_mask(
                        pcd_features,
                        aux,
                        coordinates,
                        pos_encodings_pcd,
                        click_idx=click_idx,
                        click_time_idx=click_time_idx,
                    )
                    pred_logits = outputs["pred_masks"]
                    pred = [p.argmax(-1) for p in pred_logits]

                for idx in range(batch_size):
                    sample_mask = batch_indicators == idx
                    sample_pred = pred[idx]

                    if current_num_iter != 0:
                        # update prediction with sparse gt
                        for obj_id, cids in click_idx[idx].items():
                            sample_pred[cids] = int(obj_id)

                    sample_labels = labels[idx]
                    sample_raw_coords = raw_coords[sample_mask]

                    objects_info = get_objects_iou(pred, labels)

                    new_clicks, new_clicks_num, new_click_pos, new_click_time, cluster_dict = get_iou_based_simulated_clicks(
                        sample_pred,
                        sample_labels,
                        sample_raw_coords,
                        current_num_iter,
                        objects_info=objects_info[idx],
                        current_click_idx=click_idx[idx],
                        training=True,
                        cluster_dict=cluster_dict,
                    )

                    ### add new clicks ###
                    if new_clicks is not None:
                        click_idx[idx], click_time_idx[idx] = extend_clicks(click_idx[idx], click_time_idx[idx], new_clicks, new_click_time)

                current_num_iter += 1

            return click_idx, click_time_idx

    def verify_labels_post_quantization(self, labels, bboxs_target, click_idx, obj2label, batch_size):
        # Remove objects which are not in the scene (due to quantization) and update the labels accordingly
        obj_to_remove = []
        for i in range(batch_size):
            unique_labels_after_qunatization = labels[i].unique()  # Assuming labels is a PyTorch tensor
            unique_labels_after_qunatization = {int(label) for label in unique_labels_after_qunatization}
            obj_to_remove.extend((i, key) for key in click_idx[i] if int(key) not in unique_labels_after_qunatization)

        if obj_to_remove:
            # print("Removing objects from the scene due to quantization")
            for i, key in obj_to_remove:
                if key != "0":
                    del click_idx[i][key]
                    del obj2label[i][key]

            for i in range(batch_size):
                mapping = {old_key: new_key for new_key, old_key in enumerate(sorted(click_idx[i].keys(), key=int))}
                click_idx[i] = {str(j): click_idx[i][old_key] for j, old_key in enumerate(sorted(click_idx[i].keys(), key=int))}
                obj2label[i] = {str(j): obj2label[i][old_key] for j, old_key in enumerate(sorted(obj2label[i].keys(), key=int), start=1)}
                # Update the keys in labels[i] using the mapping dictionary
                for old_id, new_id in mapping.items():
                    labels[i][labels[i] == int(old_id)] = new_id

                # Update the corresponding bboxs_target
                sorted_keys = sorted(bboxs_target[i].keys())
                key_mapping = {old_key: new_key for new_key, old_key in enumerate(sorted_keys, start=1)}
                bboxs_target[i] = {key_mapping[old_key]: value for old_key, value in bboxs_target[i].items()}

        return click_idx, obj2label, labels, bboxs_target


class CustomAdamW(AdamW):
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # Ensure state steps are on the CPU
                if "step" in state and state["step"].is_cuda:
                    state["step"] = state["step"].cpu()
        super().step(closure)
