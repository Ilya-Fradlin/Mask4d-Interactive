import statistics
import math
import hydra
import copy
import random
import warnings
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.cluster import DBSCAN
from contextlib import nullcontext
from collections import defaultdict

import utils.misc as utils
from utils.utils import associate_instances, save_predictions
from utils.seg import mean_iou, mean_iou_scene, cal_click_loss_weights, extend_clicks, get_simulated_clicks
from models.metrics.utils import Evaluator, IoU_at_numClicks, NumClicks_for_IoU
from pytorch_lightning.callbacks import ModelCheckpoint


class ObjectSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # model
        self.interactive4d = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext

        weight_dict = {
            # self.config.model.num_queries
            "loss_bce": self.config.loss.bce_loss_coef,
            "loss_dice": self.config.loss.dice_loss_coef,
        }

        # TODO: check this aux loss
        if config.loss.aux:
            aux_weight_dict = {}
            for i in range(self.interactive4d.num_decoders * self.interactive4d.num_levels):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.criterion = hydra.utils.instantiate(self.config.loss.criterion, weight_dict=weight_dict)

        # metrics
        self.class_evaluator = hydra.utils.instantiate(config.metric)
        self.last_seq = None

        # self.validation_step_outputs = []
        # self.training_step_outputs = []
        # TODO handle better the metric logger
        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.iou_at_numClicks = IoU_at_numClicks()
        self.numClicks_for_IoU = NumClicks_for_IoU()

        self.save_hyperparameters()

    def forward(self, x, raw_coordinates=None, feats=None, click_idx=None, is_eval=False):
        with self.optional_freeze():
            x = self.interactive4d(x, raw_coordinates=raw_coordinates, feats=feats, is_eval=is_eval)
        return x

    def training_step(self, batch, batch_idx):

        data, target = batch

        coords = data.coordinates
        raw_coords = data.raw_coordinates.to(self.device)
        feats = data.features
        labels = [l.to(self.device) for l in target["labels"]]
        click_idx = data.click_idx
        obj2label = [mapping[0] for mapping in data.obj2label]
        batch_indicators = coords[:, 0]
        batch_size = batch_indicators.max() + 1

        click_idx, obj2label, labels = self.verify_labels_post_quantization(labels, click_idx, obj2label, batch_size)

        # Check if there is more than just the background in the scene
        for idx in range(batch_size):
            if len(labels[idx].unique()) < 2:
                # If there is only the background in the scene, skip the scene
                return None

        data = ME.SparseTensor(coordinates=data.coordinates, features=feats, device=self.device)
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
        outputs = self.interactive4d.forward_mask(
            pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=click_idx, click_time_idx=click_time_idx
        )

        ######### 3. loss back propagation #########
        click_weights = cal_click_loss_weights(batch_indicators, raw_coords, torch.cat(labels), click_idx)
        loss_dict = self.criterion(outputs, labels, click_weights)
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
            general_miou = mean_iou(pred, labels, obj2label)
            self.metric_logger.update(mIoU=general_miou)
            self.metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        # self.training_step_outputs.extend(losses)

        self.log("train/loss", losses, prog_bar=True)
        self.log("train/mIoU", general_miou, prog_bar=True)

        return losses

    def on_training_epoch_end(self):
        # TODO
        train_stats = {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
        self.log_dict({"lr_rate": train_stats["lr"]})
        self.log_dict(
            {
                "train/epoch": self.current_epoch,
                "train/loss_epoch": train_stats["loss"],
                "train/loss_bce_epoch": train_stats["loss_bce"],
                "train/loss_dice_epoch": train_stats["loss_dice"],
                "train/mIoU_epoch": train_stats["mIoU"],
            },
            on_step=True,
        )
        print("Averaged stats:", self.metric_logger)
        self.metric_logger = utils.MetricLogger(delimiter="  ")  # reset metric

    def validation_step(self, batch, batch_idx):

        data, target = batch

        coords = data.coordinates
        raw_coords = data.raw_coordinates.to(self.device)
        feats = data.features
        labels = [l.to(self.device) for l in target["labels"]]
        labels_full = [torch.from_numpy(l).to(self.device) for l in target["labels_full"]]
        click_idx = data.click_idx
        inverse_maps = data.inverse_maps
        # TODO: handle clicks in multiple sweep better (for now just picks the first clicks from the first scene
        obj2label = [mapping[0] for mapping in data.obj2label]
        batch_indicators = coords[:, 0]
        num_obj = [len(mapping.keys()) for mapping in obj2label]
        batch_size = batch_indicators.max() + 1
        current_num_clicks = 0

        # Remove objects which are not in the scene (due to quantization)
        click_idx, obj2label, labels = self.verify_labels_post_quantization(labels, click_idx, obj2label, batch_size)
        click_time_idx = copy.deepcopy(click_idx)

        # Check if there is more than just the background in the scene
        for idx in range(batch_size):
            if len(labels[idx].unique()) < 2:
                # If there is only the background in the scene, skip the scene
                return

        ###### interactive evaluation ######
        data = ME.SparseTensor(coordinates=coords, features=feats, device=self.device)
        pcd_features, aux, coordinates, pos_encodings_pcd = self.interactive4d.forward_backbone(data, raw_coordinates=raw_coords)

        # TODO: check how is the max_num_clicks update more than 1 batch size
        max_num_clicks = num_obj[0] * self.config.general.max_num_clicks
        while current_num_clicks <= max_num_clicks:
            if current_num_clicks == 0:
                pred = [torch.zeros(l.shape).to(self.device) for l in labels]
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

            if current_num_clicks != 0:
                click_weights = cal_click_loss_weights(batch_indicators, raw_coords, torch.cat(labels), click_idx)
                loss_dict = self.criterion(outputs, labels, click_weights)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
                loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}

            updated_pred = []

            next_iou_target_index = 0
            iou_targets = [0.5, 0.65, 0.80, 0.85, 0.90, 9999]
            for idx in range(batch_indicators.max() + 1):
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
                sample_iou, _ = mean_iou_scene(sample_pred_full, sample_labels_full)

                # Logging IoU@1, IoU@3, IoU@5, IoU@10, IoU@15
                average_clicks_per_obj = current_num_clicks / num_obj[idx]
                if average_clicks_per_obj in [1, 3, 5, 10, 15]:
                    self.iou_at_numClicks.update(iou=sample_iou.item(), number_of_clicks=average_clicks_per_obj)

                # Logging NoC@50, NoC@65, NoC@80, NoC@85, NoC@90
                if iou_targets[next_iou_target_index] < sample_iou:
                    while iou_targets[next_iou_target_index] < sample_iou:
                        self.numClicks_for_IoU.update(iou=iou_targets[next_iou_target_index], noc=average_clicks_per_obj)
                        next_iou_target_index += 1
                        if next_iou_target_index == len(iou_targets) - 1:  # 9999 (impossible value) is the last element
                            break

                new_clicks, new_clicks_num, new_click_pos, new_click_time = get_simulated_clicks(
                    sample_pred, sample_labels, sample_raw_coords, current_num_clicks, training=False
                )

                ### add new clicks ###
                if new_clicks is not None:
                    click_idx[idx], click_time_idx[idx] = extend_clicks(
                        click_idx[idx], click_time_idx[idx], new_clicks, new_click_time
                    )

            if current_num_clicks != 0:
                general_miou = mean_iou(updated_pred, labels, obj2label)
                self.metric_logger.update(mIoU=general_miou)
                self.metric_logger.update(
                    loss=sum(loss_dict_reduced_scaled.values()),
                    **loss_dict_reduced_scaled,
                    **loss_dict_reduced_unscaled,
                )

            if current_num_clicks == 0:
                new_clicks_num = num_obj[idx]
            else:
                new_clicks_num = 1
            current_num_clicks += new_clicks_num

        pred = 0
        # self.validation_step_outputs.append(pred)

    def on_validation_epoch_end(self):
        # TODO
        print("--------- Evaluating Validation Performance  -----------")
        # all_metrics = torch.stack(self.validation_step_outputs)
        # evaluator = Evaluator(all_metrics)
        # results_dict = evaluator.eval_results()
        # print("****************************")
        warnings.filterwarnings(
            "ignore",
            message="The ``compute`` method of metric NumClicks_for_IoU was called before the ``update`` method",
            category=UserWarning,
        )
        results_dict = {}
        results_dict["mIoU"] = self.metric_logger.meters["mIoU"].global_avg
        metrics_dictionary, iou_thresholds = self.numClicks_for_IoU.compute()
        for iou in iou_thresholds:
            noc = metrics_dictionary[iou]["noc"]
            count = metrics_dictionary[iou]["count"]
            results_dict[f"scenes_reached_{iou}_iou"] = count.item()
            if count == 0:
                results_dict[f"NoC@{iou}"] = 0  # or return a default value or raise an error
            else:
                results_dict[f"NoC@{iou}"] = (noc / count).item()

        iou_sums, counts = self.iou_at_numClicks.compute()
        results_dict["IoU@1"] = (iou_sums[0] / counts[0]).item()
        results_dict["IoU@3"] = (iou_sums[1] / counts[1]).item()
        results_dict["IoU@5"] = (iou_sums[2] / counts[2]).item()
        results_dict["IoU@10"] = (iou_sums[3] / counts[3]).item()
        # results_dict["IoU@15"] = (iou_sums[4] / counts[4]).item()
        print(results_dict)
        stats = {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
        stats.update(results_dict)

        # self.validation_step_outputs.clear()  # free memory
        self.metric_logger = utils.MetricLogger(delimiter="  ")  # reset metric

        self.log_dict(
            {
                "val/epoch": self.current_epoch,
                "val/loss_epoch": stats["loss"],
                "val/loss_bce_epoch": stats["loss_bce"],
                "val/loss_dice_epoch": stats["loss_dice"],
                "val/mIoU_epoch": stats["mIoU"],
                "val_metrics/NoC_50": stats["NoC@50"],
                "val_metrics/NoC_50": stats["scenes_reached_50_iou"],
                "val_metrics/NoC_65": stats["NoC@65"],
                "val_metrics/NoC_65": stats["scenes_reached_65_iou"],
                "val_metrics/NoC_80": stats["NoC@80"],
                "val_metrics/NoC_80": stats["scenes_reached_80_iou"],
                "val_metrics/NoC_85": stats["NoC@85"],
                "val_metrics/NoC_85": stats["scenes_reached_85_iou"],
                "val_metrics/NoC_90": stats["NoC@90"],
                "val_metrics/NoC_90": stats["scenes_reached_90_iou"],
                "val_metrics/IoU_1": stats["IoU@1"],
                "val_metrics/IoU_3": stats["IoU@3"],
                "val_metrics/IoU_5": stats["IoU@5"],
                "val_metrics/IoU_10": stats["IoU@10"],
                # "val_metrics/IoU_15": stats["IoU@15"],
            }
        )
        self.log("mIoU", stats["mIoU"])

    def test_step(self, batch, batch_idx):
        # TODO
        pass

    def test_epoch_end(self, outputs):
        return {}

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.config.optimizer, params=self.parameters())
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(self.train_dataloader())
        lr_scheduler = hydra.utils.instantiate(self.config.scheduler.scheduler, optimizer=optimizer)
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def setup(self, stage):
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        self.validation_dataset = hydra.utils.instantiate(self.config.data.validation_dataset)
        # self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )

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
        current_num_iter = 0
        num_forward_iters = random.randint(0, 19)

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

                for idx in range(batch_indicators.max() + 1):
                    sample_mask = batch_indicators == idx
                    sample_pred = pred[idx]

                    if current_num_iter != 0:
                        # update prediction with sparse gt
                        for obj_id, cids in click_idx[idx].items():
                            sample_pred[cids] = int(obj_id)

                    sample_labels = labels[idx]
                    sample_raw_coords = raw_coords[sample_mask]

                    new_clicks, new_clicks_num, new_click_pos, new_click_time = get_simulated_clicks(
                        sample_pred, sample_labels, sample_raw_coords, current_num_iter, training=True
                    )

                    ### add new clicks ###
                    if new_clicks is not None:
                        click_idx[idx], click_time_idx[idx] = extend_clicks(
                            click_idx[idx], click_time_idx[idx], new_clicks, new_click_time
                        )

                current_num_iter += 1

            return click_idx, click_time_idx

    def verify_labels_post_quantization(self, labels, click_idx, obj2label, batch_size):
        # Remove objects which are not in the scene (due to quantization) and update the labels accordingly
        obj_to_remove = []
        for i in range(batch_size):
            unique_labels_after_qunatization = labels[i].unique()  # Assuming labels is a PyTorch tensor
            unique_labels_after_qunatization = {int(label) for label in unique_labels_after_qunatization}
            obj_to_remove.extend((i, key) for key in click_idx[i] if int(key) not in unique_labels_after_qunatization)

        if obj_to_remove:
            print("Removing objects from the scene due to quantization")
            for i, key in obj_to_remove:
                del click_idx[i][key]
                del obj2label[i][key]

            for i in range(batch_size):
                mapping = {old_key: new_key for new_key, old_key in enumerate(sorted(click_idx[i].keys(), key=int))}
                click_idx[i] = {str(j): click_idx[i][old_key] for j, old_key in enumerate(sorted(click_idx[i].keys(), key=int))}
                obj2label[i] = {
                    str(j): obj2label[i][old_key] for j, old_key in enumerate(sorted(obj2label[i].keys(), key=int), start=1)
                }
                # Update the keys in labels[i] using the mapping dictionary
                for old_id, new_id in mapping.items():
                    labels[i][labels[i] == int(old_id)] = new_id

        return click_idx, obj2label, labels
