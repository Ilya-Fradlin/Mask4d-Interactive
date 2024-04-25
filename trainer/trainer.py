import statistics
import math
import hydra
import copy
import random
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.cluster import DBSCAN
from contextlib import nullcontext
from collections import defaultdict
from utils.utils import associate_instances, save_predictions
from utils.seg import mean_iou, mean_iou_scene, cal_click_loss_weights, extend_clicks, get_simulated_clicks
import utils.misc as utils


class ObjectSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext

        weight_dict = {
            # self.config.model.num_queries
            "loss_bce": self.config.loss.bce_loss_coef,
            "loss_dice": self.config.loss.dice_loss_coef,
        }

        # TODO: check this aux loss
        if config.loss.aux:
            aux_weight_dict = {}
            for i in range(self.model.num_decoders * self.model.num_levels):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.criterion = hydra.utils.instantiate(self.config.loss.criterion, weight_dict=weight_dict)

        # metrics
        self.class_evaluator = hydra.utils.instantiate(config.metric)
        self.last_seq = None

    def forward(self, x, raw_coordinates=None, feats=None, click_idx=None, is_eval=False):
        with self.optional_freeze():
            x = self.model(x, raw_coordinates=raw_coordinates, feats=feats, is_eval=is_eval)
        return x

    def training_step(self, batch, batch_idx):
        metric_logger = utils.MetricLogger(delimiter="  ")

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

        data = ME.SparseTensor(coordinates=data.coordinates, features=feats, device=self.device)
        pcd_features, aux, coordinates, pos_encodings_pcd = self.model.forward_backbone(data, raw_coordinates=raw_coords)

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

        #########  2. real forward pass  #########
        self.model.train()
        outputs = self.model.forward_mask(
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
            metric_logger.update(mIoU=general_miou)

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        print("Averaged stats:", metric_logger)

        return losses

    def validation_step(self, batch, batch_idx):

        metric_logger = utils.MetricLogger(delimiter="  ")

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

        ###### interactive evaluation ######
        data = ME.SparseTensor(coordinates=coords, features=feats, device=self.device)
        pcd_features, aux, coordinates, pos_encodings_pcd = self.model.forward_backbone(data, raw_coordinates=raw_coords)

        # TODO: check how is the max_num_clicks update more than 1 batch size
        max_num_clicks = num_obj[0] * self.config.general.max_num_clicks
        while current_num_clicks <= max_num_clicks:
            if current_num_clicks == 0:
                pred = [torch.zeros(l.shape).to(self.device) for l in labels]
            else:
                outputs = self.model.forward_mask(
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

                line = (
                    +mean_iou_scene
                    + " "
                    + str(num_obj[idx])
                    + " "
                    + str(current_num_clicks / num_obj[idx])
                    + " "
                    + str(sample_iou.cpu().numpy())
                    + "\n"
                )

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
                metric_logger.update(mIoU=general_miou)

                metric_logger.update(
                    loss=sum(loss_dict_reduced_scaled.values()),
                    **loss_dict_reduced_scaled,
                    **loss_dict_reduced_unscaled,
                )

            if current_num_clicks == 0:
                new_clicks_num = num_obj[idx]
            else:
                new_clicks_num = 1
            current_num_clicks += new_clicks_num

        return {general_miou}

    def test_step(self, batch, batch_idx):
        # TODO
        pass

    def training_epoch_end(self, outputs):
        # TODO
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(outputs)
        results = {"train_loss_mean": train_loss}
        self.log_dict(results)

    def validation_epoch_end(self, outputs):
        # TODO
        print("In validation_epoch_end")
        print(outputs)

        all_preds = torch.stack(self.validation_step_outputs)
        # do something with all preds
        self.validation_step_outputs.clear()  # free memory

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

    def prepare_data(self):
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
            self.model.eval()
            eval_model = self.model
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
