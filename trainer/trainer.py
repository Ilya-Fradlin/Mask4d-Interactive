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

import utils.misc as utils
from utils.utils import generate_wandb_objects3d
from utils.seg import mean_iou, mean_iou_scene, cal_click_loss_weights, extend_clicks, get_simulated_clicks
from models.metrics.utils import IoU_at_numClicks, NumClicks_for_IoU, mIoU_per_class_metric, mIoU_metric, losses_metric
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

        # Initiatie the monitoring metric
        self.log("mIoU_monitor", 0, sync_dist=True, logger=False)
        # Training metrics
        self.losses_metric = losses_metric()
        self.mIoU_metric = mIoU_metric()
        self.mIoU_per_class_metric = mIoU_per_class_metric(training=True)
        # Validation metrics
        self.losses_metric_validation = losses_metric()
        self.mIoU_metric_validation = mIoU_metric()
        self.mIoU_per_class_metric_validation = mIoU_per_class_metric(training=False)
        self.iou_at_numClicks = IoU_at_numClicks(num_clicks=self.config.general.clicks_of_interest)
        self.numClicks_for_IoU = NumClicks_for_IoU(iou_thresholds=self.config.general.iou_targets)


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

        click_idx, obj2label, labels = self.verify_labels_post_quantization(labels, click_idx, obj2label, batch_size)

        # Check if there is more than just the background in the scene
        for idx in range(batch_size):
            if len(labels[idx].unique()) < 2:
                # If there is only the background in the scene, skip the scene
                print("after quntization, only background in the scene")
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
            general_miou, label_miou_dict = mean_iou(pred, labels, obj2label)
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
        feats = data["features"]
        labels = target["labels"]
        labels_full = [torch.from_numpy(l).to(coords) for l in target["labels_full"]]
        click_idx = data["click_idx"]
        inverse_maps = target["inverse_maps"]
        # TODO: handle clicks in multiple sweep better (for now just picks the first clicks from the first scene)
        obj2label = [mapping[0] for mapping in target["obj2labels"]]
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
                print("after quntization, only background in the scene")
                return

        ###### interactive evaluation ######
        data = ME.SparseTensor(coordinates=coords, features=feats, device=self.device)
        pcd_features, aux, coordinates, pos_encodings_pcd = self.interactive4d.forward_backbone(data, raw_coordinates=raw_coords)

        # TODO: check how is the max_num_clicks update more than 1 batch size
        iou_targets = self.config.general.iou_targets
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

            if current_num_clicks != 0:
                click_weights = cal_click_loss_weights(batch_indicators, raw_coords, torch.cat(labels), click_idx)
                loss_dict = self.criterion(outputs, labels, click_weights)
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

                new_clicks, new_clicks_num, new_click_pos, new_click_time = get_simulated_clicks(
                    sample_pred,
                    sample_labels,
                    sample_raw_coords,
                    current_num_clicks,
                    current_click_idx=click_idx[idx],
                    training=False,
                )

                ### add new clicks ###
                if new_clicks is not None:
                    click_idx[idx], click_time_idx[idx] = extend_clicks(
                        click_idx[idx], click_time_idx[idx], new_clicks, new_click_time
                    )

            if current_num_clicks != 0:
                # mean_iou here is calculated for just with the points responsible for the quantized values!
                general_miou, label_miou_dict = mean_iou(updated_pred, labels, obj2label)
                
                self.losses_metric_validation.update(sum(loss_dict_reduced_scaled.values()))
                self.mIoU_metric_validation.update(general_miou)
                self.mIoU_per_class_metric_validation.update(label_miou_dict)

            if current_num_clicks == 0:
                new_clicks_num = num_obj[idx]
            else:
                new_clicks_num = 1
            current_num_clicks += new_clicks_num

        pred = 0

        if (
            (self.config.general.experiment_name != "debugging")
            and (self.trainer.is_global_zero)
            and (batch_idx % self.config.general.visualization_frequency == 0)
        ):  # Condition for visualization logging
            # choose a random scene to visualize from the batch
            chosen_scene_idx = random.randint(0, batch_size - 1)
            scene_name = scene_names[chosen_scene_idx]
            sample_mask = batch_indicators == chosen_scene_idx
            sample_raw_coords = raw_coords[sample_mask]
            sample_pred = updated_pred[chosen_scene_idx]
            sample_labels = labels[chosen_scene_idx]
            sample_click_idx = click_idx[chosen_scene_idx]

            gt_scene, pred_scene = generate_wandb_objects3d(sample_raw_coords, sample_labels, sample_pred, sample_click_idx)
            wandb.log({f"point_scene/ground_truth_{scene_name}_{batch_idx}": gt_scene})
            wandb.log({f"point_scene/prediction_{scene_name}_{batch_idx}": pred_scene})
            # self.config.general.visualization_dir,
        # self.validation_step_outputs.append(pred)

    def on_validation_epoch_end(self):
            print("\n")
            print("--------- Evaluating Validation Performance  -----------")
            warnings.filterwarnings(
                "ignore",
                message="The ``compute`` method of metric NumClicks_for_IoU was called before the ``update`` method",
                category=UserWarning,
            )
            results_dict = {}

            # Evaluate the NoC@IoU Metric
            metrics_dictionary, iou_thresholds = self.numClicks_for_IoU.compute()
            for iou in iou_thresholds:
                noc = metrics_dictionary[iou]["noc"]
                count = metrics_dictionary[iou]["count"]
                results_dict[f"scenes_reached_{iou}_iou"] = count.item()
                if count == 0:
                    results_dict[f"validation/Interactive_metrics/NoC@{iou}"] = 0  # or return a default value or raise an error
                else:
                    results_dict[f"validation/Interactive_metrics/NoC@{iou}"] = (noc / count).item()

            # Evaluate the IoU@NoC Metric
            metrics_dictionary, evaluated_num_clicks = self.iou_at_numClicks.compute()
            for noc in evaluated_num_clicks:
                iou = metrics_dictionary[noc]["iou"]
                count = metrics_dictionary[noc]["count"]
                if count == 0:
                    results_dict[f"validation/Interactive_metrics/IoU@{noc}"] = 0  # or return a default value or raise an error
                else:
                    results_dict[f"validation/Interactive_metrics/IoU@{noc}"] = (iou / count).item()

            miou_epoch = self.mIoU_metric_validation.compute()
            miou_per_class_epoch = self.mIoU_per_class_metric_validation.compute()
            losses_epoch = self.losses_metric_validation.compute()
        
            print("\nValidation Epoch Results:\n")
            print(f"miou_epoch: {miou_epoch}, losses_epoch: {losses_epoch} \n")
            print(f"miou_per_class_epoch: {miou_per_class_epoch} \n")
            print("Interactive_metrics:\n")
            print(results_dict)
            
            self.mIoU_metric_validation.reset()
            self.mIoU_per_class_metric_validation.reset()
            self.losses_metric_validation.reset()
            self.iou_at_numClicks.reset()
            self.numClicks_for_IoU.reset()

    def test_step(self, batch, batch_idx):
        # TODO
        pass

    def test_epoch_end(self, outputs):
        # TODO
        return {}

    def configure_optimizers(self):
        # Adjust the learning rate based on the number of GPUs
        # self.config.optimizer.lr = self.config.optimizer.lr * math.sqrt(self.trainer.num_devices)
        # self.config.scheduler.scheduler.max_lr = self.config.optimizer.lr

        optimizer = hydra.utils.instantiate(self.config.optimizer, params=self.parameters())
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = math.ceil(len(self.train_dataloader()) / self.trainer.num_devices)
        lr_scheduler = hydra.utils.instantiate(self.config.scheduler.scheduler, optimizer=optimizer)
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        print(f"optimizer_max_lr: {self.config.optimizer.lr}, steps_per_epoch:{self.config.scheduler.scheduler.steps_per_epoch}")
        return [optimizer], [scheduler_config]

    def setup(self, stage):
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        self.validation_dataset = hydra.utils.instantiate(self.config.data.validation_dataset)
        # self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)

    def train_dataloader(self):
        print(f"num devices: {self.trainer.num_devices}")
        print(
            f"train_dataloader - batch_size: {self.config.data.train_dataloader.batch_size}, effective_batch_size: {self.config.data.train_dataloader.batch_size * self.trainer.num_devices}, num_workers: {self.config.data.train_dataloader.num_workers}"
        )
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        print(
            f"val_dataloader - batch_size: {self.config.data.train_dataloader.batch_size}, effective_batch_size: {self.config.data.train_dataloader.batch_size * self.trainer.num_devices}, num_workers: {self.config.data.train_dataloader.num_workers}"
        )
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

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

                    new_clicks, new_clicks_num, new_click_pos, new_click_time = get_simulated_clicks(
                        sample_pred,
                        sample_labels,
                        sample_raw_coords,
                        current_num_iter,
                        current_click_idx=click_idx[idx],
                        training=True,
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
