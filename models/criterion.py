import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from torch import Tensor


def box_loss(inputs: torch.Tensor, targets: torch.Tensor, num_bboxs: float):
    loss = F.l1_loss(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_bboxs


box_loss_jit = torch.jit.script(box_loss)


class SetCriterion(nn.Module):

    def __init__(self, losses, weight_dict):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def multiclass_dice_loss(
        self,
        input: Tensor,
        target: Tensor,
        eps: float = 1e-6,
        check_target_validity: bool = True,
        ignore_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Computes DICE loss for multi-class predictions. API inputs are identical to torch.nn.functional.cross_entropy()
        :param input: tensor of shape [N, C, *] with unscaled logits
        :param target: tensor of shape [N, *]
        :param eps:
        :param check_target_validity: checks if the values in the target are valid
        :param ignore_mask: optional tensor of shape [N, *]
        :return: tensor
        """
        assert input.ndim >= 2
        input = input.softmax(1)
        num_classes = input.size(1)

        if check_target_validity:
            class_ids = target.unique()
            assert not torch.any(torch.logical_or(class_ids < 0, class_ids >= num_classes)), f"Number of classes = {num_classes}, but target has the following class IDs: {class_ids.tolist()}"

        target = torch.stack([target == cls_id for cls_id in range(0, num_classes)], 1).to(dtype=input.dtype)  # [N, C, *]

        if ignore_mask is not None:
            ignore_mask = ignore_mask.unsqueeze(1)
            expand_dims = [-1, input.size(1)] + ([-1] * (ignore_mask.ndim - 2))
            ignore_mask = ignore_mask.expand(*expand_dims)

        return self.dice_loss(input, target, eps=eps, ignore_mask=ignore_mask)

    def dice_loss(self, input: Tensor, target: Tensor, ignore_mask: Optional[Tensor] = None, eps: Optional[float] = 1e-6):
        """
        Computes the DICE or soft IoU loss.
        :param input: tensor of shape [N, *]
        :param target: tensor with shape identical to input
        :param ignore_mask: tensor of same shape as input. non-zero values in this mask will be
        :param eps
        excluded from the loss calculation.
        :return: tensor
        """
        assert input.shape == target.shape, f"Shape mismatch between input ({input.shape}) and target ({target.shape})"
        assert input.dtype == target.dtype

        if torch.is_tensor(ignore_mask):
            assert ignore_mask.dtype == torch.bool
            assert input.shape == ignore_mask.shape, f"Shape mismatch between input ({input.shape}) and " f"ignore mask ({ignore_mask.shape})"
            input = torch.where(ignore_mask, torch.zeros_like(input), input)
            target = torch.where(ignore_mask, torch.zeros_like(target), target)

        input = input.flatten(1)
        target = target.detach().flatten(1)

        numerator = 2.0 * (input * target).mean(1)
        denominator = (input + target).mean(1)

        soft_iou = (numerator + eps) / (denominator + eps)

        return torch.where(numerator > eps, 1.0 - soft_iou, soft_iou * 0.0)

    def loss_bce(self, outputs, targets, bboxs, obj2label, weights=None):

        pred_masks = outputs["pred_masks"]

        loss = 0.0

        for i in range(len(pred_masks)):
            loss_sample = (F.cross_entropy(pred_masks[i], targets[i].long(), reduction="none") * weights[i]).mean()
            loss += loss_sample

        loss = loss / len(pred_masks)

        return {"loss_bce": loss}

    def loss_dice(self, outputs, targets, bboxs, obj2label, weights=None):

        pred_masks = outputs["pred_masks"]
        loss = 0.0
        for i in range(len(pred_masks)):
            loss_sample = (self.multiclass_dice_loss(pred_masks[i], targets[i].long()) * weights[i]).mean()
            loss += loss_sample

        loss = loss / len(pred_masks)
        return {"loss_dice": loss}

    def loss_bbox(self, outputs, targets, target_bboxs, obj2label, weights):
        loss_bbox = torch.tensor(0.0, device=outputs["pred_masks"][0].device)
        for b, pred_bboxs in enumerate(outputs["bboxs"]):  # for each scene in the batch
            predictions, ground_truth = [], []
            for obj_num, original_label in obj2label[b].items():
                if (not self.is_thing(original_label)) or (torch.count_nonzero(pred_bboxs[int(obj_num)]) == 0):
                    continue
                else:
                    predictions.append(pred_bboxs[int(obj_num)])
                    ground_truth.append(target_bboxs[b][int(obj_num)])

            if len(predictions) != 0:
                # target_bboxs = target_bboxs[keep_things]
                # pred_bboxs = pred_bboxs[keep_things]
                predictions = torch.vstack(predictions)
                ground_truth = torch.vstack(ground_truth)
                num_bboxs = predictions.shape[0]
                loss_bbox += box_loss_jit(predictions, ground_truth, num_bboxs)
        return {
            "loss_bbox": loss_bbox,
        }

    def get_loss(self, loss, outputs, targets, bboxs, obj2label, weights=None):
        loss_map = {"bce": self.loss_bce, "dice": self.loss_dice, "bbox": self.loss_bbox}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, bboxs, obj2label, weights)

    def forward(self, outputs, targets, bboxs, obj2label, weights=None):

        # outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"}

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, bboxs, obj2label, weights))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, bboxs, obj2label, weights)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def is_thing(self, label):
        semantic_label = label & 0xFFFF
        things_targets = [1, 2, 3, 4, 5, 6, 7, 8]  # [1:car,  2:bicycle,  3:motorcycle,  4:truck,  5:other-vehicle,  6:person,  7:bicyclist,  8:motorcyclist ]
        if semantic_label in things_targets:
            return True
        else:
            return False
