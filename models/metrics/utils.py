import json
import torch
from torchmetrics import Metric


class IoU_at_numClicks(Metric):
    higher_is_better = True

    def __init__(self, num_clicks=[1, 3, 5, 10, 15]):
        super().__init__()
        self.num_clicks = num_clicks

        for noc in num_clicks:
            self.add_state(f"iou_at_{noc}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"count_for_{noc}", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, iou, noc):
        if noc in self.num_clicks:
            noc = int(noc)
            self.__dict__[f"iou_at_{noc}"] += iou
            self.__dict__[f"count_for_{noc}"] += 1
        else:
            print("iou not found")
            raise ValueError

    def compute(self):
        metrics_dictionary = {}

        for noc in self.num_clicks:
            metrics_dictionary[noc] = {}
            metrics_dictionary[noc]["iou"] = self.__dict__[f"iou_at_{noc}"]
            metrics_dictionary[noc]["count"] = self.__dict__[f"count_for_{noc}"]

        return metrics_dictionary, self.num_clicks


class NumClicks_for_IoU(Metric):

    def __init__(self, iou_thresholds=[50, 65, 80, 85, 90]):
        super().__init__()

        self.iou_thresholds = iou_thresholds
        for iou in iou_thresholds:
            self.add_state(f"noc_for_{iou}", default=torch.tensor(0.0), dist_reduce_fx="mean")
            self.add_state(f"count_for_{iou}", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, iou, noc):
        iou = int(100 * iou)
        if iou in self.iou_thresholds:
            self.__dict__[f"noc_for_{iou}"] += noc
            self.__dict__[f"count_for_{iou}"] += 1
        else:
            print("iou not found")
            raise ValueError

    def compute(self):
        metrics_dictionary = {}

        for iou in self.iou_thresholds:
            metrics_dictionary[iou] = {}
            metrics_dictionary[iou]["noc"] = self.__dict__[f"noc_for_{iou}"]
            metrics_dictionary[iou]["count"] = self.__dict__[f"count_for_{iou}"]

        return metrics_dictionary, self.iou_thresholds


class mIou_per_class(Metric):
    def __init__(self, num_classes=19):
        super().__init__()
        self.num_classes = num_classes
        for i in range(1, num_classes + 1):
            self.add_state(f"iou_class_{i}", default=torch.tensor(0.0), dist_reduce_fx="mean")
            self.add_state(f"counts_class_{i}", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, iou, class_id):
        self.__dict__[f"iou_class_{class_id}"] += iou
        self.__dict__[f"counts_class_{class_id}"] += 1

    def compute(self):
        metrics_dictionary = {}

        for i in range(self.num_classes):
            metrics_dictionary[i] = {}
            metrics_dictionary[i]["iou"] = self.__dict__[f"iou_for_{i}"]
            metrics_dictionary[i]["count"] = self.__dict__[f"count_for_{i}"]

        return metrics_dictionary, self.num_classes


# class NumClicks_for_IoU(Metric):

#     def __init__(self, iou_thresholds=[50, 65, 80, 85, 90]):
#         super().__init__()

#         self.iou_thresholds = iou_thresholds
#         self.add_state("noc_for_50", default=torch.tensor(0), dist_reduce_fx="mean")
#         self.add_state("count_for_50", default=torch.tensor(0), dist_reduce_fx="mean")
#         self.add_state("noc_for_65", default=torch.tensor(0), dist_reduce_fx="mean")
#         self.add_state("count_for_65", default=torch.tensor(0), dist_reduce_fx="mean")
#         self.add_state("noc_for_80", default=torch.tensor(0), dist_reduce_fx="mean")
#         self.add_state("count_for_80", default=torch.tensor(0), dist_reduce_fx="mean")
#         self.add_state("noc_for_85", default=torch.tensor(0), dist_reduce_fx="mean")
#         self.add_state("count_for_85", default=torch.tensor(0), dist_reduce_fx="mean")
#         self.add_state("noc_for_90", default=torch.tensor(0), dist_reduce_fx="mean")
#         self.add_state("count_for_90", default=torch.tensor(0), dist_reduce_fx="mean")

#     def update(self, iou, noc):
#         if iou == 0.5:
#             self.noc_for_50 += noc
#             self.count_for_50 += 1
#         elif iou == 0.65:
#             self.noc_for_65 += noc
#             self.count_for_50 += 1
#         elif iou == 0.80:
#             self.noc_for_80 += noc
#             self.count_for_50 += 1
#         elif iou == 0.85:
#             self.noc_for_85 += noc
#             self.count_for_50 += 1
#         elif iou == 0.90:
#             self.noc_for_90 += noc
#             self.count_for_90 += 1
#         else:
#             print("iou not found")
#             raise ValueError

#     def compute(self):
#         metrics_dictionary = {}

#         metrics_dictionary[50] = {}
#         metrics_dictionary[50]["noc"] = self.noc_for_50
#         metrics_dictionary[50]["count"] = self.count_for_50
#         metrics_dictionary[65] = {}
#         metrics_dictionary[65]["noc"] = self.noc_for_65
#         metrics_dictionary[65]["count"] = self.count_for_65
#         metrics_dictionary[80] = {}
#         metrics_dictionary[80]["noc"] = self.noc_for_80
#         metrics_dictionary[80]["count"] = self.count_for_80
#         metrics_dictionary[85] = {}
#         metrics_dictionary[85]["noc"] = self.noc_for_85
#         metrics_dictionary[85]["count"] = self.count_for_85
#         metrics_dictionary[90] = {}
#         metrics_dictionary[90]["noc"] = self.noc_for_90
#         metrics_dictionary[90]["count"] = self.count_for_90

#         return metrics_dictionary, self.iou_thresholds
