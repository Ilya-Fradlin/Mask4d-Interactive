import json
import torch
from torchmetrics import Metric


# class IoU_at_numClicks(Metric):
#     def __init__(self, num_classes, num_clicks, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#         self.num_classes = num_classes
#         self.num_clicks = num_clicks
#         self.add_state("iou_at_1", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("iou_at_3", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("iou_at_5", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("iou_at_10", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("iou_at_15", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, iou, number_of_clicks):
#         if number_of_clicks == 1:
#             self.iou_at_1 += iou
#             self.count += 1
#         elif number_of_clicks == 3:
#             self.iou_at_3 += iou
#         elif number_of_clicks == 5:
#             self.iou_at_5 += iou
#         elif number_of_clicks == 10:
#             self.iou_at_10 += iou
#         elif number_of_clicks == 15:
#             self.iou_at_15 += iou

#     def compute(self, number_of_clicks):
#         if number_of_clicks == 1:
#             return self.iou_at_1 / self.count
#         elif number_of_clicks == 3:
#             return self.iou_at_3 / self.count
#         elif number_of_clicks == 5:
#             return self.iou_at_5 / self.count
#         elif number_of_clicks == 10:
#             return self.iou_at_10 / self.count
#         elif number_of_clicks == 15:
#             return self.iou_at_15 / self.count


# class NumClicks_for_IoU(Metric):
#     def __init__(self, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#         self.add_state("noc_for_50", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("noc_for_65", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("noc_for_80", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("noc_for_85", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("noc_for_90", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("count_50", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("count_65", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("count_80", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("count_85", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("count_90", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, iou):
#         if iou == 50:
#             self.noc_for_50 += iou
#             self.count_50 += 1
#         elif iou == 65:
#             self.noc_for_65 += iou
#             self.count_65 += 1
#         elif iou == 80:
#             self.noc_for_80 += iou
#             self.count_80 += 1
#         elif iou == 85:
#             self.noc_for_85 += iou
#             self.count_85 += 1
#         elif iou == 90:
#             self.noc_for_90 += iou
#             self.count_90 += 1

#     def compute(self, iou):
#         if iou == 50:
#             return self.noc_for_50 / self.count_50
#         elif iou == 65:
#             return self.noc_for_65 / self.count_65
#         elif iou == 80:
#             return self.noc_for_80 / self.count_80
#         elif iou == 85:
#             return self.noc_for_85 / self.count_85
#         elif iou == 90:
#             return self.noc_for_90 / self.count_90


class IoU_at_numClicks(Metric):
    def __init__(self, num_clicks=[1, 3, 5, 10, 15]):
        super().__init__()
        self.num_clicks = num_clicks
        self.iou_sum = torch.zeros(len(num_clicks))
        self.count = torch.zeros(len(num_clicks))

    def update(self, iou, number_of_clicks):
        index = self.num_clicks.index(number_of_clicks)
        self.iou_sum[index] += iou
        self.count[index] += 1

    def compute(self):
        return self.iou_sum, self.count


class NumClicks_for_IoU(Metric):
    def __init__(self, iou_thresholds=[50, 65, 80, 85, 90]):
        super().__init__()
        self.iou_thresholds = iou_thresholds
        self.metrics_dictionary = {iou: {"noc": torch.tensor(0.0), "count": torch.tensor(0)} for iou in iou_thresholds}

    def update(self, iou, noc):
        if iou in self.iou_thresholds:
            self.metrics_dictionary[iou]["noc"] += noc
            self.metrics_dictionary[iou]["count"] += 1

    def compute(self):
        return self.metrics_dictionary, self.iou_thresholds


class Evaluator:
    def __init__(self, scene_list_file, result_file, MAX_IOU):

        self.MAX_IOU = MAX_IOU  # [0.5, 0.65, 0.8, 0.85, 0.9]

        with open(scene_list_file) as json_file:
            self.dataset_list = json.load(json_file)

        self.result_file = result_file

    def eval_results(self):
        print("--------- Evaluating -----------")
        NOC = {}
        NOO = {}

        for iou_max in self.MAX_IOU:
            NOC[iou_max] = []
            NOO[iou_max] = []
            IOU_PER_CLICK_dict = None
            NOO_PER_CLICK_dict = None

            _, noc_perclass, noo_perclass, iou_per_click, noo_per_click = self.eval_per_class(iou_max, self.dataset_list)
            NOC[iou_max].append(noc_perclass)
            NOO[iou_max].append(noo_perclass)

            if IOU_PER_CLICK_dict == None:
                IOU_PER_CLICK_dict = iou_per_click
            else:
                for k in IOU_PER_CLICK_dict.keys():
                    IOU_PER_CLICK_dict[k] += iou_per_click[k]

            if NOO_PER_CLICK_dict == None:
                NOO_PER_CLICK_dict = noo_per_click
            else:
                for k in NOO_PER_CLICK_dict.keys():
                    NOO_PER_CLICK_dict[k] += noo_per_click[k]

        results_dict = {
            "NoC@50": sum(NOC[0.5]) / sum(NOO[0.5]),
            "NoC@65": sum(NOC[0.65]) / sum(NOO[0.65]),
            "NoC@80": sum(NOC[0.8]) / sum(NOO[0.8]),
            "NoC@85": sum(NOC[0.85]) / sum(NOO[0.85]),
            "NoC@90": sum(NOC[0.9]) / sum(NOO[0.9]),
            "IoU@1": IOU_PER_CLICK_dict["1.0"] / NOO_PER_CLICK_dict["1.0"],
            "IoU@3": IOU_PER_CLICK_dict["3.0"] / NOO_PER_CLICK_dict["3.0"],
            "IoU@5": IOU_PER_CLICK_dict["5.0"] / NOO_PER_CLICK_dict["5.0"],
            "IoU@10": IOU_PER_CLICK_dict["10.0"] / NOO_PER_CLICK_dict["10.0"],
            "IoU@15": IOU_PER_CLICK_dict["15.0"] / NOO_PER_CLICK_dict["15.0"],
        }
        print("****************************")
        print(results_dict)

        return results_dict

    def eval_per_class(self, MAX_IOU=0.8, dataset_=None):
        objects = {}
        for ii in dataset_.keys():
            scene = ii.replace("scene_", "")
            num_obj_validated = dataset_[ii]["number_of_objects_to_validate"]
            validation_key = "_".join((scene, str(num_obj_validated)))
            # objects[ii.replace("scene", "").replace("obj_", "")] = 1
            objects[validation_key] = 1
        print("number of objects kept: ", len(objects))

        results_dict_KatIOU = {}
        num_objects = 0
        ordered_clicks = []

        all_object = {}
        results_dict_per_click = {}
        results_dict_per_click_iou = {}
        all = {}
        with open(self.result_file, "r") as f:
            while True:
                # line = [instance_counter + idx, scene_name[idx], num_obj[idx], current_num_clicks / num_obj[idx], sample_iou.numpy())]
                line = f.readline()
                if not line:
                    break
                splits = line.rstrip().split(" ")
                scene_name = splits[1].replace("scene", "")
                object_id = splits[2]
                num_clicks = splits[3]
                iou = splits[4]

                if (scene_name + "_" + object_id) in objects:
                    if (scene_name + "_" + object_id) not in all_object:
                        all_object[(scene_name + "_" + object_id)] = 1
                        all[(scene_name + "_" + object_id)] = []
                    all[(scene_name + "_" + object_id)].append((num_clicks, iou))

                    if float(iou) >= MAX_IOU:
                        if (scene_name + "_" + object_id) not in results_dict_KatIOU:
                            results_dict_KatIOU[scene_name + "_" + object_id] = float(num_clicks)
                            num_objects += 1
                            ordered_clicks.append(float(num_clicks))

                    elif float(num_clicks) >= 20 and (float(iou) >= 0):
                        if (scene_name + "_" + object_id) not in results_dict_KatIOU:
                            results_dict_KatIOU[scene_name + "_" + object_id] = float(num_clicks)
                            num_objects += 1
                            ordered_clicks.append(float(num_clicks))

                    results_dict_per_click.setdefault(num_clicks, 0)
                    results_dict_per_click_iou.setdefault(num_clicks, 0)

                    results_dict_per_click[num_clicks] += 1
                    results_dict_per_click_iou[num_clicks] += float(iou)
                else:
                    # print(scene_name + '_' + object_id)
                    pass
        if len(results_dict_KatIOU.values()) == 0:
            print("no objects to eval")
            return 0

        click_at_IoU = sum(results_dict_KatIOU.values()) / len(results_dict_KatIOU.values())
        print("click@", MAX_IOU, click_at_IoU, num_objects, len(results_dict_KatIOU.values()))

        return (
            ordered_clicks,
            sum(results_dict_KatIOU.values()),
            len(results_dict_KatIOU.values()),
            results_dict_per_click_iou,
            results_dict_per_click,
        )
