import time

import numpy as np

from .python_utils import safe_divide


class StatisticsMeter:
    def __init__(self):
        self.records = []

    def update(self, result):
        if isinstance(result, list) or isinstance(result, tuple):
            self.records += result
        else:
            self.records.append(result)

    def reset(self):
        self.records.clear()

    def sum(self):
        return np.sum(self.records)

    def mean(self):
        return np.mean(self.records)

    def std(self):
        return np.std(self.records)

    def median(self):
        return np.median(self.records)


class AccuracyMeter:
    def __init__(self, num_class):
        self.num_class = num_class
        self.num_correct = 0
        self.num_record = 0
        self.num_correct_per_class = [0] * num_class
        self.num_record_per_class = [0] * num_class

    def update(self, preds, labels):
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        results = np.equal(preds, labels)
        self.num_correct += np.sum(results)
        self.num_record += results.size
        for i in range(self.num_class):
            results_per_class = results[labels == i]
            self.num_correct_per_class[i] += np.sum(results_per_class)
            self.num_record_per_class[i] += results_per_class.size

    def overall_accuracy(self):
        return safe_divide(self.num_correct, self.num_record)

    def mean_accuracy(self):
        return np.mean([self.accuracy(i) for i in range(self.num_class)])

    def accuracy(self, class_id):
        return safe_divide(self.num_correct_per_class[class_id], self.num_record_per_class[class_id])


class PartMeanIoUMeter:
    r"""
    Mean IoU (Intersect over Union) metric for Part Segmentation task.
    """
    def __init__(self, num_class, num_part, class_id_to_part_ids):
        self.num_class = num_class
        self.num_part = num_part
        self.class_id_to_part_ids = class_id_to_part_ids
        self.ious = []
        self.ious_per_class = [[] for _ in range(num_class)]

    def update(self, preds, labels, class_ids):
        batch_size = preds.shape[0]
        for i in range(batch_size):
            self.update_one_sample(preds[i], labels[i], class_ids[i])

    def update_one_sample(self, preds, labels, class_id):
        ious = []
        part_ids = self.class_id_to_part_ids[class_id]
        for part_id in part_ids:
            labels_per_part = np.equal(labels, part_id)
            preds_per_part = np.equal(preds, part_id)
            intersect_per_part = np.sum(np.logical_and(labels_per_part, preds_per_part))
            union_per_part = np.sum(np.logical_or(labels_per_part, preds_per_part))
            if union_per_part > 0:
                iou = intersect_per_part / union_per_part
            else:
                iou = 1.
            ious.append(iou)
        iou = np.mean(ious)
        self.ious.append(iou)
        self.ious_per_class[class_id].append(iou)

    def mean_iou_over_instance(self):
        return np.mean(self.ious)

    def mean_iou_over_class(self):
        mean_iou_per_class = [self.mean_iou_per_class(i) for i in range(self.num_class)]
        return np.mean(mean_iou_per_class)

    def mean_iou_per_class(self, class_id):
        return np.mean(self.ious_per_class[class_id])


class MeanIoUMeter:
    def __init__(self, num_class):
        self.num_class = num_class
        self.intersect_per_class = [0] * num_class
        self.union_per_class = [0] * num_class

    def update(self, preds, labels):
        for class_id in range(self.num_class):
            preds_per_class = np.equal(preds, class_id)
            labels_per_class = np.equal(labels, class_id)
            intersect = np.count_nonzero(np.logical_and(preds_per_class, labels_per_class))
            union = np.count_nonzero(np.logical_or(preds_per_class, labels_per_class))
            self.intersect_per_class[class_id] += intersect
            self.union_per_class[class_id] += union

    def mean_iou(self):
        return np.mean([self.mean_iou_per_class(i) for i in range(self.num_class)])

    def mean_iou_per_class(self, class_id):
        return safe_divide(self.intersect_per_class[class_id], self.union_per_class[class_id])


class StatisticsDictMeter:
    def __init__(self):
        self.meter_dict = {}

    def register_meter(self, key):
        self.meter_dict[key] = StatisticsMeter()

    def reset_meter(self, key):
        self.meter_dict[key].reset()

    def check_key(self, key):
        if key not in self.meter_dict:
            raise KeyError('No meter for key "{}".'.format(key))

    def update(self, key, value):
        self.check_key(key)
        self.meter_dict[key].update(value)

    def update_from_result_dict(self, result_dict):
        if not isinstance(result_dict, dict):
            raise TypeError('`result_dict` must be a dict, but {} is used.'.format(type(result_dict)))
        for key, value in result_dict.items():
            if key in self.meter_dict:
                self.meter_dict[key].update(value)

    def sum(self, key):
        self.check_key(key)
        return self.meter_dict[key].sum()

    def mean(self, key):
        self.check_key(key)
        return self.meter_dict[key].mean()

    def std(self, key):
        self.check_key(key)
        return self.meter_dict[key].std()

    def median(self, key):
        self.check_key(key)
        return self.meter_dict[key].median()

    def summary(self):
        items = ['{}: {:.3f}'.format(key, meter.mean()) for key, meter in self.meter_dict.items()]
        summary = items[0]
        for item in items[1:]:
            summary += ', {}'.format(item)
        return summary


class Timer:
    def __init__(self):
        self.total_prepare_time = 0
        self.total_process_time = 0
        self.num_prepare_time = 0
        self.num_process_time = 0
        self.last_time = time.time()

    def reset_stats(self):
        self.total_prepare_time = 0
        self.total_process_time = 0
        self.num_prepare_time = 0
        self.num_process_time = 0

    def reset_time(self):
        self.last_time = time.time()

    def add_prepare_time(self):
        current_time = time.time()
        self.total_prepare_time += current_time - self.last_time
        self.num_prepare_time += 1
        self.last_time = current_time

    def add_process_time(self):
        current_time = time.time()
        self.total_process_time += current_time - self.last_time
        self.num_process_time += 1
        self.last_time = current_time

    def get_prepare_time(self):
        return self.total_prepare_time / self.num_prepare_time

    def get_process_time(self):
        return self.total_process_time / self.num_process_time
