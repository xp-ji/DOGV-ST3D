import json
import numpy as np

def read_json_param(filename):
    with open(filename, 'r') as lcf:
        config_param = json.load(lcf)
        return config_param

class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(val, self.max)

def calculate_accuracy_percent(outputs, targets):
    batch_size = targets.size(0)
 
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum()
    percent_accuracy = n_correct_elems.mul(100.0 / batch_size)

    return percent_accuracy, n_correct_elems
