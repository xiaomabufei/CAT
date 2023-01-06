import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import math
class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]
        self.dynamic_radius = [torch.Tensor([10]) for i in range(total_num_classes)]
        self.mean = [None for _ in range(total_num_classes)]
    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(list(item))
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item))
                all_items.append(items)
            return all_items

    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])
class Memory_loss:
    def __init__(self, length, shuffle=False):
        self.shuffle = shuffle
        self.length = length
        self.Ws = 0.2
        self.Wf = 0.8
        self.memory = deque(maxlen=length)
        Sum1 = sum([i for i in range(1, length+1-20)])
        self.weight1 = [i/Sum1 for i in range(1, length+1-20)]
        Sum2 = sum([i for i in range(1, 21)])
        self.weight2 = [i/Sum2 for i in range(1, 21)]
    def add(self, loss):
        self.memory.append(loss)
    def delta(self):
        loss1 = 0
        loss2 = 0
        for idx, item in enumerate(self.memory):
            if idx < self.length-20:
                loss1 = loss1 + self.memory[idx]*self.weight1[idx]
            else:
                loss2 = loss2 + self.memory[idx-self.length+20]*self.weight2[idx-self.length+20]
        return loss2/loss1
    def Adaptive_weight(self, delta_loss):
        delta_weight=0
        assert delta_loss >= 0, 'loss should be positive'
        if delta_loss > 1:
            delta_weight = (1/(1 + math.exp(1-delta_loss)))/2
        else:
            delta_weight = -delta_loss/3
        return delta_weight

    def update_weight(self, delta_weight):
        self.Ws = self.Ws - delta_weight*self.Ws
        self.Wf = self.Wf + delta_weight*self.Wf
        total = self.Ws+self.Wf
        self.Ws, self.Wf = self.Ws/total, self.Wf/total
        

if __name__ == "__main__":
    memory = Memory_loss(20)
    for i in range(20):
        memory.add(i)
    loss = memory.delta()
    print(loss)