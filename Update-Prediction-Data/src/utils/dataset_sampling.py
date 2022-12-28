from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import count

import sys
import os
from collections import defaultdict
import numpy as np
import pickle
import random

def dataset(base_dir, n):
    # print("base_dir : {}, n : {}".format(base_dir, n))
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            #print(file_path)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)

    tags = sorted(d.keys())

    X = []
    y = []

    min = len(d['0'])
    for class_index, class_name in enumerate(tags):
        if min > len(d[class_name]):
            min = len(d[class_name])

    for class_index, class_name in enumerate(tags):
        origin_filenames = d[class_name]
        filenames = random.sample(origin_filenames, k=min)

        for filename in filenames:
            with open(f'{filename}', 'rb') as f:
                img = pickle.load(f)

                X.append(img)
                y.append(class_index)

    X = np.array(X)
    y = np.array(y)

    return X, y, tags