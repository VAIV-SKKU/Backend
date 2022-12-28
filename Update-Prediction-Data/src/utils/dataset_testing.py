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

def dataset_testing(base_dir, n):
    # print("base_dir : {}, n : {}".format(base_dir, n))
    print("dataset_testing")
    #exit()
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
    f_names = []
    tickers = []
    date = []

    min = len(d['0'])
    for class_index, class_name in enumerate(tags):
        if min > len(d[class_name]):
            min = len(d[class_name])

    for class_index, class_name in enumerate(tags):
        origin_filenames = d[class_name]
        filenames = random.sample(origin_filenames, k=min)

        for filename in filenames:
            with open(f'{filename}', 'rb') as f:
                # print(filename.split('/')[-1])
                img = pickle.load(f)

                filename_ = filename.split('/')[-1]
                tickers.append(filename_.split('_')[0])
                date.append(filename_.split('_')[1])
                X.append(img)
                y.append(class_index)

                #exit()

    X = np.array(X)
    y = np.array(y)

    return X, y, tags, tickers, date