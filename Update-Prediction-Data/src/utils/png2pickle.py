import os
import sys
import pickle
import numpy as np
import scipy.misc
import imageio.v2 as imageio
import cv2

"""
import pickle

temp_dict = {'name': 'S', 'id': 1}

# 데이터 저장
with open('filename.pkl', 'wb') as f:
	pickle.dump(temp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
# 데이터 로드
with open('filename.pkl', 'rb') as f:
	data = pickle.load(f)
"""

base_dir = sys.argv[1]

for root, subdirs, files in os.walk(base_dir):
    for filename in files:
        img = imageio.imread(f'{root}/{filename}')

        new_name = filename.split('.')[0] + '.pkl'
        with open(f'{root}/{new_name}', 'wb') as f:
            pickle.dump(img, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.remove(f'{root}/{filename}')
        



