import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
from tqdm import trange, tqdm
from PIL import Image

files_path = '/home/ubuntu/2022_VAIV_Dataset/Image/1/224x224/Kospi'
files = os.listdir(files_path)
files = [file for file in files if file[0] == 'A']
count = 0
images = []
f_output = open('try/curr_image_info_kospi.txt', 'w')
for file in tqdm(files):
    f_output.write(f"{file}\n")
    images.append(file)
f_output.close()