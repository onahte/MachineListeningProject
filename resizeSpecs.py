import os
from PIL import Image

import CONFIG


if not os.path.exists('/Volumes/ExtremeSSD/Data/ML_Dataset/specs_100/'):
    os.mkdir('/Volumes/ExtremeSSD/Data/ML_Dataset/specs_100/')

S = open(CONFIG.dataset_list, 'r')
S = S.read()
specs = S.split(',')
for spec in specs:
    img = Image.open(spec)
    img_resize = img.resize((100, 100))
    filename = spec.split('/')[-1]
    new_filepath = os.path.join('/Volumes/ExtremeSSD/Data/ML_Dataset/specs_100/', filename)    
    img_resize.save(new_filepath)

