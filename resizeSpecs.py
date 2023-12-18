import os
from PIL import Image

import CONFIG

S = open(CONFIG.dataset_list, 'r')
S = S.read()
specs = S.split(',')
for spec in specs:
    img = Image.open(spec)
    img_resize = img.resize((256, 256))
    filename = spec.split('/')[-1]
    new_filepath = os.path.join('/Volumes/ExtremeSSD/Data/ML_Dataset/specs_sm/', filename)    
    img_resize.save(new_filepath)

