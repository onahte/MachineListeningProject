import os
import CONFIG


with open(CONFIG.dataset_list, 'w+') as f:
    for root, dirs, files in os.walk(CONFIG.spectrograms):
        for file in files:
            f.write(os.path.join(root, file) + ',')
f.close()

