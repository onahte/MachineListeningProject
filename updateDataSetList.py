import os
import CONFIG


with open(CONFIG.dataset_list, 'w+') as f:
    for root, dirs, files in os.walk(CONFIG.spectrograms):
        for idx in range(len(files)):
            comma=','
            if idx == 0:
                comma = ''
            f.write(comma + os.path.join(root, file))
f.close()

