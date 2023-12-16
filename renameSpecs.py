import os
import CONFIG


label_dict = {}
label = 0
for roots, dirs, files in os.walk(CONFIG.spectrograms):
    for file in files:
        old_filename = os.path.join(roots, file)
        file = file.split('_')
        speaker = file[0]
        speaker_label = 0
        if speaker not in label_dict:
            label_dict[speaker] = label
            speaker_label = label
            label += 1
        else:
            speaker_label = label_dict[speaker]
        new_filename = str(speaker_label) + '_' + file[1]
        os.rename(old_filename, os.path.join(roots, new_filename))

with open(CONFIG.label_count, 'w+') as f:
    f.write(str(label))
f.close()
