import os

import librosa
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

import CONFIG as C


def buildFileList(data_path):
    f = open(C.dataset_list, 'w+')
    for roots, dirs, files in os.walk(data_path):
        for file in files:
            file_format = file.split('.')
            if not file_format[-1] == 'wav' or '_' in file_format[-2]:
                continue
            path = os.path.join(roots, file)
            f.writelines(path + ',')


def saveSpec(file_path, last_file):
    file_index = '1'
    last_id = ''
    path_split = file_path.split('/')
    curr_id = path_split[-3]
    curr_id = curr_id[2:]
    if not last_file == None:
        last_id = last_file[0]
        idx = int(last_file[1])
        if curr_id == last_id:
            file_index = str(idx + 1)
    print(f'file index: {file_index} | last id: {last_id} | match: {curr_id == last_id}')
    save_path = os.path.join(C.spectrograms, curr_id + '_' + file_index + '.png')
    print(save_path)
    wavToMelSpec(file_path, save_path)
    wavToMelSpec.cache_clear()
    return [curr_id, file_index]


@lru_cache(maxsize=128)
def wavToMelSpec(file, save_path):
    print(f'Loading {file}')
    signal, sr = librosa.load(file, duration=3)
    mel_spec = librosa.feature.melspectrogram(y=signal,
                                              sr=sr,
                                              hop_length=C.wav.hop_length,
                                              n_fft=C.wav.n_fft,
                                              n_mels=C.wav.n_mels)
    fig, ax = plt.subplots(1, figsize=(64, 64))
    mel_spec_img = librosa.display.specshow(np.log(mel_spec), sr=sr)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close()
    return


if __name__=='__main__':
    if not os.path.exists(C.spectrograms):
       os.mkdir(C.spectrograms)
    file_list = []
    if not os.path.exists(C.dataset_list):
        buildFileList(C.dataset)
    d = open(C.dataset_list, 'r').read()
    d = d.split(',')
    file_list = [file for file in d]
    last_file = None
    if os.path.exists(C.last_file):
        f = open(C.last_file, 'r')
        last_file = f.read()
        f.close()
    for i in range(300, len(file_list)):
        lf = saveSpec(file_list[i], last_file)
        last_file = lf
    with open(C.last_file, 'w+') as f:
        f.writelines(last_file)
