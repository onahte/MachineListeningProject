import os
from argparse import Namespace


# Local
#home = '/Users/eo/Documents/NJIT/CS698-MachineListening/Final'
home = '/Volumes/ExtremeSSD/Data'
#home = '/home/e/eo238/MaLi/'

# Lochness
#home = '/home/e/eo238/MaLi/'

dataset = os.path.join(home, 'ML_Dataset/wav')
dataset_list = os.path.join(home, 'ML_Dataset/dataset_list.txt')
spectrograms = os.path.join(home, 'ML_Dataset/specs')
last_file = os.path.join(home, 'ML_Datatset/last_file.txt')
label_count = os.path.join(home, 'ML_Dataset/label_count.txt')

wav = Namespace(
    n_fft=2048,
    hop_length=512,
    n_mels=40
)

model_config= Namespace(
    random_seed = 1111,
    batch_size = 16,
    epochs = 40,
    learning_rate = 3e-3,
    num_classes = 10,
    patch_height = 64,
    patch_width = 1,
    img_size = 64,
    in_channels = 3,
    num_heads = 8,
    dropout = 0.1,
    hidden_dim = 768,
    adam_weight_decay = 0,
    adam_betas = (0.9, 0.999),
    num_encoders = 8,
    embed_dim = 128,
    num_patches = (64 ** 2) // (64 * 1)

)
