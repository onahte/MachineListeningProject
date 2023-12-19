import os
from argparse import Namespace
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
 

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
    epochs = 20,
    learning_rate = 3e-3,
    num_classes = 932,
    patch_height = 256,
    patch_width = 1,
    img_size = 256,
    in_channels = 4,
    num_heads = 1,
    dropout = 0.1,
    hidden_dim = 768,
    adam_weight_decay = 0,
    adam_betas = (0.9, 0.999),
    num_encoders = 4,
    emb_dim = 768,
    num_patches = (256 ** 2) // (256 * 1),
    confusion_mat = os.path.join(home, 'confusion_matrix.png')
)
