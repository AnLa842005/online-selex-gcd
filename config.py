import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent

# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '../../data/datasets/cifar10'
cifar_100_root = '../../data/datasets/cifar100'
cub_root = os.environ.get('SELEX_DATA_ROOT', str(_REPO.parent / 'data'))
car_root = '../../data/datasets/stanford_car/cars_{}/'
pets_root = '../../data/datasets/pets/'

aircraft_root = '../../data/datasets/aircraft/fgvc-aircraft-2013b'
herbarium_dataroot = '../../data/datasets/herbarium_19/'
imagenet_root = '../../data/datasets/ImageNet/'#ILSVRC12'

# OSR Split dir
osr_split_dir = str(_REPO / 'data' / 'ssb_splits')

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path2 = '../../pretrained_models/dino/dinov2_vitb14_reg4_pretrain.pth'#

dino_pretrain_path = os.environ.get('SELEX_PRETRAINED_PATH', str(_REPO.parent / 'pretrained' / 'dino_vitbase16_pretrain.pth'))
warmup_pretrain_path = '../../pretrained_models/GCDTeacher/model_best.pth'
feature_extract_dir = str(Path(os.environ.get('SELEX_OUTPUT_ROOT', str(_REPO.parent / 'runs' / 'selex'))) / 'features')     # Extract features to this directory
exp_root = os.environ.get('SELEX_OUTPUT_ROOT', str(_REPO.parent / 'runs' / 'selex'))          # All logs and checkpoints will be saved here