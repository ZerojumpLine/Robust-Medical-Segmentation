# Installation

For Conda users, you can create a new Conda environment using

```
conda create -n robustseg python=3.10
```

after activating the environment with 
```
source activate robustseg
```
try to install all the dependencies with

```
pip install -r requirements.txt
```
also install the conda environment for the jupyter notebook kernel.

```
python -m ipykernel install --user --name=robustseg
```

## Dataset

Download the [ATLAS data](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html), [prostate MRI data](https://liuquande.github.io/SAML/) and [Cardiac MRI data](https://www.ub.edu/mnms/), and put extracted the data into `./data`.

```
data/
├── ATLAS_R2.0/
  ├── ATLAS_2/
    ├── Training/
    ├── ...
    └── ...
├── Processed_data_nii/
  ├── BIDMC/
  ├── ...
  └── ...
├── OpenDataset/
  ├── Training/
  ├── ...
  └── ...
```

Run `datapreprocessing_ATLAS.ipynb`, `datapreprocessing_MnMCardiac.ipynb` and `datapreprocessing_Prostate.ipynb` in the `./data` folder to preprocess the data step by step. After preprocessing, we should have the data format like:

```
data/
├── Dataset_Cardiac/
  ├── 1/
  ├── ...
  └── 5/
├── Dataset_Prostate/
  ├── ISBI/
  ├── ...
  ├── HK/
├── Dataset_Brain_lesion/
  ├── GE Signa Excite/
  ├── ...
  └── Siemens Skyra/
```

## Training

### Stardard training.

Cardiac
```
python UNetSegmentationTrain.py --name 3DUNet_vanilla_Cardiac_det --tensorboard --features 30 --deepsupervision --batch-size 2 --patch-size 128 128 8 --epochs 1000 --evalevery 100 --numIteration 100 --sgd0orAdam1orRms2 0 --lr 1e-2 --print-freq 20 --ATLAS0Cardiac1Prostate2 1 --gpu 0 --det
```

Prostate
```
python UNetSegmentationTrain.py --name 3DUNet_vanilla_Prostate_det --tensorboard --features 30 --deepsupervision --batch-size 2 --patch-size 64 64 32 --epochs 1000 --evalevery 100 --numIteration 100 --sgd0orAdam1orRms2 0 --lr 1e-2 --print-freq 20 --ATLAS0Cardiac1Prostate2 2 --gpu 0 --det
```

Brain lesion

```
python UNetSegmentationTrain.py --name 3DUNet_vanilla_ATLAS_det --tensorboard --features 30 --deepsupervision --batch-size 2 --patch-size 128 128 128 --epochs 1000 --evalevery 100 --numIteration 100 --sgd0orAdam1orRms2 0 --lr 1e-2 --print-freq 20 --ATLAS0Cardiac1Prostate2 0 --gpu 0 --det
```

### Robust training.

### Class balanced training.

### Robust class balanced training

## Evaluation

```
CUDA_VISIBLE_DEVICES=0 python src/train_adv_supervised_segmentation_triplet.py --cval 0 --seed 40 --json_config_path ./config/ACDC/1500_epoch/MICCAI2022_MaxStyle.json --log --data_setting 10 --no_train
```