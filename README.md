# Installation

For Conda users, you can create a new Conda environment using

```
conda create -n robustseg python=3.9
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

Download the ATLAS data from [this website](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html). After putting the downloaded data in `./data`, run `datapreprocessing_ATLAS.ipynb` in the `./data` folder to preprocess the data step by step.

Download the preprocessed prostate MRI from [this link](https://drive.google.com/file/d/1fMPqHETCvohh1e6D2rIlddWPLfHuyI8j/view) and contact us for preprocessed cardiac MRI after the usage permissions are granted.

```
data/
├── MICCAI2022_cardiac_dataset/
  ├── ACDC/
  ├── ACDC_artefacted/
  ├── MM/
  └── MSCMRSeg_resampled/
├── MICCAI2022_multi_site_prostate_dataset/
  ├── reorganized/
    ├── A-ISBI/
    ├── ...
    ├── F-HK/
    └── G-MedicalDecathlon/
├── Brain_lesion_dataset/
  ├──reorganized/
    ├── GE Signa Excite/
    ├── ...
    └── Siemens Skyra/
```

## Training

### Stardard training.

Cardiac
```
python src/train_adv_supervised_segmentation_triplet.py --json_config_path ./config/ACDC/1500_epoch/standard_training.json --cval 0 --seed 40 --data_setting 'standard' --auto_test --log
```

Prostate
```
python src/train_adv_supervised_segmentation_triplet.py --json_config_path ./config/Prostate/MICCAI2022_MaxStyle.json  --cval 0 --seed 40 --data_setting 'all' --auto_test --log
```

Brain lesion


### Robust training.

### Class balanced training.

### Robust class balanced training

## Evaluation

```
python src/train_adv_supervised_segmentation_triplet.py --cval 0 --seed 40 --json_config_path ./config/ACDC/1500_epoch/MICCAI2022_MaxStyle.json --log --data_setting 10 --no_train
```