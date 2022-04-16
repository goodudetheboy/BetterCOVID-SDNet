# BetterCOVIDSD-Net

## Segmentation-based cropping

Crop data based on the spatial position of the lungs to make data more clean

### Instruction

1. Download the pretrained weight `unet-lung-seg.hdf5` [here on Kaggle](https://www.kaggle.com/datasets/salemrezzag/unet-lung-seghdf5).
2. Create a new directory named `input`.
3. Put the pretrained weight `unet-lung-seg.hdf5` you have just downloaded inside the `input` folder.
5. Run all `seg_crop.ipynb` or `seg_crop.py`, whichever is your preference.

Alternatively, in a terminal you can execute the following code
```
mkdir ../input
mv /full/path/to/weight/unet-lung-seg.hdf5 ./input
```