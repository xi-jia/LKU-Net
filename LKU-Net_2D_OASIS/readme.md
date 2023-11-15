This folder contains the training/testing code and trained models used in the original LK-UNet paper on 2D OASIS Dataset.
Table 1 C1 LKU-Net 7 8 Y Y 76.55(4.06)


```
Training
CUDA_VISIBLE_DEVICES=0 python train.py --start_channel 8 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 402001 --checkpoint 4020
Testing
CUDA_VISIBLE_DEVICES=0 python infer.py --start_channel 8 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 402001 --checkpoint 4020
```
