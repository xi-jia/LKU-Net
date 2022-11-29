# Large Kernel U-Net (LKU-Net) for Medical Image Registration
Key facts of [LKU-Net](https://arxiv.org/pdf/2208.04939.pdf):
* LKU-Net is inspired by [RepVGG](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html), [RepLK-ResNet](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Scaling_Up_Your_Kernels_to_31x31_Revisiting_Large_Kernel_Design_CVPR_2022_paper.html), [IC-Net](https://github.com/zhangjun001/ICNet), [SYM-Net,](https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration). 

* LKU-Net ranks first on the 2021 MICCAI [Learn2Reg](https://learn2reg.grand-challenge.org/evaluation/task-3-validation/leaderboard/) Challenge Task 3 Validation as of 18th September, 2022

* LKU-Net is also the winning entry on the 2022 MICCAI [Learn2Reg](documents/result.png) Challenge Task 1.  

* Please see our [poster](http://www.cs.bham.ac.uk/~duanj/slides/UNet-TransMorph.pdf) or watch the [video](http://www.cs.bham.ac.uk/~duanj/slides/UNet-TransMorph.mp4) for further information. 

If you find this repo helpful, please consider citing our work.

    @article{jia2022lkunet,
      title={U-Net vs Transformer: Is U-Net Outdated in Medical Image Registration?},
      author={Jia, Xi and Bartlett, Joseph and Zhang, Tianyang and Lu, Wenqi and Qiu, Zhaowen and Duan, Jinming},
      journal={arXiv preprint arXiv:2208.04939},
      year={2022}
    }
    
Note that the data used in this work can be publicly accessed from [MICCAI 2021 L2R Challenge](https://learn2reg.grand-challenge.org/) and [TransMorph IXI](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md).

## IXI
### Updates (2022 Sep 08)

Thanks for the kind reminder from [@Junyu](https://github.com/junyuchen245), the leading author of [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) (which is accepted by MIA now), that using bilinear interpolation improves the Dice scores when even warping image labels.

Hence, in line with TransMorph, we accordingly updated all the results (see the table below) on the IXI data by changing the interpolation of LKU-Net from 'nearest' to 'bilinear'.
| Dice             | Nearest     | Bilinear    |
|------------------|-------------|-------------|
| TransMorph       | 0.746±0.128 | 0.753±0.123 |
| TransMorph-Bayes | 0.746±0.123 | 0.754±0.124 |
| TransMorph-bspl  | 0.752±0.128 | 0.761±0.122 |
| TransMorph-diff  | 0.599±0.156 | 0.594±0.163 |
| U-Net4           | 0.727±0.126 | 0.733±0.125 |
| U-Net-diff4      | 0.744±0.123 | 0.751±0.123 |
| LKU-Net4,5       | 0.752±0.131 | 0.758±0.130 |
| LKU-Net-diff4,5  | 0.746±0.133 | 0.752±0.132 |
| LKU-Net8,5       | 0.757±0.128 | 0.765±0.129 |
| LKU-Net-diff8,5  | 0.753±0.132 | 0.760±0.132 |
| LKU-Net-diff16,5 | 0.757±0.132 | 0.764±0.131 |

Note that we list the results of TransMorph and LKU-Net only, the results of other compared methods can be found in the [IXI page of TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md).

Additionaly, we add one more results of the LKU-Net-Dif(16,5), the model can be downloaded from [this Google Drive Link](https://drive.google.com/file/d/1VzgsZuHoMxobO5CxKDNGcM46Q7_n5-FA/view?usp=sharing).

### Data Preparing
We directly used [the preprocessed IXI data](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md) and followed the exact training, validation, and testing protocol.
### Training and Testing
Before training and testing, please change the image loading and model saving directories accordingly.

The kernel size of LKU-Net can be changed in  class `LK_encoder` in Models.py.

    python train.py --start_channel 8 --using_l2 2 --lr 1e-4 --checkpoint 403 --iteration 403001 --smth_labda 5.0
    python infer.py / infer_bilinear.py --start_channel 8 --using_l2 2 --lr 1e-4 --checkpoint 403 --iteration 403001 --smth_labda 5.0
    python compute_dsc_jet_from_quantiResult.py
Using the command above, one can easily reproduce our results.
Additionally, we provided the trained models for directly computing the reported results.

## L2R 2021 OASIS

Pretrained models can be downloaded from this [Google Drive Link](https://drive.google.com/drive/folders/1lEVKNEyUqKMVqhtjr9iL27fLs8RuR1oe?usp=share_link)

## Discussion
1) Batch size.
In the paper, we used `batch size = 1`, this parameter is however not carefully tuned. The only reason we use `batch size = 1`  is that we want to eliminate the effects caused by GPU memory, as some lighter GPUs may not be able to fit on larger batch sizes.
One may try on large different batch sizes if larger memory is available.
2) Kernel size.
In the 2D experiments, as we reported in our experiments, `LK==7`  produces the best results.
However, an `LK==5` gives better results on 3D IXI. We believe this may be caused by the limited training data. If larger datasets are available, one may try a larger kernel size.
