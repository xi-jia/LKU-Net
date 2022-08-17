# LKU-Net
[LKU-Net](https://arxiv.org/pdf/2208.04939.pdf) is heavily built on [RepVGG](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html), [RepLK-ResNet](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Scaling_Up_Your_Kernels_to_31x31_Revisiting_Large_Kernel_Design_CVPR_2022_paper.html), [IC-Net](https://github.com/zhangjun001/ICNet), [SYM-Net,](https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration).

The data used in this work can be publicly accessed from [MICCAI 2021 L2R Challenge](https://learn2reg.grand-challenge.org/) and [TransMorph IXI](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md) page.

If you find the code helpful, please consider citing our work.

    @article{jia2022lkunet,
      title={U-Net vs Transformer: Is U-Net Outdated in Medical Image Registration?},
      author={Jia, Xi and Bartlett, Joseph and Zhang, Tianyang and Lu, Wenqi and Qiu, Zhaowen and Duan, Jinming},
      journal={arXiv preprint arXiv:2208.04939},
      year={2022}
    }

## IXI
### Data Preparing
We directly used [the preprocessed IXI data](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md) and followed the exact training, validation, and testing protocol.
### Training and Testing
Before training and testing, please change the image loading and model saving directories accordingly.

The kernel size of LKU-Net can be changed in  class `LK_encoder` in Models.py.

    python train.py --start_channel 8 --using_l2 2 --lr 1e-4 --checkpoint 403 --iteration 403001 --smth_labda 5.0
    python infer.py --start_channel 8 --using_l2 2 --lr 1e-4 --checkpoint 403 --iteration 403001 --smth_labda 5.0
    python compute_dsc_jet_from_quantiResult.py
Using the command above, one can easily reproduce our results.
Additionally, we provided the trained models for directly computing the reported results.

## Discussion
1) Batch size.
In the paper, we used `batch size = 1`, this parameter is however not carefully tuned. The only reason we use `batch size = 1`  is that we want to eliminate the effects caused by GPU memory, as some lighter GPUs may not be able to fit on larger batch sizes.
One may try on large different batch sizes if larger memory is available.
2) Kernel size.
In the 2D experiments, as we reported in our experiments, `LK==7`  produces the best results.
However, an `LK==5` gives better results on 3D IXI. We believe this may be caused by the limited training data. If larger datasets are available, one may try a larger kernel size.