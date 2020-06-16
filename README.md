# Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image
In this repository we provide the official implementation of Self2Self
with Dropout.
## General Information
- Codename: Self2Self (CVPR 2020)
- Writers: Yuhui Quan (csyhquan@scut.edu.cn); Mingqin Chen
  (csmingqinchen@mail.scut.edu.cn); Tongyao Pang (matpt@nus.edu.sg); Hui
  Ji (matjh@nus.edu.sg)
- Institute: School of Computer Science and Engineering, South China
  University of Technology; Department of Mathematics, National University of Singapore

For more information please see:
- [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Quan_Self2Self_With_Dropout_Learning_Self-Supervised_Denoising_From_Single_Image_CVPR_2020_paper.pdf)
- [[supmat]](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Quan_Self2Self_With_Dropout_CVPR_2020_supplemental.pdf)
- [[website]](https://csyhquan.github.io/)


## Requirements
Here is the list of libraries you need to install to execute the code:
* Python 3.6
* Tensorflow-gpu 1.14.0
* keras
* scikit-image
* scipy
* cv2 (opencv for python)

All of them can be installed via `conda` (`anaconda`), e.g.
```
conda install scikit-image
```

## How to Execute Demo
1. (Optional) Download the dataset you want and save them in
   './testsets/'.
2. Run the demo code in `demo_denosing.py`.

## Citation
```
@InProceedings{Quan_2020_CVPR,
author = {Quan, Yuhui and Chen, Mingqin and Pang, Tongyao and Ji, Hui},
title = {Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Contacts
For questions, please send an email to **csmingqinchen@mail.scut.edu.cn**
