# Introduction
This is a repository of the official implementation of the following paper: 
* [[Paper]](https://ieeexplore.ieee.org/document/9924205)[[Code]](https://github.com/EIS-VIPG/SN6_OTD.git) Optical-Enhanced Oil Tank Detection in High-Resolution SAR Images (TGRS, 2022)

  

## DataSet
We conduct the **SpaceNet6-OTD** (Oil Tank Detection) datasets based on the [SpaceNet6](https://spacenet.ai/sn6-challenge/) dataset. The annotations of oil tank is provided in the projectes. 




## Installation and Get Started

Required environments:
* Linux
* Python 3.7
* PyTorch 1.8.0
* CUDA 10.1
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)


Install SN6_OTD:

Note that our SN6_OTD is based on the [MMDetection 2.19.0](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/EIS-VIPG/SN6_OTD.git
cd mmdet_sn6_otd
pip install -r requirements/build.txt
pip install -v -e .
```

## Citation

If you use this repo in your research, please consider citing these papers.

```
@ARTICLE{9924205,
author={Zhang, Ruixiang and Guo, Haowen and Xu, Fang and Yang, Wen and Yu, Huai and Zhang, Haijian and Xia, Gui-Song},
journal={IEEE Transactions on Geoscience and Remote Sensing}, 
title={Optical-Enhanced Oil Tank Detection in High-Resolution SAR Images}, 
year={2022},
volume={60},
number={},
pages={1-12},
doi={10.1109/TGRS.2022.3215543}}
```

## References
* [MMDetection](https://github.com/open-mmlab/mmdetection)
* [SpanceNet6](https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions)



