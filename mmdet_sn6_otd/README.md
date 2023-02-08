## Installation and Get Started

Required environments:
* Linux
* Python 3.7
* PyTorch 1.8.0
* CUDA 10.1
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)


Install SN6_OTD :

Note that our SN6_OTD is based on the [MMDetection 2.19.0](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/EIS-VIPG/SN6_OTD.git
cd mmdet_sn6_otd
pip install -r requirements/build.txt
pip install -v -e .
```

Get Started：

We use the [MMSelfSup](https://github.com/open-mmlab/mmselfsup) to apply the OS_SSL pre-training. The pre_trained model can be Download from  [Google Drive](https://drive.google.com/file/d/1Oky3l5LpDt0aaULx1C0x7sJuwgFtykdt/view?usp=share_link).

Before training the SAR detector, we should train the optical detector first, so that we can get the optical feature maps for guidance. 

* To train the optical detector

```
./tools/dist_train.sh config_otd/baseline-oiltank-opt-v1/faster_rcnn_r50_imagenet.py 2
```

* To get the feature maps of optical images

```
python tools/test.py config_otd/baseline-oiltank-opt-v1/faster-output_backbone.py work_dirs/baseline-oiltank-opt-v1/faster_rcnn_r50_imagenet/epoch_36.pth --eval bbox
```

* To train the SAR detector

```
./tools/dist_train.sh config_otd/SN6_OTD_faster_rcnn_r50_OS_SSL_AKD.py 2
```

(The trained model of our method can be downloaded from  [Google Drive](https://drive.google.com/file/d/1Jhzex9eUGKlfcjICmAUxhpPJcyzdiWR7/view?usp=share_link).)
