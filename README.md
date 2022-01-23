# GainedVAE

A Pytorch Implementation of a continuously rate adjustable learned image compression framework, Gained Variational Autoencoder(GainedVAE). 

*Note that This Is Not An Official Implementation Code.*

More details can be found in the following paper:

>[Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation.](https://openaccess.thecvf.com/content/CVPR2021/html/Cui_Asymmetric_Gained_Deep_Image_Compression_With_Continuous_Rate_Adaptation_CVPR_2021_paper.html)  
>Huawei Technologies, CVPR 2021  
>Ze Cui, Jing Wang, Shangyin Gao, Tiansheng Guo, Yihui Feng, Bo Bai

Todo:
Reproduce Implementation of the following paper:
>[INTERPOLATION VARIABLE RATE IMAGE COMPRESSION](https://arxiv.org/abs/2109.09280)  
>Alibaba Group, arxiv 2021.9.20  
>Zhenhong Sun, Zhiyu Tan, Xiuyu Sun, Fangyi Zhang, Yichen Qian, Dongyang Li, Hao Li

# Environment

* Python == 3.7.10
* Pytorch == 1.7.1
* CompressAI

# Dataset
#### Training set
I use a part of the OpenImages Dataset to train the models (train06, train07, train08, about 54w images). You can download from here. [Download OpenImages](https://www.appen.com.cn/datasets/open-images-annotated-with-bounding-boxes/)
Maybe train08 (14w images) is enough.

#### Test set
[Download Kodak dataset](http://r0k.us/graphics/kodak/)

The dataset fold structure is as follows:
```
.dataset/
│  
├─test
│      kodim01.png
│      kodim02.png
│      kodim03.png
...
├─train
│      000002b66c9c498e.jpg
│      000002b97e5471a0.jpg
│      000002c707c9895e.jpg
...
```

# Train Your Own Model
>python3 trainGain.py -d /path/to/your/image/dataset/ --epochs 200 -lr 1e-4 --batch-size 16 --model-save /path/to/your/model/save/dir --cuda

# Result
I try to train the Gained Mean-Scale Hyperprior model/Gained Scale Hyperprior model. See details in ./results
I retrained the single rate baseline but can not achieve the official performance. Results from Google tensorflow/compression library is very strong probably because of their large and diverse training data set and long training time. 

![results1](https://github.com/mmSir/GainedVAE/tree/master/results/MSHyperprior%20Results.jpg)

![results2](https://github.com/mmSir/GainedVAE/tree/master/results/ScaleHyperprior%20Results.jpg)

# Acknowledgement

The framework is based on CompressAI, I add the model in compressai.models.gain, compressai.models.gain_utils.  
And trainGain/trainGain.py is modified with reference to compressai_examples/train.py.

# More Variable Rate Image Compression Repositories
["Variable-Rate Deep Image Compression through Spatially-Adaptive Feature Transform"](https://arxiv.org/abs/2108.09551) (ICCV 2021).  
[code](https://github.com/micmic123/QmapCompression)

["Variable Bitrate Image Compression with Quality Scaling Factors"](https://ieeexplore.ieee.org/abstract/document/9053885/) (ICASSP 2020).  
[code](https://github.com/tongxyh/ImageCompression_VariableRate)

["Variable Rate Deep Image Compression with Modulated Autoencoders"](https://ieeexplore.ieee.org/document/8977394) (IEEE SPL 2020)  
[code](https://github.com/FireFYF/modulatedautoencoder)

["Slimmable Compressive Autoencoders for Practical Neural Image Compression"](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_Slimmable_Compressive_Autoencoders_for_Practical_Neural_Image_Compression_CVPR_2021_paper.html) (CVPR 2021)  
[code](https://github.com/FireFYF/SlimCAE)


# Contact
Feel free to contact me if there is any question about the code or to discuss any problems with image and video compression. (mxh_wine@qq.com)
