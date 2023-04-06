# GainedVAE

A Pytorch Implementation of a continuously rate adjustable learned image compression framework, Gained Variational Autoencoder(GainedVAE).

*Note that It Is Not An Official Implementation Code.*

More details can be found in the following paper:

>[Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation.](https://openaccess.thecvf.com/content/CVPR2021/html/Cui_Asymmetric_Gained_Deep_Image_Compression_With_Continuous_Rate_Adaptation_CVPR_2021_paper.html)  
>Huawei Technologies, CVPR 2021  
>Ze Cui, Jing Wang, Shangyin Gao, Tiansheng Guo, Yihui Feng, Bo Bai


The differences with the original paper are not limited to the following:
* The number of model channels may be different.
* The pre-defined lambda set may be different.
* Asymmetric Gaussian entropy model is not implemented.

-------------------------------2023.4.6 Update--------------------------------
* eval script is uploaded.
* A spatial-channel feature modulation framework with Gain Units is Added in compressai.models.gain . You could find more details in [1][<sup>1</sup>](#refer-anchor-1).

# Environment
* Python == 3.7.10
* Pytorch == 1.7.1
* CompressAI

# Dataset
I use a part of the OpenImages Dataset to train the models (train06, train07, train08, about 54w images). You can download from here[Download OpenImages](https://www.appen.com.cn/datasets/open-images-annotated-with-bounding-boxes/).
Maybe train08 (14w images) is enough.


# Train Your Own Model
>python3 trainGain.py -d /path/to/your/image/dataset/ --epochs 200 -lr 1e-4 --batch-size 16 --model-save /path/to/your/model/save/dir --cuda


# Eval Your Own Model
Currently only two eval modes is supported, one is 'gain' and the other is 'scgain'.
>python3 eval_gain.py -d /path/to/your/image/dataset/ --checkpoint /path/to/your/model.pth --logpath /path/to/save/result/log/ --cuda --mode (gain/scgain)


# Result
I try to train the Gained Mean-Scale Hyperprior model/Gained Scale Hyperprior model. See details in ./results
I retrained the single rate baseline but can not achieve the official performance. Results from Google tensorflow/compression library is very strong probably because of their large and diverse training data set and long training time. 

![results1](https://github.com/mmSir/GainedVAE/blob/master/results/mshyper.png)

![results2](https://github.com/mmSir/GainedVAE/blob/master/results/scalehyper.png)

# Pretrained Model
You can download the checkpoint trained by me from [Pretrained Model](https://drive.google.com/file/d/1EqemQB54rz4GZ1vtwCu98LnRZn8gSHhJ/view?usp=sharing).

# Acknowledgement

The framework is based on CompressAI, I add the model in compressai.models.gain, compressai.models.gain_utils.  
And trainGain/trainGain.py is modified with reference to compressai_examples/train.py.

# More Variable Rate Image Compression Repositories
[1] ["Variable-Rate Deep Image Compression through Spatially-Adaptive Feature Transform"](https://arxiv.org/abs/2108.09551) (ICCV 2021).  [code](https://github.com/micmic123/QmapCompression)

[2] ["Variable Bitrate Image Compression with Quality Scaling Factors"](https://ieeexplore.ieee.org/abstract/document/9053885/) (ICASSP 2020).  [code](https://github.com/tongxyh/ImageCompression_VariableRate)

[3] ["Variable Rate Deep Image Compression with Modulated Autoencoders"](https://ieeexplore.ieee.org/document/8977394) (IEEE SPL 2020)  [code](https://github.com/FireFYF/modulatedautoencoder)

[4] ["Slimmable Compressive Autoencoders for Practical Neural Image Compression"](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_Slimmable_Compressive_Autoencoders_for_Practical_Neural_Image_Compression_CVPR_2021_paper.html) (CVPR 2021)  [code](https://github.com/FireFYF/SlimCAE)

# Related work
>[Sun Z, Tan Z, Sun X, et al. Interpolation variable rate image compression[C]//Proceedings of the 29th ACM International Conference on Multimedia. 2021: 5574-5582.](https://dl.acm.org/doi/abs/10.1145/3474085.3475698)
>
>This work can be regarded as the feature modulation version of GainVAE and can be easily implemented based on this repo.

# Contact
Feel free to contact me if there is any question about the code or to discuss any problems with image and video compression. (mxh_wine@qq.com)
