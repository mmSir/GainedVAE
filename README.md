# GainedVAE

A Pytorch Implementation of a continuously rate adjustable learned image compression framework, Gained Variational Autoencoder(GainedVAE).

*Note that It Is Not An Official Implementation Code.*

More details can be found in the following paper:

>[Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation.](https://openaccess.thecvf.com/content/CVPR2021/html/Cui_Asymmetric_Gained_Deep_Image_Compression_With_Continuous_Rate_Adaptation_CVPR_2021_paper.html)  
>Huawei Technologies, CVPR 2021  
>Ze Cui, Jing Wang, Shangyin Gao, Tiansheng Guo, Yihui Feng, Bo Bai
Todo:
Reproduce Implementation of following paper:
>[INTERPOLATION VARIABLE RATE IMAGE COMPRESSION](https://arxiv.org/abs/2109.09280)  
>Alibaba Group, arxiv 2021.9.20  
>Zhenhong Sun, Zhiyu Tan, Xiuyu Sun, Fangyi Zhang, Yichen Qian, Dongyang Li, Hao Li
# Environment

* Python == 3.7.10
* Pytorch == 1.7.1
* CompressAI

# Dataset
I use a part of the OpenImages Dataset to train the models (train06, train07, train08, about 54w images). You can download from here[Download OpenImages](https://www.appen.com.cn/datasets/open-images-annotated-with-bounding-boxes/).
Maybe train08 (14w images) is enough.


# Train Your Own Model
>python3 trainGain.py -d /path/to/your/image/dataset/ --epochs 200 -lr 1e-4 --batch-size 16 --model-save /path/to/your/model/save/dir --cuda
# Result

# Acknowledgement

The framework is based on CompressAI, I add the model in compressai.models.gain, compressai.models.gain_utils.  
And trainGain/trainGain.py is modified with reference to compressai_examples/train.py.

# Contact
Feel free to contact me if there is any question about the code or to discuss any problems with image and video compression. (mxh_wine@qq.com)