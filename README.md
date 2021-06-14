# Class project of computational photography

The implementation and improvement of CVPR 2019 paper "[Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements](https://arxiv.org/abs/1904.00637)"




## Prerequisites
* Python >=3.5, PyTorch >= 0.4.1
* Requirements: opencv-python, tensorboardX, visdom
* Platforms: Ubuntu 16.04, cuda-8.0



## Folder list

* ERRNet-ref 中包含经过我们微调的ERRNet原始代码
* ERRNet-mydata 中包含调用我们自己生成的数据集训练模型的代码（生成数据的代码在sync_model文件夹中，这里直接调用了生成好的数据集）
* ERRNet-TR 中包含我们提出的ERRNet-TR网络模型及其相关的训练/测试代码
* ERRNet-TR-lite 中包含我们提出的ERRNet-TR-lite 网络模型及其相关的训练/测试代码
* sync_model 中包含各种我们自己写的生成合成数据的代码，且内涵一个生成样例用以展示



## Quick Start

#### Testing
 * 进入相应文件夹后用命令行运行 ```python test_errnet.py --name errnet -r --icnn_path checkpoints/errnet/ourmodel.pt --hyper --gpu_ids -1``` (gpu_ids=-1表示用cpu进行测试)

#### Training
* 进入相应文件夹后用命令行运行 ```python train_errnet.py --name errnet --hyper``` （需要预先在相应文件夹中的reflection_data文件夹下面准备好训练数据集）
* 查看 ```options/errnet/train_options.py``` 以获得更多训练选项
