# Class project of computational photography

The implementation and improvement of CVPR 2019 paper "[Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements](https://arxiv.org/abs/1904.00637)"




## Prerequisites
* Python >=3.5, PyTorch >= 0.4.1
* Requirements: opencv-python, tensorboardX, visdom
* Platforms: Ubuntu 16.04, cuda-8.0



## Folder list

* `ERRNet-ref` folder contains the debugged original code of ERRNet 
* `ERRNet-mydata` folder contains the code for training using our own synthetic data (Code for data generating is in `sync_model` folder）
* `ERRNet-TR` folder contains our newly proposed network: ERRNet-TR and the relating training and test code
* `ERRNet-TR-lite`  folder contains our newly proposed network: ERRNet-TR-lite and the relating training and test code
* `sync_model` folder contains code for data generating and a data sample



## Quick Start

#### Testing
 * Running the following command after entering the corresponding folder ```python test_errnet.py --name errnet -r --icnn_path checkpoints/errnet/ourmodel.pt --hyper --gpu_ids -1``` (gpu_ids=-1 means using CPU to test)

#### Training
* Running the following command after entering the corresponding folder ```python train_errnet.py --name errnet --hyper``` （Need to prepare the dataset in the folder `reflection_data` before training）
* Viewing ```options/errnet/train_options.py``` for more training options
