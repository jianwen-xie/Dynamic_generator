# Learning Dynamic Generator Model by Alternating Back-Propagation Through Time

This repository contains a tensorflow implementation for the paper "[Learning Dynamic Generator Model by Alternating Back-Propagation Through Time](http://www.stat.ucla.edu/~jxie/DynamicGenerator/DynamicGenerator_file/doc/DynamicGenerator.pdf)".

Project Page: (http://www.stat.ucla.edu/~jxie/DynamicGenerator/DynamicGenerator.html)

## Reference
    @article{DG,
        author = {Xie, Jianwen and Gao, Ruiqi and Zheng, Zilong and Zhu, Song-Chun and Wu, Ying Nian},
        title = {Learning Dynamic Generator Model by Alternating Back-Propagation Through Time},
        journal={The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI)},
        year = {2019}
    }
  
 ## Requirements
- Python 2.7 or Python 3.3+
- [Tensorflow r1.0+](https://www.tensorflow.org/install/)
- [Scipy](https://www.scipy.org/install.html)
- [pillow](https://pillow.readthedocs.io/en/latest/installation.html)

## Usage

### (1) For dynamic texture synthesis

(i) Training

First, prepare your data into a folder, for example `./data/scene/rock` 
  
To train a model with ***rock*** dataset:

    $ python main.py --category rock --data_dir ./data/scene --output_dir ./output --net_type scene --image_size 64

The synthesized results will be saved in `./output/rock/synthesis`. 

The learned models will be saved in `./output/rock/checkpoints`. 

If you want to calculate inception score, use --calculate_inception=True. 

(ii) Testing for image synthesis  
    
    $ python main.py --test --test_type syn --category rock --net_type scene --image_size 64 --output_dir ./output --ckpt ./output/rock/checkpoints/model.ckpt-82000

testing results will be saved in `./output/rock/test/synthesis`

  
