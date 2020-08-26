

## Quaternion Equivariant Capsule Networks for 3D Point Clouds
Created by <a href="http://campar.in.tum.de/Main/YongHengZhao" target="_blank">Yongheng Zhao</a>, 
<a href="http://tbirdal.me/" target="_blank">Tolga Birdal</a>, 
<a href="https://scholar.google.de/citations?user=enXCzCgAAAAJ&hl=en" target="_blank">Jan Eric Lenssen</a>, 
<a href="http://www.dei.unipd.it/~emg/" target="_blank">Emanuele Menegatti</a>, 
<a href="https://profiles.stanford.edu/leonidas-guibas" target="_blank">Leonidas Guibas</a>, 
<a href="http://campar.in.tum.de/Main/FedericoTombari" target="_blank">Federico Tombari </a>.

This repository contains the implementation of our [ECCV 2020 paper *Quaternion Equivariant Capsule Networks for 3D Point Clouds*](https://arxiv.org/abs/1912.12098) (QEC-Net). In particular, we release code for training and testing QEC-Net for classification and relative rotation estimation for 3D shapes as well as the pre-trained models for quickly replicating our results. 

For an intuitive explanation of the QEC-Net, please check out [ECCV oral presentation](https://youtu.be/LHh56snwhTA).

For the source code, please visit [this github repository](https://github.com/tolgabirdal/qecnetworks).

![](https://github.com/yongheng1991/qec_net/blob/master/docs/teaser.png )



#### Abstract
We present a 3D capsule module for processing point clouds that is equivariant to 3D rotations and translations, as well as invariant to permutations of the input points. The operator receives a sparse set of local reference frames, computed from an input point cloud and establishes end-to-end transformation equivariance through a novel dynamic routing procedure on quaternions. Further, we theoretically connect dynamic routing between capsules to the well-known Weiszfeld algorithm, a scheme for solving \emph{iterative re-weighted least squares} (IRLS) problems with provable convergence properties. It is shown that such group dynamic routing can be interpreted as robust IRLS rotation averaging on capsule votes, where information is routed based on the final inlier scores. Based on our operator, we build a capsule network that disentangles geometry from pose, paving the way for more informative descriptors and a structured latent space. Our architecture allows joint object classification and orientation estimation without explicit supervision of rotations. We validate our algorithm empirically on common benchmark datasets.

### Citation
If you find our work useful in your research, please consider citing:
		  

		  
		  @article{zhao2019quaternion,
			title={Quaternion Equivariant Capsule Networks for 3D Point Clouds},
			author={Zhao, Yongheng and Birdal, Tolga and Lenssen, Jan Eric and Menegatti, Emanuele and Guibas, Leonidas and Tombari, Federico},
			journal={arXiv preprint arXiv:1912.12098},
			year={2019}
		  },
		  
		  @inproceedings{zhao20193d, 
			author={Zhao, Yongheng and Birdal, Tolga and Deng, Haowen and Tombari, Federico}, 
			booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)}, 
			title={3D Point 
			Capsule Networks}, 
			organizer={IEEE/CVF},
			year={2019}
		  }		  

### Installation

The code is based on PyTorch. It has been tested with Python 3.6+, PyTorch 1.1.0, CUDA 10.0(or higher) on Ubuntu 18.04. We suggest the users to build the environment with anaconda. 
 
Install batch-wise eigenvalue decomposition package:
```bash
  cd models/pytorch-cusolver
  python setup.py install
  cd ../models/pytorch-autograd-solver
  python setup.py install
```
(Be aware of installing pytorch-cusolver before 'pytorch-autograd-solver')


To visualize the training process in PyTorch, consider installing  <a href="https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard" target="_blank">TensorBoard</a>.
```bash
  pip install tensorflow==1.14
```

To visualize the point cloud, consider installing <a href="http://www.open3d.org/docs/getting_started.html" target="_blank">Open3D</a>.

### Datasets

Coming soon...

Generate multiple random samples and downsample:
```bash
  cd my_dataloader
  python gen_downsample_index.py
```



### Pre-trained model

Coming soon...

You can download the pre-trained models <a href="to be done" target="_blank">here</a>.


### Usage


3. Train the classification without rotation augmentation:
```bash
  python train_cls.py --inter_out_channels 128 --num_iterations 3 
```
		

4. Train with siamese architecture with relative rotation loss:
```bash
  python train_cls_sia.py --inter_out_channels 128 --num_iterations 3 
```

5. Test Classification under unseen orientation:
```bash
  python test_cls.py --inter_out_channels 128 --num_iterations 3 
```
		
6. Test rotation estimation with Siamese architecture:

```bash
  python test_rot_sia.py
```
		
		
### License
Our code is released under MIT License (see LICENSE file for details).

### Code Reference 
To do 
add the python cusolver
add Jan's repo
add 3d point cpas repo



### To do
1. Add more detials of the experiment 
2. Add the dataset and pre-trained model with google drive link.
3. Add more experiment
6. Add code reference in Readme.
7. Add more animations
...




