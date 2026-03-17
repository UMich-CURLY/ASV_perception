# Welcome!

Thank you for your interest in Convolutional Bayesian Kernel Inference (ConvBKI).
ConvBKI is an optimized semantic mapping algorithm which combines the best of 
probabilistic mapping and learning-based mapping. ConvBKI was previously presented
at ICRA 2023, which you can read more about [here](https://arxiv.org/abs/2209.10663) or 
explore the code from [here](https://github.com/UMich-CURLY/NeuralBKI). 

In this repository and subsequent paper, we further accelerate ConvBKI and test
on more challenging test cases with perceptually difficult scenarios including
marine surface environment. An example from the VRX simulation is playing below.

![Alt Text](./video.gif)

ConvBKI runs as an end-to-end network which you can test using this repository! To test ConvBKI,
clone the repository and have your pre-processed ROS2 bags ready (or your vrx simulation if feeding the odometry/pose online).

Next, simply navigate to the EndToEnd directory and run 'ros2_node_pt_cloud.py'. Once the 
network is up and running as a ROS2 node, begin playing the ROS2 bag (or the simulation for online feed). Note that you will need
to open RVIZ2 if you want to visualize the results.
We use Grounded-SAM2 for semantic segmentation, which you can find installation instructions on [here](git@github.com:IDEA-Research/Grounded-SAM-2.git).

But we RECOMMEND following the below instructions to install Grounded-SAM2 else there maybe dependency conflicts. We also provide a configuration file to create a conda environment, tested on Ubuntu 22.

For more information, please see the below sections on how we preprocessed poses,
and more information on parameters. 

### Note
This branch provides a ROS2 wrapper (ROS2 Humble) with open-vocabulary semantic segmentation using Grounded-SAM2, for ConvBKI. The primary intention of this work was to do 3D Mapping, though we do provide resources how localization can be supported with this wrapper (if ground_truth odometry is not available).

## Install

### Localization (for pre-processing the poses)
You can ignore the Localization instructions if you already have the pre-processed pose data in a ROS2 bag file, or if you have the ground-truth odometry published online.

See LIO-SAM documentation for software and hardware dependency information.

- If using ROS1 (in which case you'll likely use ros1bridge to talk to ros2_node), use the following commands to download and compile the package.

```
git clone -b ros2_grounded_sam2 git@github.com:spsingh37/BKI_ROS.git
mv ~/BKI_ROS/lio-sam/liorf ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
```

- If using ROS2, use the ros2 branch of- https://github.com/TixiaoShan/LIO-SAM

### Mapping
- Tested on Ubuntu 22.04 (with cuda 11.8.0)
```
git clone -b ros2_grounded_sam2 git@github.com:spsingh37/BKI_ROS.git
cd ~/BKI_ROS/EndToEnd
conda env create -f environment.yaml
conda activate ros2_grounded_sam2
export CUDA_HOME=/usr/local/cuda-11.8/
sudo apt-get install libsparsehash-dev
cd Segmentation/
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd checkpoints
bash download_ckpts.sh
cd ..
cd gdino_checkpoints
bash download_ckpts.sh
cd ../..
git clone git@github.com:mit-han-lab/torchsparse.git
cd torchsparse/
git checkout v1.4.0
python setup.py install
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
```

## Run mapping

You can run the mapping module which will create a ros2 publisher that publish the map and can be visualized on rviz2.

1. Run ros2_node_pt_cloud.py:
```
cd ~/BKI_ROS/EndToEnd
python ros2_node_pt_cloud.py
```
2. If you have odometry (not pose), run this:
```
python odom_to_pose.py
```
3. For publishing global map (needed for object localization; visualizing can be expensize in rviz if large map):
```
python pointcloud_publisher_global.py
```
4. Run object localization (with Augmented reality like object position viewer):
```
python cluster_global_pub_objects.py
```
5. 
(a) Either play processed ros2 bag:
```
ros2 bag play your-bag.db3
```
OR
(b) Run VRX simulation:
```
ros2 launch vrx_gz competition.launch.py world:=sydney_regatta
```

#### YAML Parameters

Parameters can be set in the yaml config file, and it can be found in EndtoEnd/Configs/KITTI.yaml

* pc_topic - the name of the pointcloud topic to subscribe to
* pose_topic - the name of the pose topic to subscribe to
* num_classes - number of semantic classes

* For now, the semantic_classes, their colors, and LiDAR-camera intrinsic-extrinisics are all specified in EndtoEnd/Segmentation/utils.py

* grid_size, min_bound, max_bound, voxel_sizes - parameters for convbki layer
* model_path - saved weights for convbki layer
* f - convbki layer kernel size...if you actually want to change this, pls do so in EndtoEnd/ConvBKI/ConvBKI.py...its the variable 'max_dist' there at line 12

- Not using the following:
* res, cr - parameters for SPVNAS segmentation net
* seg_path - saved weights for SPVNAS segmentation net




## Acknowledgement
We utilize data and code from: 
- [1] [SemanticKITTI](http://www.semantic-kitti.org/)
- [2] [RELLIS-3D](https://arxiv.org/abs/2011.12954)
- [3] [SPVNAS](https://github.com/mit-han-lab/spvnas)
- [4] [LIO-SAM](https://github.com/YJZLuckyBoy/liorf)
- [5] [Semantic MapNet](https://github.com/vincentcartillier/Semantic-MapNet)

## Reference
If you find our work useful in your research work, consider citing [our paper](https://arxiv.org/abs/2209.10663)
```
@ARTICLE{wilson2022convolutional,
  title={Convolutional Bayesian Kernel Inference for 3D Semantic Mapping},
  author={Wilson, Joey and Fu, Yuewei and Zhang, Arthur and Song, Jingyu and Capodieci, Andrew and Jayakumar, Paramsothy and Barton, Kira and Ghaffari, Maani},
  journal={arXiv preprint arXiv:2209.10663},
  year={2022}
}
```
Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping ([PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954837))
```
@ARTICLE{gan2019bayesian,
author={L. {Gan} and R. {Zhang} and J. W. {Grizzle} and R. M. {Eustice} and M. {Ghaffari}},
journal={IEEE Robotics and Automation Letters},
title={Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping},
year={2020},
volume={5},
number={2},
pages={790-797},
keywords={Mapping;semantic scene understanding;range sensing;RGB-D perception},
doi={10.1109/LRA.2020.2965390},
ISSN={2377-3774},
month={April},}

```
