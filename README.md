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
- Tested on Ubuntu 22.04 (with cuda 11.8.0)
```bash
git clone --recurse-submodules -b ros2_w26 git@github.com:UMich-CURLY/ASV_perception.git
cd ~/ASV_perception/src/semantic_mapping/ConvBKI
conda env create -f environment.yaml
conda activate asv_perception
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
cd ../../torchsparse/
git checkout v1.4.0
python setup.py install
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
```

### Build ROS workspace
```bash
cd ~/ASV_perception
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run Mapping Pipeline

This pipeline publishes a semantic map in ROS 2 for visualization in RViz2. A localization source (e.g. DRIFT) is required.

The pipeline has three main components. Open **three separate terminals** and start each module:

### 1. Sensor Frontend (Camera + LiDAR Processing)

Processes raw sensor data and generates inputs for downstream jobs.

Executed scripts: 
1. Image segmentation using YOLO: ```objseg_yolo.py```
2. Point clouds projection and filtering: ```pcdFilter_node.py```

```bash
conda activate asv_perception
cd ~/ASV_perception
source install/setup.bash
ros2 launch asv_perception sensor_process.launch.py
```

---

### 2. Semantic Map Publisher

Publishes the semantic point cloud map for visualization in RViz2.

Executed scripts: 
1. Per-frame semantic point clouds publisher: ```semantic_pcd_publisher.py```
2. Global semantic point cloud publisher: ```semantic_pcd_publisher_global.py```
3. Occupancy grid map publisher: ```ogm_builder.py```

```bash
conda activate asv_perception
cd ~/ASV_perception
source install/setup.bash
ros2 launch asv_perception map_publisher.launch.py
```

**Note:** Visualizing large maps in RViz2 can be computationally expensive.

#### (Optional): Object Localization Visualization

Launches augmented-reality-style object position visualization:

```bash
conda activate asv_perception
cd ~/ASV_perception
source install/setup.bash
python cluster_global_pub_objects.py
```

---

### 3. Object SLAM

Runs object-level SLAM for global object association and map refinement.

```bash
conda activate asv_perception
cd ~/ASV_perception
source install/setup.bash
ros2 launch asv_perception obj_slam.launch.py
```

---

## Provide Sensor Data

After all modules are running, provide sensor data using one of the following:

### Option A: Play a Processed ROS 2 Bag

```bash
ros2 bag play your-bag.db3
```

### Option B: Run VRX Simulation

```bash
ros2 launch vrx_gz competition.launch.py world:=sydney_regatta
```

#### YAML Parameters

Parameters can be configured in:

```bash
ASV_perception/src/configs/params.yaml
```

### Sensor Topics

* `lidar_topic`  
  LiDAR point cloud topic to subscribe to.

* `camera_topic`  
  Camera image topic to subscribe to.

* `caminfo_topic`  
  Camera intrinsic calibration (`CameraInfo`) topic.

### Localization Topics

* `pose_topic`  
  Pose topic used for localization (e.g. DRIFT).

---

### Frame Definitions

* `camera_frame`  
  Camera optical frame name.

* `lidar_frame`  
  LiDAR frame name.

---

* num_classes - number of semantic classes
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
