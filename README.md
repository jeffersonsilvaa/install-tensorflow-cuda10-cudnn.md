# Installing Tensorflow with CUDA 10.0 and cuDNN 7.4.1 on Ubuntu 16.04

## Preinstallation
```
$ sudo apt-get update && sudo apt-get upgrade
$ sudo apt install python3-dev python3-pip
$ sudo apt-get install python3-venv
```
## Step 1: Updated Driver Nvidia

Before installing the driver, it is important to check the compatibility of the chosen driver with CUDA.

- Compatibility Driver Nvidia with CUDA: [link](https://docs.nvidia.com/deploy/cuda-compatibility/)

### Selecting and downloading Driver Nvidia:

#### Mode I (Recommended):
[Download](https://www.nvidia.com/download/find.aspx) the driver version available for your GPU.
```
$ cd ~/Downloads
$ sudo sh NVIDIA-Linux-x86_64-'version'.run        | e.g. sudo sh NVIDIA-Linux-x86_64-418.81.run
```
#### Mode II:
```
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
```
-> Open the 'drivers' application and go to 'additional drivers' to choose the version of the driver to install.

### Installation Verification:
```
$ nvidia-smi
```
If appears something like:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.78       Driver Version: 410.78       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 2060     Off  | 00000000:01:00.0  On |                  N/A|
|  0%   38C    P8     9W / 130W |    757MiB /  5911MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1125      G   /usr/lib/xorg/Xorg                           130MiB |
|    0      2071      G   compiz                                        84MiB |
|    0      2348      G   ...quest-channel-token=8366946694074305377   540MiB |
+-----------------------------------------------------------------------------+
```
The driver was installed.

## Step 2: Installation CUDA 10.0

[Download](https://developer.nvidia.com/cuda-toolkit-archive) CUDA version.

Selecting -> Operating System: Linux | Architecture: x86_64 | Distribution: Ubuntu | Version: 16.04

### Selecting and downloading CUDA 10.0 for Ubuntu 16.04:

#### Mode I (Recommended):
Download archive [runfile(local)]
```
$ cd ~/Downloads
$ sudo sh cuda_10.0.130_410.48_linux.run
```
Installation Setup Tips:

- ctrl+c (Skip the contract)
- No  (For installing the CUDA 10.0 built-in 410.48 driver)
- Yes (For the other questions)

#### Mode II:
Download archive [deb(local)]
```
$ cd ~/Downloads
$ sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
$ sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get install cuda
```
### Export the PATH:

#### Mode I (Recommended):
```
$ echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
$ echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64' >> ~/.bashrc

$ source ~/.bashrc
$ sudo ldconfig
```
#### Mode II:
```
$ nano ~/.bashrc
$ export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
(ctrl+o) + (ctrl+x)

$ source ~/.bashrc
$ sudo ldconfig
```

### Installation Verification:
```
$ cat /proc/driver/nvidia/version
$ nvcc -V	| or (nvcc --version)
```
If appears something like:
```
NVRM version: NVIDIA UNIX x86_64 Kernel Module  410.78  Sat Nov 10 22:09:04 CST 2018
GCC version:  gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.12)

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```
CUDA has been successfully installed!

## Step 3: Install cuDNN

### Download cuDNN v7.4.1, for CUDA 10.0:

Before installing cuDNN, it is important to verify compatibility with your version of CUDA.
- Compatibility cuDNN with cuda: [link](https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html)

Download the cuDNN [cuDNN Library for Linux] at the NVIDIA official website: [link](https://developer.nvidia.com/rdp/cudnn-archive)
```
$ cd ~/Downloads
$ tar -xzvf cudnn-10.0-linux-x64-v7.4.1.5.tgz
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```
### Installation Verification:
```
$ grep CUDNN_MAJOR -A 2 /usr/local/cuda/include/cudnn.h
```
If appears something like:
```
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 4
#define CUDNN_PATCHLEVEL 1
--
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#include "driver_types.h"
```
cuDNN library has been successfully installed!

## Step 3: Install Tensorflow-gpu for CUDA 10.0

- Compatibility tensorflow-gpu with cuda: [link](https://www.tensorflow.org/install/source#tested_build_configurations)

### Creating virtual environments and requirements:
```
$ cd ~
$ python3 -m venv envTF
$ source envTF/bin/activate

$ pip install --upgrade pip
$ pip install scipy matplotlib pillow
$ pip install imutils h5py requests progressbar2
$ pip install scikit-learn scikit-image
$ pip install setuptools==41.0.0
$ pip install numpy==1.16.4
```

### Installation Tensorflow-gpu v1.13.1:
```
$ pip install tensorflow-gpu==1.13.1
```
### CAREFUL! If your GPU has Turing architecture, add the code below to your training algorithm:
```
-----------------------------------------------------------------------------
import tensorflow as tf

config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.keras.backend.set_session(session)
-----------------------------------------------------------------------------
```

Everything is ready for you use the GPU to make great things!
