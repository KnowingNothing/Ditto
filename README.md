# Ditto
---
Ditto is an automatic framework built on top of deep learning compilers.
It focuses on the front-end.
The three critical modules of Ditto include:
- AutoScheduler
- AutoTensorize
- AutoCompute

Ditto is also designed for compiling for training.

#### Supported Compiler Backends
- TVM

## Install with TVM backend
### 1. Download the source code
```sh
cd ~
git clone git@github.com:KnowingNothing/Ditto.git
```
Then get the submodules
```sh
cd Ditto
git submodule update --init --recursive
```
Considering the https addresses are forbidden by GFW in China, we use ssh addresses instead. This requires you to generate ssh keys locally through `ssh-keygen` and configure your github settings including `user.name` and `user.email`.
The TVM we use is a mirror of official TVM.

### 2. Prepare the config file
```sh
mkdir build
cd build
cp ../cmake/tvm_cmake/config.cmake .
```

If you are not familiar with TVM, please stick to the following steps to configure config.cmake, otherwise, just jump to the cmake step. We recommend you to refer to the documents of TVM (https://tvm.apache.org/docs/install/from_source.html) for details.

```sh
vim config.cmake
```
### 2.1 LLVM settings
Download LLVM source code from https://github.com/llvm/llvm-project to ~/LLVM. You can install LLVM to anywhere you want. Here we choose `~/LLVM/llvm-12`
```sh
mkdir -p ~/LLVM
cd ~/LLVM
git clone git@github.com:llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-12.0.0
mkdir build
cmake -DCMAKE_INSTALL_PREFIX=/home/<your-home-dir>/LLVM/llvm-12 -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld;lldb" ../llvm
make -j 20
make install
```
Then, go back to Ditto directory and modify the config.cmake file.
```sh
cd ~/Ditto/build
vim config.cmake
```
Change the `USE_LLVM` variable to the path to `llvm-config`, i.e., `/home/<your-home-dir>/LLVM/llvm-12/bin/llvm-config` in our example.
### 2.2 CUDA settings
Usually, CUDA toolkit should be installed by the administrater. If you can install CUDA for your own, you can follow the steps of https://developer.nvidia.com/cuda-downloads.
Assume we have CUDA-11.5 installed in `/usr/local/cuda-11.5`.
You can add `/usr/local/cuda-11.5/bin` to your `PATH` so that you have access to `nvcc`.
Then you can further modify the config.cmake file to change `USE_CUDA` variable to value `ON`.
### 2.3 OpenCL settings
To use OpenCL, we can use the OpenCL implementation of Nvidia, which is shipped with CUDA toolkit.
You can simply add `/usr/local/cuda-11.5/lib64` and `/usr/local/cuda-11.5/include` to your `PATH` so that OpenCL libraries can be found.
And modify config.cmake file by changing the value `USE_OPENCL` to `ON`.


### 3. Make and set environments
```sh
cmake ..
make -j 20
```

### 4. Prepare your Python environments
First, we recommend you to use `virualenv` to manage your Python libraries.
If you don't have virtualenv, you can install it locally. If you don't have a pip installed, there are many workarounds, e.g., you can install a Python from source (https://www.python.org/downloads/source/). The details about building Python locally can be found here (https://realpython.com/installing-python/).
```sh
python3 -m pip install --user virtualenv
```
Then use virtualenv to establish your first environment.
```sh
cd ~
mkdir venv
cd venv
python3 -m virtualenv <vir-name> -p python3
```
You can activate your virtual environment by
```sh
source ~/venv/<vir-name>/bin/activate
```
If you find it inconvenient to activate the environment, you can use symbolic link
```sh
mkdir -p .local/bin
cd .local/bin
ln -s /home/<your-home-dir>/venv/<vir-name>/bin/activate <vir-name>
```
Add `/home/<your-home-dir>/.local/bin` to your `PATH` so that you can use a simple `source <vir-name>` to activate your Python environment.
```sh
source <vir-name>
```

Then, install python dependencies of TVM after activating your virtual environment.
```sh
(<vir-name>) pip install numpy decorator attrs tornado psutil xgboost cloudpickle
```

At last, setup the environments.
```sh
(<vir-name>) cd ~/Ditto/envs
(<vir-name>) source tvm_env.sh
(<vir-name>) source ditto_env.sh
```

### Example
To check if you can use Ditto
```sh
(<vir-name>) cd /path/to/Ditto
(<vir-name>) cd python_test/graph
(<vir-name>) python test_graph.py --case 1
```
