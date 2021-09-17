# Ditto
---
Ditto is an automatic framework built on top of deep learning compilers.
The three critical modules of Ditto include:
- AutoScheduler
- AutoTensorize
- AutoCompute

Besides, Ditto is also designed for compiling for training.

#### Supported Compiler Backends
- TVM

## Install
```sh
git clone git@github.com:KnowingNothing/Ditto.git
```
Then get the submodules
```sh
cd Ditto
git submodule update --init --recursive
```
Considering the https addresses are forbidden by GFW in China, we use ssh addresses instead. This requires you to generate ssh keys locally through `ssh-keygen` and configure your github settings including `user.name` and `user.email`.
The TVM we use is a mirror of official TVM.

```sh
mkdir build
cd build
cp ../cmake/tvm_cmake/config.cmake .
```
Copy the cmake configure file into build directory and modify it according to the requirements of TVM (refer to https://tvm.apache.org/docs/install/from_source.html).

```sh
cmake ..
make -j 32
```

After building, you may need to install python dependencies of TVM.
```sh
pip install numpy decorator attrs tornado psutil xgboost cloudpickle
```

At last, setup the environments.
```sh
cd ../envs
source tvm_env.sh
source ditto_env.sh
```

### Example
To check if you can use Ditto
```sh
cd /path/to/Ditto
cd python_test/graph
python test_graph.py --case 1
```