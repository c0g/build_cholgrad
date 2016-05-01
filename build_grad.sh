TF_INC=/home/tron/Source/tensorflow
TF_INC2=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
NV_INC=/usr/local/cuda-7.5/include
NV_LIB=/usr/local/cuda-7.5/lib64
SRC_GPU_O=gpu_cholgrad.cu.so
SRC_GPU_S=/home/tron/Source/tensorflow/tensorflow/core/user_ops/gpu_cholgrad.cu.cc
SRC_CPU_O=gpu_cholgrad.so
SRC_CPU_S=/home/tron/Source/tensorflow/tensorflow/core/user_ops/gpu_cholgrad.cc

nvcc -std=c++11 -c -o $SRC_GPU_O $SRC_GPU_S\
    -D GOOGLE_CUDA=1 -I $TF_INC2 -I $TF_INC -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o $SRC_CPU_O $SRC_CPU_S $SRC_GPU_O -D GOOGLE_CUDA=1\
    -I $TF_INC2 -I $TF_INC -I$NV_INC -fPIC -L$NV_LIB -lcudart
