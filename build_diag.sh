TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
ME_INC=/home/tron/Source/tensorflow
SRC_CPU_O=get_diag.so
SRC_CPU_S=/home/tron/Source/tensorflow/tensorflow/core/user_ops/get_diag.cc
SRC_GPU_O=get_diag.cu.so
SRC_GPU_S=/home/tron/Source/tensorflow/tensorflow/core/user_ops/get_diag.cu.cc


nvcc -std=c++11 -c -o $SRC_GPU_O $SRC_GPU_S \
    -I $TF_INC -I $ME_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++  -std=c++11 -shared -o $SRC_CPU_O $SRC_CPU_S \
    $SRC_GPU_O -I $TF_INC -I $ME_INC -fPIC -L/usr/local/cuda-7.5/lib64 -lcudart
