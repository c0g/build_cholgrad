TF_INC=/home/tron/Source/tensorflow
TF_INC2=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
SRC_CPU_O=triangle.so
SRC_CPU_S=/home/tron/Source/tensorflow/tensorflow/core/user_ops/triangle.cc
SRC_GPU_O=triangle.cu.so
SRC_GPU_S=/home/tron/Source/tensorflow/tensorflow/core/user_ops/triangle.cu.cc


nvcc -std=c++11 -c -o $SRC_GPU_O $SRC_GPU_S \
    -I $TF_INC2 -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o $SRC_CPU_O $SRC_CPU_S $SRC_GPU_O \
    -I $TF_INC2 -I $TF_INC -fPIC -L/usr/local/cuda-7.5/lib64 -lcudart
