## To Setup the gpu sharing experiment 
0. You have already setup conda environment and replace transform.py file in torchvision with `gpu-core-exps/transform.py`. The path of the file that should be replaced is under `/home/USERNAME/miniconda3/envs/torch/lib/python3.9/site-packages/torchvision/models/detection/transform.py`

1. Install memcached and its library https://memcached.org/downloads
```bash
wget http://memcached.org/latest
tar -zxvf memcached-1.x.x.tar.gz
cd memcached-1.x.x
./configure && make && make test && sudo make install
sudo apt install libmemcached-dev
```

2. make sure memcached server is run in the back group `netstat -ap | grep 11211` otherwise use `memcached &` to start it in background.

2. Compile `contorller` under `gpu-core-exps/gpu-sched-centralcontrol/centralcontrol`. This module provides initialization script of memcached based central controller. `libgeek.so` will also be built. It is a dynamic library that will be used by PyTorch process to talk to central controller and memcached server.\
`./make.sh`
`./controller` Current controller does not have control logic in it but it supports adding control polic in it. At this time, we run it once and can terminate it just to initalize fields in the memcached server. 

3. Compile hooks library under `gpu-core-exps/gpu-sched-centralcontrol/intercept-lib/`\
`./make.sh`\
This command will create one versions of hooks library: controller with preemption on job 0.

## To Run the GPU sharing Experiment 
Under `gpu-core-exps/gpu-sched-centralcontrol/gpu-tester`, run the experiment with FastRCNN (job 1, inserted job) and DeepLab (job 0, preempted job) sharing the GPU.
`./run_test_nsight.sh`\
Run the experiment with FastRCNN (job 1, inserted job) and DeepLab (job 0, preempted job) sharing the GPU with controller.\
`./run_test_nsight_control.sh`\
Run the experiment with FastRCNN exclusivly on GPU. \
`./run_test_nsight_1.sh`\
Run the experiment with DeepLab exclusivly on GPU. \
`./run_test_nsight_0.sh`\
Run the experiment with FastRCNN with controller exclusivly on GPU. \
`./run_test_nsight_control_1.sh`\
Run the experiment with DeepLab exclusivly on GPU. \
`./run_test_nsight_control_0.sh`\
The JCT output of job 0 is in file `out_0.txt` and the JCT output of job 1 is in file `out_1.txt`.

