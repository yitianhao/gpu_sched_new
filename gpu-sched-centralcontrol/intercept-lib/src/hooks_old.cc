/*
Copyright (c) 2022 Futurewei Technologies.
Author: Hao Xu (@hxhp)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <pthread.h>
#include "hooks.h"
#include "log_trace.h"
#include "nvml.h"
#include "lat_profile.h"
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

// #include <ctime>

using namespace std;
// static named_semaphore sem(open_only, "named_semaphore");

// static named_mutex mutex(open_or_create, "named_mutex");
#ifdef _SCHEDULER_LOCK

static bool inited_shared_mem = false;
static boost::interprocess::shared_memory_object shm (boost::interprocess::open_only, "MySharedMemory3", boost::interprocess::read_only);
static boost::interprocess::mapped_region region(shm, boost::interprocess::read_only);
static boost::interprocess::named_mutex named_mtx(boost::interprocess::open_only, "named_mutex3");
static boost::interprocess::named_condition named_cnd(boost::interprocess::open_only, "named_cnd3");

static volatile int *current_process;


void init_shared_mem() {
    #ifdef _VERBOSE_WENQING
	char *timestamp = printUTCTime();
	PrintThread{} << timestamp << " hook" << get_id() << " init_shared_mem entered" << std::endl;
    free(timestamp);
	#endif
    inited_shared_mem = true;
    int *mem = static_cast<int*>(region.get_address());

    current_process = &mem[0]; 
    #ifdef _VERBOSE_WENQING
	timestamp = printUTCTime();
	PrintThread{} << timestamp << " hook" << get_id() << " init_shared_mem exited" << std::endl;
    free(timestamp);
	#endif
}

#endif

int get_start_time() {
    struct timespec timespc;
    clock_gettime(CLOCK_REALTIME, &timespc);
    float cur = (timespc.tv_sec * 1000000000 + timespc.tv_nsec) / 1000 % 100000000 / 1000.0;
    printf("Starting at %f, id =  %d\n", cur, get_id());
    return 1;
}




static int some_int = get_start_time();

static float last_len = 1;



static void adjust_fill_rate(uint32_t target_usage, uint32_t cur_group_usage)
{
#ifndef _VANILLA
	uint32_t adjusted_group_usage =  cur_group_usage + target_usage;
	// if the difference is larger than half of the target_usage, it will
	// trigger a more intense adjustment
	uint32_t trigger = target_usage / 2 > MAX_DIFF_ALLOWANCE 
		? MAX_DIFF_ALLOWANCE : target_usage / 2;
	if (cur_group_usage < target_usage) {
		uint32_t diff = target_usage - cur_group_usage;
    	uint32_t adjust = ADJUST_ADD_CONST * diff;
		if (diff > trigger)
			adjust = uint32_t(
					ADJUST_ADD_CONST * diff * target_usage);
        tb.fill_rate = tb.fill_rate + adjust > tb.fill_rate_cap 
			? tb.fill_rate_cap : tb.fill_rate + adjust;
	} else {
		if (adjusted_group_usage > target_usage) {
			uint32_t diff = adjusted_group_usage - target_usage;
			if (diff > trigger) tb.cur_tokens = 0;
			tb.fill_rate = 0;
		}
	}
#else
	uint32_t diff = target_usage > cur_group_usage 
		? target_usage - cur_group_usage 
		: cur_group_usage - target_usage;
    uint32_t adjust = ADJUST_ADD_CONST * diff;
    if (diff > target_usage / 2)
        adjust = adjust * diff * 2 / (target_usage + 1);

    if (target_usage > cur_group_usage)
        tb.fill_rate = tb.fill_rate + adjust > tb.fill_rate_cap
			? tb.fill_rate_cap
			: tb.fill_rate + adjust;
    else
        tb.fill_rate = tb.fill_rate > adjust ? tb.fill_rate - adjust : 0;
#endif
	DEBUG_FILL_R(target_usage, cur_group_usage);
}

static void dyn_adjust_fr_sched() 
{
    uint32_t cur_group_usage;

    for (;;) {
		// adjust refill rate
        if (!get_current_group_usage(&cur_group_usage))
			adjust_fill_rate(gpu_compute_limit, cur_group_usage);

		// refill
        pthread_mutex_lock(&tb.mutex);
        tb.cur_tokens = (tb.cur_tokens + tb.fill_rate) > tb.max_burst
			? tb.max_burst : tb.cur_tokens + tb.fill_rate;
        pthread_mutex_unlock(&tb.mutex);        
    }
}


static void kernel_lat_sched()
{
	uint32_t fill_rate = gpu_compute_limit / 100 * tb.sample_period.tv_nsec;
	for (;;) {
        pthread_mutex_lock(&tb.mutex);
        tb.cur_tokens = fill_rate;
        pthread_mutex_unlock(&tb.mutex);        
		nanosleep(&tb.sample_period, NULL);
	}
}

static void *sched_bootstrap(void *args)
{
	// If user doesn't want to constrain the compute usage, no need to run this
	// thread. This will always be true if more than one devices are visible.
    if (gpu_compute_limit == 100) return NULL;
    
	// wait for cuInit hook and posthook finish
	while (!pre_initialized || !post_initialized)
		nanosleep(&tb.sample_period, NULL);

#ifdef _DYN_ADJUST_FR
	dyn_adjust_fr_sched();
#else
	kernel_lat_sched();
#endif

	return NULL;
}

static void pre_cuinit(void)
{
    // init nvml library
    NVML_CHECK(nvmlInit());
    pre_initialized = true;
}

static void post_cuinit(void)
{
    CUresult cu_ret = CUDA_SUCCESS;
    CUdevice dev;
    int num_sm, num_thd_per_sm;

    // No need to continue if user doesn't want to constrain the compute usage.
	if (gpu_compute_limit == 100) {
		post_initialized = true;
		return;
	}

    // Here we only support compute resource sharing within a single device.
    // If multiple devices are visible, gpuComputeLimit would be 100, 
    // and the previous statement would have already exited. 
    CUDA_CHECK(cuDeviceGet(&dev, 0));
    CUDA_CHECK(
		cuDeviceGetAttribute(
			&num_sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    CUDA_CHECK(
		cuDeviceGetAttribute(
			&num_thd_per_sm,
			CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev));

	// fill rate cap is determined by the sample period and gpu clock
	// frequency?
	// max_instruct_per_second = num_sm * num_thd_per_sm * clock_hz
	// max_instruct_per_ns = num_sm * num_thd_per_sm * clock_hz / 1e9
	// max_instruct_per_sample_period_ns =
	// 		num_sm * num_thd_per_sm * clock_hz / 1e9 * sample_period
    // tb.fill_rate_cap = num_sm * num_thd_per_sm * GPU_CLOCK_HZ 
	// 	/ SEC_IN_NS * tb.refill_interval.tv_nsec * gpu_compute_limit / 100;

    tb.fill_rate_cap = num_sm * num_thd_per_sm /
		(tb.sample_period.tv_nsec / 1000000) * GPU_CLOCK_KHZ;
    tb.max_burst = tb.fill_rate_cap;
    tb.cur_tokens = tb.fill_rate_cap;

    // thread to fill the token bucket
    pthread_t sched_thd;
    if (pthread_create(&sched_thd, NULL, sched_bootstrap, NULL) < 0)
		fprintf(stderr,
				"sched thread creation failed, errno=%d\n", errno);
    post_initialized = true;

#ifdef _CHANGE_ALLOC_TEST
	pthread_t change_allocation_thd;
	pthread_create(&change_allocation_thd, NULL, change_allocation, NULL);
#endif

#ifdef _RECORD_UTIL
	pthread_t log_sm_util_thd;
	if(pthread_create(&log_sm_util_thd, NULL, log_sm_util, NULL) < 0)
		fprintf(stderr, "Failed to start log thread\n");
#endif

}

/*************************** hooks functions below ***************************/
CUresult cuInit_hook(uint32_t flags)
{
    CUresult cures = CUDA_SUCCESS;
    int res = 0;
    
    // initialize for GPU memory monitoring
    res = pthread_once(&pre_cuinit_ctrl, pre_cuinit);
    if (res < 0) fprintf(stderr,"pre_cuinit failed, errno=%d\n", errno);
    return cures;
}

CUresult cuInit_posthook(uint32_t flags)
{
    CUresult cures = CUDA_SUCCESS;
    int res = 0;

    // initialize for GPU compute monitoring
    res = pthread_once(&post_cuinit_ctrl, post_cuinit);
    if (res < 0) fprintf(stderr,"post_cuinit failed, errno=%d\n", errno);

    return cures;
}

CUresult cuMemAlloc_hook(CUdeviceptr* dptr, size_t byte_sz)
{
    std::stringstream ss;
    ss << "cuMemAlloc," << byte_sz;
    log_api_call(ss.str().c_str());

    return validate_memory(byte_sz);
}

CUresult cuMemAllocManaged_hook(CUdeviceptr *dptr, size_t byte_sz, uint32_t
		flags)
{
    return validate_memory(byte_sz);
}

CUresult cuMemAllocPitch_hook(CUdeviceptr *dptr, size_t *pPitch, size_t
		WidthInBytes, size_t Height, uint32_t ElementSizeBytes)
{
    
    size_t toAllocate = WidthInBytes * Height / 100 * 101;
    return validate_memory(toAllocate);
}

CUresult cuArrayCreate_hook(
		CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
    size_t height = pAllocateArray->Height;
    size_t toAllocate;

    height = (height == 0) ? 1 : height;
    toAllocate = pAllocateArray->NumChannels * pAllocateArray->Width * height;
    toAllocate *= get_size_of(pAllocateArray->Format);
    return validate_memory(toAllocate);  
}

CUresult cuArray3DCreate_hook(
		CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
{
    size_t depth = pAllocateArray->Depth;
    size_t height = pAllocateArray->Height;
    size_t toAllocate;

    height = (height == 0) ? 1 : height;
    depth = (depth == 0) ? 1 : depth;
    toAllocate =
		pAllocateArray->NumChannels * pAllocateArray->Width * height * depth;
    toAllocate *= get_size_of(pAllocateArray->Format);
    return validate_memory(toAllocate);
}

CUresult cuMipmappedArrayCreate_hook(
		CUmipmappedArray *pHandle, 
		const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, 
		uint32_t numMipmapLevels)
{
    size_t depth = pMipmappedArrayDesc->Depth;
    size_t height = pMipmappedArrayDesc->Height;
    size_t toAllocate;

    height = (height == 0) ? 1 : height;
    depth = (depth == 0) ? 1 : depth;
    toAllocate = pMipmappedArrayDesc->NumChannels * pMipmappedArrayDesc->Width
		* height * depth;
    toAllocate *= get_size_of(pMipmappedArrayDesc->Format);
    return validate_memory(toAllocate);
}

/**
 * Wenqing: Logic executed before intercepted cuLaunchKernel CUDA call.
*/
CUresult cuLaunchKernel_hook(
		CUfunction f, uint32_t gridDimX, uint32_t gridDimY, 
		uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY, 
		uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream, 
		void** kernelParams, void** extra)
{
    CUresult cures = CUDA_SUCCESS;
	uint32_t cost = gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY *
		blockDimZ;
    kernel_launch_time++;
    // printf("%d %d\n", kernel_launch_time, get_id());
#ifdef _VERBOSE_WENQING
	char *timestamp = printUTCTime();
	PrintThread{} << timestamp << " hook" << get_id() << "kernelpre entered" << std::endl;
    free(timestamp);
#endif
#ifdef _SCHEDULER_LOCK
    
    if(!inited_shared_mem) {
        init_shared_mem();
    }
#ifdef _VERBOSE_WENQING
    if (get_id() == 0) {
	    char *timestamp = printUTCTime();
	    PrintThread{} << timestamp << " hook" << get_id() << "kernelpre curr" << *current_process << std::endl;
        free(timestamp);
    }
#endif
    if (get_id() == 0) {
        //job 1 is running, give way.
#ifdef _VERBOSE_WENQING
	    char *timestamp = printUTCTime();
	    PrintThread{} << timestamp << " hook" << get_id() << "before loop" << *current_process << std::endl;
        free(timestamp);
#endif 
        boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(named_mtx); 
        while(*current_process == 1) {
#ifdef _VERBOSE_WENQING
	        char *timestamp = printUTCTime();
	        PrintThread{} << timestamp << " hook" << get_id() << "in loop" << *current_process << std::endl;
            free(timestamp);
#endif 
            named_cnd.wait(lock);
        }  
#ifdef _VERBOSE_WENQING
	    timestamp = printUTCTime();
	    PrintThread{} << timestamp << " hook" << get_id() << "after loop" << *current_process << std::endl;
        free(timestamp);
#endif 
    }
#ifdef _VERBOSE_WENQING
	timestamp = printUTCTime();
	PrintThread{} << timestamp << " hook" << get_id() << "kernelpre exit" << std::endl;
    free(timestamp);
#endif
#endif

#ifdef _KERNEL_COUNT_WENQING
    static float last = 0;
    struct timespec timespc;
    clock_gettime(CLOCK_REALTIME, &timespc);
    float cur = (timespc.tv_sec * 1000000000 + timespc.tv_nsec) / 1000 % 100000000 / 1000.0;
    printf("launching kernel %f %d %d\n", cur, kernel_launch_time, get_id());
    // printf("launching kernel since last %f, ratio = %f, %d %d\n", cur - last, (cur - last) / last_len, kernel_launch_time, get_id());

    last = cur;
#endif

#ifdef _RECORD_LAUNCH_TIMES
	kernel_launch_time++;
	printf("%s\t%d\t%d\n", CGROUP_DIR, kernel_launch_time, cost);
#endif

    if(kernel_launch_time == 1) {
        cudaEventCreate(&cu_dummy);
    }

#ifdef _PROFILE_ELAPSED_TIME
    float elapsed_time_ms = 0;
    if (kernel_launch_time == 1) {
        cudaEventCreate(&cu_global_start);
        for (int i = 0; i < CNT_KERNER_CALL; i++) {
            cudaEventCreate(&cu_start[i]);
            cudaEventCreate(&cu_end[i]);
        }
    }
    
    if((kernel_launch_time - 1) / CNT_KERNER_CALL == 15) {
        if((kernel_launch_time - 1) % CNT_KERNER_CALL == 0) {
            cudaEventRecord(cu_global_start, hStream);
        }
        cudaEventRecord(cu_start[(kernel_launch_time - 1) % CNT_KERNER_CALL], hStream);
    }

    // 

    // if(kernel_launch_time > 1) {
    //     cudaEventSynchronize(cu_start);
    //     cudaEventElapsedTime(&elapsed_time_ms, cu_end, cu_start);
    //     printf("1 %d\t%s\t%d\t%f\n", get_id(), CGROUP_DIR, kernel_launch_time, elapsed_time_ms *
    //             MS_IN_US);
    // }

#endif
#ifdef _KERNEL_TIMESTAMP_WENQING 
	PrintThread{} << getMicrosecUTCTime() << " " << get_id() << " kernelStart" << std::endl;
#endif 
    return cures;
}

/**
 * Wenqing: Logic executed after cuLaunchKernel CUDA call.
*/
CUresult cuLaunchKernel_posthook(
		CUfunction f, uint32_t gridDimX, uint32_t gridDimY, 
		uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY, 
		uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream, 
		void** kernelParams, void** extra)
{
	CUresult ret = CUDA_SUCCESS;
#ifdef _PROFILE_ELAPSED_TIME
	float elapsed_time_ms = 0;

    if(get_id() == 0 && (kernel_launch_time - 1) / CNT_KERNER_CALL == 15) {
        cudaEventRecord(cu_end[(kernel_launch_time - 1) % CNT_KERNER_CALL], hStream);
        pts[(kernel_launch_time - 1) % CNT_KERNER_CALL] = (int*)hStream;
    }
    if(get_id() == 0 && kernel_launch_time == CNT_KERNER_CALL * 20) {
        float last_end = 0;
        for (int i = 0; i <  CNT_KERNER_CALL; i++) {
            float start_time = 0, end_time = 0;
            cudaEventElapsedTime(&start_time, cu_global_start, cu_start[i]);
            cudaEventElapsedTime(&end_time, cu_global_start, cu_end[i]);
            // printf("%d-th kernel: time: (%f, %f), (%f, %f)\n", i, start_time, end_time, end_time - start_time, start_time - last_end);

            printf("%f %f 0x%p\n", start_time, end_time, (void*)pts[i]);
            last_end = end_time;
        }
    }
#endif
#ifdef _SCHEDULER_LOCK
    
#ifdef _VERBOSE_WENQING
	char *timestamp = printUTCTime();
	PrintThread{} << timestamp << " hook" << get_id() << "kernelpost entered" << std::endl;
    free(timestamp);
#endif

#endif
#ifdef _VERBOSE_WENQING
	timestamp = printUTCTime();
	PrintThread{} << timestamp << " hook" << get_id() << "kernelpost exited" << std::endl;
    free(timestamp);
#endif
#ifdef _KERNEL_TIMESTAMP_WENQING 
	PrintThread{} << getMicrosecUTCTime() << " " << get_id() << " kernelEnd" << std::endl;
#endif 
	return ret;
}

CUresult cuLaunchCooperativeKernel_hook(
		CUfunction f, uint32_t gridDimX, uint32_t gridDimY, 
		uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
		uint32_t blockDimZ, uint32_t sharedMemBytes,
		CUstream hStream, void **kernelParams)
{
    return cuLaunchKernel_hook(
			f,
			gridDimX, gridDimY, gridDimZ,
			blockDimX, blockDimY, blockDimZ, 
            sharedMemBytes, hStream, kernelParams, NULL);
}

CUresult cuLaunchCooperativeKernel_posthook(
		CUfunction f, uint32_t gridDimX, uint32_t gridDimY, 
		uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
		uint32_t blockDimZ, uint32_t sharedMemBytes,
		CUstream hStream, void **kernelParams)
{
    return cuLaunchKernel_posthook(
			f,
			gridDimX, gridDimY, gridDimZ,
			blockDimX, blockDimY, blockDimZ, 
            sharedMemBytes, hStream, kernelParams, NULL);
}

CUresult cuDeviceTotalMem_posthook(size_t* bytes, CUdevice dev)
{
    *bytes = gpu_mem_limit;
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo_posthook(size_t* free, size_t* total)
{
    // get process ids within the same container
    std::set<pid_t> pids;
    read_pids(pids);

    // get per process gpu memory usage
    uint32_t procCount = MAXPROC;
    nvmlProcessInfo_t procInfos[MAXPROC];
    size_t totalUsed = 0;
    int ret = get_gpu_compute_processes(&procCount, procInfos);
    if (ret != 0) {
        return CUDA_SUCCESS;
    }

    for (int i = 0; i < procCount; ++i) {
        uint32_t pid = procInfos[i].pid;
        if (pids.find(pid) != pids.end())
			totalUsed += procInfos[i].usedGpuMemory;
    }

    *total = gpu_mem_limit;
    *free = totalUsed > gpu_mem_limit ? 0 : gpu_mem_limit - totalUsed;
    return CUDA_SUCCESS;
}
