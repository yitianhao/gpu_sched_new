#ifndef _HOOKS_H
#define _HOOKS_H
#include "cuda_runtime_api.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <iterator>
#include <ratio>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <unistd.h>
#include <pthread.h>
#include <nvml.h>
#include <cuda.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
using namespace boost::interprocess;

//Added by Wenqing
#include <chrono>
#include <mutex>
#include <sstream>
#include <sys/time.h>

// unit definition
#define KB 			(size_t)1024
#define MB 			KB * KB
#define SEC_IN_MS 	1000L
#define SEC_IN_US 	1000L * SEC_IN_MS
#define SEC_IN_NS 	1000L * SEC_IN_US
#define US_IN_NS 	1000L
#define MS_IN_NS 	US_IN_NS * 1000L
#define MS_IN_US 	1000L
#define MAXPROC 	102400

// options
#define KERNEL_WAIT_PER_NS 10 * MS_IN_NS
#define CGROUP_DIR getenv("CGROUP_DIR")
#define UTIL_LOG_PATH getenv("UTIL_LOG_PATH")
#define GPU_CLOCK_KHZ 1400000

//Wenqing: Option for Synchronization per kernels
#ifdef _SYNC_QUEUE
#define SYNC_KERNELS stoi(getenv("SYNC_KERNELS"))
#endif
// options for fill rate adjustment
#define SAMPLE_PERIOD_NS 50 * MS_IN_NS
#define ADJUST_ADD_CONST 50000
#define MAX_DIFF_ALLOWANCE 10

// function macro for checking and debugging
#define DEBUG_FILL_R(target_usage, cur_group_usage)
#ifdef _DEBUG_FILL_R
#undef DEBUG_FILL_R
#define DEBUG_FILL_R(target_usage, cur_group_usage) 						\
	debug_fill_rate(target_usage, cur_group_usage)
#endif

#define CUDA_CHECK(ret) cuda_assert((ret), false, __FILE__, __LINE__); 			
#define CUDA_CHECK_DONT_ABORT(ret)	 										\
	cuda_assert((ret), true, __FILE__, __LINE__);

#define NVML_CHECK(ret) nvml_assert((ret), false, __FILE__, __LINE__);				
#define NVML_CHECK_DONT_ABORT(ret)  										\
	nvml_assert((ret), true, __FILE__, __LINE__);

// prototypes used for static variable definition
static std::string get_cgroup();
static size_t get_memory_limit();
static int get_compute_limit();
static void *flush_sm_log(void *args);

// structure
struct token_bucket {
    pthread_mutex_t mutex;
    uint64_t cur_tokens;
    uint64_t fill_rate;
    uint64_t fill_rate_cap;
    uint64_t max_burst;
    struct timespec sample_period;
};
struct flush_sm_log_args {
	uint32_t num_proc;
	nvmlProcessUtilizationSample_t *samples;
	uint64_t offset_us;
	uint64_t last_seen_us;
};

// static variables
static std::string cgroup = get_cgroup();
static volatile size_t gpu_mem_limit = get_memory_limit();
static volatile int gpu_compute_limit = get_compute_limit();
static pthread_once_t pre_cuinit_ctrl = PTHREAD_ONCE_INIT;
static pthread_once_t post_cuinit_ctrl = PTHREAD_ONCE_INIT;
static volatile bool pre_initialized = false;
static volatile bool post_initialized = false;
static uint32_t next_log = 100; 		// log frequency
static pthread_mutex_t *flush_mutex =
	(pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
static timespec kernel_wait_period = {
    .tv_sec = 	0,
    .tv_nsec = 	KERNEL_WAIT_PER_NS,
};
static struct token_bucket tb = {
    .mutex = 			PTHREAD_MUTEX_INITIALIZER,
    .cur_tokens = 		0,
    .fill_rate = 		0,
    .fill_rate_cap = 	0,
    .max_burst = 		0,
    .sample_period = 	{
        .tv_sec = 	0,
        .tv_nsec = 	SAMPLE_PERIOD_NS,
    },
};
static int32_t kernel_launch_time = 0; 		// monitor the progress of the
											// infernece

// conditional define variables for debugging and profiling

#ifdef _PROFILE_ELAPSED_TIME
// const static int CNT_KERNER_CALL = 914;
// const static int CNT_KERNER_CALL = 183;
// const static int CNT_KERNER_CALL = 1461;
const static int CNT_KERNER_CALL = 454;
static cudaEvent_t cu_start[CNT_KERNER_CALL], cu_end[CNT_KERNER_CALL];
static int* pts[CNT_KERNER_CALL];
static cudaEvent_t cu_global_start;
#endif

#ifdef _VERBOSE_WENQING
static char *timestamp;
#endif

#ifdef _GROUP_EVENT
const static int EVENT_POOL_SIZE  = 100;
static cudaEvent_t cu_event_cycle[EVENT_POOL_SIZE];
static int cur_event_idx = 0;
static int queue_group_size = std::stoi(getenv("EVENT_GROUP_SIZE"));
#endif

static cudaEvent_t cu_dummy;


static int ID = -1;

/**
 * Get the environment variable set before an instance of tester.py is called. The ID is used
 * to distinguish kernel launch functions from different python process with different ID.
 * NOTE: Each tester.py python process is in a separated environment, so their cuda calls 
 * intercepted by different instances of hooks in their process. 
*/
int get_id() {
	if(ID == -1) {
		// printf("%s\n", getenv("ID"));
		ID = getenv("ID")[0] - '0';
	}
	// printf("get_id %d\n", ID);
	return ID;
}



/**********************     Added by wenqing for timestamps logging Start ******************/
/** Thread safe cout class
  * Exemple of use:
  *    PrintThread{} << "Hello world!" << std::endl;
  * https://stackoverflow.com/questions/14718124/how-to-easily-make-stdcout-thread-safe
  */
class PrintThread: public std::ostringstream
{
public:
    PrintThread() = default;

    ~PrintThread()
    {
        std::lock_guard<std::mutex> guard(_mutexPrint);
        std::cout << this->str();
    }

private:
    static std::mutex _mutexPrint;
};
std::mutex PrintThread::_mutexPrint{};

// Added by Wenqing for debugging
// Put UTC timestamp to std::cout stream.
auto printUTCTime() {
	char buffer[26];
  	int millisec;
  	struct tm* tm_info;
  	struct timeval tv;

  	gettimeofday(&tv, NULL);

  	millisec = lrint(tv.tv_usec/1000.0); // Round to nearest millisec
  	if (millisec>=1000) { // Allow for rounding up to nearest second
    	millisec -=1000;
    	tv.tv_sec++;
  	}

  	tm_info = localtime(&tv.tv_sec);
  	strftime(buffer, 10, "%H:%M:%S", tm_info);
	char *result = (char *)malloc(sizeof(char) * 64);
	sprintf(result, "%s.%03d", buffer, millisec);
	return result;
}

/**
 * Added by Wenqing, a c++ 11 compatible milisec level utc timestamp helper function.
 * Reference: https://stackoverflow.com/questions/19555121/how-to-get-current-timestamp-in-milliseconds-since-1970-just-the-way-java-gets
*/
uint64_t timeSinceEpochMicrosec() {
  using namespace std::chrono;
  return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

std::string getMicrosecUTCTime() {
	char buffer[26];
  	uint64_t microsec = timeSinceEpochMicrosec();
	return std::to_string(microsec);
}

/**********************     Added by wenqing for timestamps logging End ******************/

static std::string get_cgroup() 
{

	#ifdef _VERBOSE_WENQING
	char *timestamp = printUTCTime();
	PrintThread{} << timestamp << " hook" << get_id() << " get_cgroup entered" << std::endl;
	free(timestamp);
	#endif
    std::ifstream fs("/proc/self/cgroup");
    for (std::string line; std::getline(fs, line); ) {
        std::stringstream ss(line);
        std::string item;
        while(std::getline(ss, item, ':')) {
            if(item == "memory") {
                std::getline(ss, item, ':');
                return item;
            }
        }
    }
    fs.close();
	#ifdef _VERBOSE_WENQING
	timestamp = printUTCTime();
	PrintThread{} << timestamp << " hook" << get_id() << " get_cgroup exited" << std::endl;
	free(timestamp);
	#endif
    return "";
}

// directly return the gpu memory capacity as, for now, we do not limit the
// memory usage.
static size_t get_memory_limit() { return 24220 * MB; }

// setting the compute limitation by setting the env var
// ALNAIR_VGPU_COMPUTE_PERCENTILE 
static int get_compute_limit()
{
	#ifdef _VERBOSE_WENQING
	char *timestamp = printUTCTime();
	PrintThread{} << timestamp << " hook" << get_id() << " get_compute_limit entered" << std::endl;
	free(timestamp);
	#endif
    char *var = NULL;
    var = getenv("ALNAIR_VGPU_COMPUTE_PERCENTILE");
	#ifdef _VERBOSE_WENQING
	timestamp = printUTCTime();
	PrintThread{} << timestamp << " hook" << get_id() << " get_compute_limit exited" << std::endl;
	free(timestamp);
	#endif	
    if (!var) {
        return 100;
    } else {
        uint32_t ret = atoi(var);
        if(ret <= 0 || ret > 100) return 100;
        return ret;
    }
}

inline void cuda_assert(
	CUresult code, bool suppress, const char *file, int line)
{
	if (code != CUDA_SUCCESS) {
		const char **err_str_p = NULL;
		const char **err_name_p = NULL;
		if (!suppress) {
			cuGetErrorString(code, err_str_p);
			cuGetErrorName(code, err_name_p);
			fprintf(stderr,"%s:%d: %s: %s\n",
					file, line, *err_name_p, *err_str_p);
			exit(code);
		}
	}
}

inline void nvml_assert(
	nvmlReturn_t code, bool suppress, const char *file, int line)
{
	if (code != NVML_SUCCESS) {
		if (!suppress) {
			fprintf(stderr,"%s:%d: %s\n", file, line, nvmlErrorString(code));
			exit(code);
		}
	}
}

static void read_pids(std::set<pid_t> &pids)
{
	char* cgroup_procs = (char *)malloc(sizeof(char) * 100);
	snprintf(cgroup_procs, 100, "%s/cgroup.procs", CGROUP_DIR);
    std::ifstream fs(cgroup_procs);
    for(std::string line; std::getline(fs, line); )
        pids.insert(atoi(line.c_str()));
    fs.close();
	free(cgroup_procs);
}

static void debug_fill_rate(uint32_t target_usage, uint32_t cur_group_usage)
{
	static bool is_start_ts_init = false;
	static uint64_t start_ns;
	struct timespec ts;
	if (!is_start_ts_init) {
		struct timespec start;
		clock_gettime(CLOCK_MONOTONIC, &start);
		start_ns = start.tv_sec * SEC_IN_NS + start.tv_nsec;
		is_start_ts_init = true;
	}
	if (--next_log == 0 && !clock_gettime(CLOCK_REALTIME, &ts)) {
		printf(
			"%s %lu %d %d %lu %lu\n",
			CGROUP_DIR, (ts.tv_sec * SEC_IN_NS + ts.tv_nsec),
			target_usage, cur_group_usage, tb.fill_rate, tb.cur_tokens);
		next_log = 50;
	}
}

static int get_gpu_compute_processes(
		uint32_t *proc_count, nvmlProcessInfo_t *proc_infos) 
{
    nvmlReturn_t ret;
    nvmlDevice_t device;
    NVML_CHECK(
		nvmlDeviceGetHandleByIndex(0, &device));
    NVML_CHECK(
		nvmlDeviceGetComputeRunningProcesses(device, proc_count, proc_infos));
    return 0;
}

static int get_current_mem_usage(size_t* total_usage)
{
    // get process ids within the same container by cgroups
    std::set<pid_t> pids;
    read_pids(pids);

    // get per process gpu memory usage
    uint32_t num_proc = MAXPROC;
    nvmlProcessInfo_t proc_infos[MAXPROC];
    uint32_t ret = get_gpu_compute_processes(&num_proc, proc_infos);
    if (ret != 0) return ret;

    *total_usage = 0;
    for (uint32_t i = 0; i < num_proc; ++i) {
        uint32_t pid = proc_infos[i].pid;
        if (pids.find(pid) != pids.end())
			(*total_usage) += proc_infos[i].usedGpuMemory;
    }
    return 0;
}

static CUresult validate_memory(size_t to_allocate)
{
    CUresult cu_res = CUDA_SUCCESS;
    size_t totalUsed = 0;

    // TODO handle race condition
    if (totalUsed + to_allocate > gpu_mem_limit)
		return CUDA_ERROR_OUT_OF_MEMORY;

	return cu_res;
}

static size_t get_size_of(CUarray_format fmt)
{
    size_t byte_sz = 1;
    switch (fmt) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
    case CU_AD_FORMAT_SIGNED_INT8:
    case CU_AD_FORMAT_NV12:
        byte_sz = 1;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
    case CU_AD_FORMAT_SIGNED_INT16:
    case CU_AD_FORMAT_HALF:
        byte_sz = 2;
        break;
    case CU_AD_FORMAT_UNSIGNED_INT32:
    case CU_AD_FORMAT_SIGNED_INT32:
    case CU_AD_FORMAT_FLOAT:
        byte_sz = 4;
        break;
    }
    return byte_sz;
}

static int get_current_group_usage(uint32_t *group_usage)
{
    nvmlReturn_t ret;
    nvmlDevice_t device;
    nvmlProcessUtilizationSample_t sample[MAXPROC];
    uint32_t num_proc = MAXPROC;
    struct timespec now;
    size_t sample_start_ns;
	size_t now_ns;

	// read the pids in current cgroup
    std::set<pid_t> pids;
    read_pids(pids);

	//limits: only assume this container mount 1 GPU
    NVML_CHECK(
		nvmlDeviceGetHandleByIndex(0, &device));
	clock_gettime(CLOCK_REALTIME, &now);
	now_ns = now.tv_sec * SEC_IN_NS + now.tv_nsec;
	sample_start_ns = (now_ns - tb.sample_period.tv_nsec) / US_IN_NS;
	NVML_CHECK_DONT_ABORT(
		nvmlDeviceGetProcessUtilization(
			device, sample, &num_proc, sample_start_ns));
    *group_usage = 0;
	// sum up the gpu usage in current cgroup pids list
    for (int i = 0; i < num_proc; ++i) {
        uint32_t pid = sample[i].pid;
		if (pids.find(pid) != pids.end()) *group_usage += sample[i].smUtil;
    }

    return 0;
}

static void *change_allocation(void *arg)
{
	sleep(15);
	gpu_compute_limit = 100 - gpu_compute_limit;
	return NULL;
}

static void* log_sm_util(void *args)
{
	uint32_t usage;
	nvmlDevice_t device;
	std::set<pid_t> pids;
	nvmlProcessUtilizationSample_t sample[MAXPROC];
	struct timespec offset_ts, last_seen_ts;
	struct timespec sample_period_ts = {
		.tv_sec = 0,
		.tv_nsec = 100 * MS_IN_NS,
	};
	uint64_t offset_us, last_seen_us;
	uint64_t sample_period_ns = US_IN_NS;
	uint32_t num_proc = MAXPROC;

	NVML_CHECK_DONT_ABORT(
		nvmlDeviceGetHandleByIndex(0, &device));

	pthread_mutex_init(flush_mutex, NULL);
	clock_gettime(CLOCK_REALTIME, &offset_ts);
	offset_us = (offset_ts.tv_nsec + offset_ts.tv_sec * SEC_IN_NS) / US_IN_NS;
	last_seen_us = offset_us;
	for (;;) {
		pthread_t flush_thd;
		struct flush_sm_log_args args;
		uint64_t last_seen_us_used = last_seen_us;

		NVML_CHECK_DONT_ABORT(
			nvmlDeviceGetProcessUtilization(
				device, sample, &num_proc, last_seen_us));
		clock_gettime(CLOCK_REALTIME, &last_seen_ts);
		last_seen_us = (last_seen_ts.tv_sec * SEC_IN_NS + last_seen_ts.tv_nsec)
			/ US_IN_NS;

		args = {
			.num_proc 		= num_proc,
			.samples 		= sample,
			.offset_us 		= offset_us,
			.last_seen_us 	= last_seen_us_used
		};
		pthread_create(&flush_thd, NULL, flush_sm_log, (void *)&args);

		nanosleep(&sample_period_ts, NULL);
	}
	return NULL;
}

static void *flush_sm_log(void *args)
{
	flush_sm_log_args *casted_args = (struct flush_sm_log_args *)args;
	nvmlProcessUtilizationSample_t *samples = casted_args->samples; 
	uint32_t nproc = casted_args->num_proc;
	uint64_t last_seen_us = casted_args->last_seen_us;
	uint64_t offset_us = casted_args->offset_us;

	uint32_t usage = 0;
	uint32_t index = 0;
	std::set<pid_t> pids;
	FILE *log_fd = fopen(UTIL_LOG_PATH, "a+");
	if (log_fd == NULL)
		perror("Failed to open log file: ");

	read_pids(pids);
	for (int i = 0; i < nproc; ++i)
		if (pids.find(samples[i].pid) != pids.end()) {
			usage += samples[i].smUtil;
			index = i;
		}
	pthread_mutex_lock(flush_mutex);
	fprintf(log_fd, "%s %lu %lu %llu %d\n",
			CGROUP_DIR, last_seen_us - offset_us, last_seen_us,
			samples[index].timeStamp, usage);
	fclose(log_fd);
	pthread_mutex_unlock(flush_mutex);
	return NULL;
}

#endif
