#ifndef _MYEXPSET
#define _MYEXPSET
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <pthread.h>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

using namespace std;

static int inited_shared_mem = 0;
static boost::interprocess::shared_memory_object shm (boost::interprocess::open_only, "MySharedMemory2", boost::interprocess::read_write);
static boost::interprocess::mapped_region region(shm, boost::interprocess::read_write);
static boost::interprocess::named_mutex named_mtx(boost::interprocess::open_only, "named_mutex2");
static boost::interprocess::named_condition named_cnd{boost::interprocess::open_only, "named_cnd2"};

static volatile int *current_process;

void init_shared_mem_expset() {
    inited_shared_mem = 1;
    int *mem = static_cast<int*>(region.get_address());
    current_process = &mem[0]; 
}

extern "C" void setMem(int input) {
    if (!inited_shared_mem) {
        init_shared_mem_expset();
    }
    if (input == 0) {
        // https://en.cppreference.com/w/cpp/thread/condition_variable
        boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(named_mtx);
        *current_process = input;
        named_cnd.notify_one();
    } else {
        boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(named_mtx);
        *current_process = input;
    }
} 

extern "C" void printCurr()
{
    printf("lib curr%d\n", *current_process);
}

// int main()
// {
//     setMem(1);
// }
#endif