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

static std::shared_ptr<boost::interprocess::shared_memory_object> shm_ptr;
static std::shared_ptr<boost::interprocess::mapped_region> region_ptr;
static std::shared_ptr<boost::interprocess::named_mutex> named_mtx_ptr;
static std::shared_ptr<boost::interprocess::named_condition> named_cnd_ptr;

static volatile int *current_process;

extern "C" void create_shared_mem_and_locks(char* suffix) {
    if (suffix == NULL) {
        return;
    }
    std::string suffix_str(suffix);
    if (shm_ptr == NULL) {
        std::string shm_name("MySharedMemory_" + suffix_str);
        shm_ptr = make_shared<boost::interprocess::shared_memory_object>(
            boost::interprocess::create_only, shm_name.c_str(),
            boost::interprocess::read_write);
        shm_ptr->truncate(1000);

        boost::interprocess::mapped_region my_region(
            *shm_ptr, boost::interprocess::read_write);
        std::memset(my_region.get_address(), -1, my_region.get_size());
    }

    if (named_mtx_ptr == NULL) {
        std::string named_mtx_name("named_mutex_" + suffix_str);
        named_mtx_ptr = make_shared<boost::interprocess::named_mutex>(
            boost::interprocess::create_only, named_mtx_name.c_str());
    }

    if (named_cnd_ptr == NULL) {
        std::string named_cnd_name("named_cnd_" + suffix_str);
        named_cnd_ptr = make_shared<boost::interprocess::named_condition>(
            boost::interprocess::create_only, named_cnd_name.c_str());
    }
}

extern "C" void remove_shared_mem_and_locks(char* suffix) {
    if (suffix == NULL) {
        return;
    }
    std::string suffix_str(suffix);
    if (shm_ptr != NULL) {
        boost::interprocess::shared_memory_object::remove(shm_ptr->get_name());
    }
    if (named_mtx_ptr != NULL) {
        std::string named_mtx_name("named_mutex_" + suffix_str);
        boost::interprocess::named_mutex::remove(named_mtx_name.c_str());
    }
    if (named_cnd_ptr != NULL) {
        std::string named_cnd_name("named_cnd_" + suffix_str);
        boost::interprocess::named_condition::remove(named_cnd_name.c_str());
    }
}


extern "C" void setMem(int input, char* suffix) {
    if (suffix == NULL) {
        return;
    }
    std::string suffix_str(suffix);
    if (shm_ptr == NULL || region_ptr == NULL) {
        std::string shm_name("MySharedMemory_" + suffix_str);
        shm_ptr = make_shared<boost::interprocess::shared_memory_object>(
            boost::interprocess::open_only, shm_name.c_str(),
            boost::interprocess::read_write);
        region_ptr = make_shared<boost::interprocess::mapped_region>(
            *shm_ptr, boost::interprocess::read_write);
        int *mem = static_cast<int*>(region_ptr->get_address());
        current_process = &mem[0];
    }
    if (named_mtx_ptr == NULL) {
        std::string named_mtx_name("named_mutex_" + suffix_str);
        named_mtx_ptr = make_shared<boost::interprocess::named_mutex>(
            boost::interprocess::open_only, named_mtx_name.c_str());
    }
    if (named_cnd_ptr == NULL) {
        std::string named_cnd_name("named_cnd_" + suffix_str);
        named_cnd_ptr = make_shared<boost::interprocess::named_condition>(
            boost::interprocess::open_only, named_cnd_name.c_str());
    }
    boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(
        *named_mtx_ptr);
    // https://en.cppreference.com/w/cpp/thread/condition_variable
    *current_process = input;
    if (input == 0) {
        named_cnd_ptr->notify_one();
    }
}

extern "C" void printCurr()
{
    printf("lib curr%d\n", *current_process);
}
#endif
