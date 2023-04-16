#include <csignal>
#include <iostream>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
using namespace std;
using namespace boost::interprocess;

static int *current_process;
static string username(getenv("USER"));
static string shm_name("MySharedMemory_" + username);
static string named_mtx_name("named_mutex_" + username);
static string named_cnd_name("named_cnd_" + username);

volatile int running = 1;

void signal_handler(int signum) {
    running = 0;
}

int main()
{
    signal(SIGINT, signal_handler);
    struct shm_remove
    {
        shm_remove() { shared_memory_object::remove(shm_name.c_str()); }
        ~shm_remove() { shared_memory_object::remove(shm_name.c_str()); }
    } remover;
    struct mutex_remove
    {
        mutex_remove() { named_mutex::remove(named_mtx_name.c_str()); }
        ~mutex_remove() { named_mutex::remove(named_mtx_name.c_str()); }
    } remover2;
    struct condition_remove
    {
        condition_remove() { named_condition::remove(named_cnd_name.c_str()); }
        ~condition_remove() { named_condition::remove(named_cnd_name.c_str()); }
    } cond_remover;
    shared_memory_object shm(create_only, shm_name.c_str(), read_write);
    named_mutex named_mtx(open_or_create, named_mtx_name.c_str());
    named_condition named_cond(open_or_create, named_cnd_name.c_str());
    shm.truncate(1000);

    // Map the whole shared memory in this process
    mapped_region region(shm, read_write);

    // Write all the memory to 0
    std::memset(region.get_address(), 0, region.get_size());

    int *mem = static_cast<int*>(region.get_address());
    current_process = &mem[0];
    *current_process = -1;
    printf("init curr: %d\n", *current_process);

    while (running)
    {
        // do nothing. Keep shared memory alive.
        if (*current_process == 1) {
            printf("curr%d\n", *current_process);
        }
    }

    return 0;
}
