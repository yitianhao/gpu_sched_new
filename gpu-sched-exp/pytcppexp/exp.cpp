#include <iostream>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
using namespace std;
using namespace boost::interprocess; 

static int *current_process;

int main()
{
     struct shm_remove
    {
        shm_remove() { shared_memory_object::remove("MySharedMemory3"); }
        ~shm_remove(){ shared_memory_object::remove("MySharedMemory3"); }
    } remover;
    struct mutex_remove
    {
        mutex_remove() { named_mutex::remove("named_mutex3"); }
        ~mutex_remove(){ named_mutex::remove("named_mutex3"); }
    } remover2;
     struct condition_remove
    {
        condition_remove() { named_condition::remove("named_cnd3"); }
        ~condition_remove(){ named_condition::remove("named_cnd3"); }
    } cond_remover;
    named_mutex named_mtx(open_or_create, "named_mutex3");
    shared_memory_object shm (create_only, "MySharedMemory3", read_write);
    named_condition named_cond(open_or_create, "named_cnd3");
    shm.truncate(1000);

      //Map the whole shared memory in this process
    mapped_region region(shm, read_write);

      //Write all the memory to 1
      // std::memset(region.get_address(), 1, region.get_size());
    std::memset(region.get_address(), 0, region.get_size());

    int *mem = static_cast<int*>(region.get_address());
    current_process = &mem[0]; 
    *current_process = -1;
    printf("init curr: %d\n", *current_process);
    
    while (true) 
    {
      //do nothing. Keep shared memory alive. 
      if (*current_process == 1) {
        printf("curr%d\n", *current_process);
      }
    } 

    return 0;
}

