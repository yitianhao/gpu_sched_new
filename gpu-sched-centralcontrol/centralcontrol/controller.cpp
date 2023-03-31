#include <iostream>
#include <libmemcached/memcached.h>
#include <unistd.h>
#include <chrono>
#include <thread>
using namespace std;

static int *current_process;

int main(int argc, char *argv[])
{
    //connect to cache server 
    memcached_server_st *servers = NULL;
    memcached_st *memc;
    memcached_return rc;
    
     memc= memcached_create(NULL);

    servers= memcached_server_list_append(servers, "localhost", 11211, &rc);
    rc= memcached_server_push(memc, servers);

    if (rc == MEMCACHED_SUCCESS)
      fprintf(stderr,"Controller add server successfully\n");
    else
      fprintf(stderr,"Controller Couldn't add server: %s\n",memcached_strerror(memc, rc));

    // init current process control bit
    const char *key= "current_process";
    const char *value= "-1";
    printf("init current_process: %s\n", value);

    rc= memcached_set(memc, key, strlen(key), value, strlen(value), (time_t)0, (uint32_t)0);

    if (rc != MEMCACHED_SUCCESS)
      fprintf(stderr,"Couldn't set current_process: %s\n",memcached_strerror(memc, rc));
    else
      fprintf(stderr,"Set current_process successfully\n");
    
    while (true) 
    {
      size_t returnSize;
      uint32_t returnFlag;
      char *returnValue;
      returnValue = memcached_get(memc, key, strlen(key), &returnSize, &returnFlag, &rc);

      if (rc != MEMCACHED_SUCCESS)
        fprintf(stderr,"Couldn't get current_process: %s\n",memcached_strerror(memc, rc));
      //Monitor current control bit. 
      if (strcmp(returnValue, "-1") != 0) {
        printf("current_process %s\n", returnValue);
      }
      free(returnValue);
      this_thread::sleep_for(chrono::milliseconds(1));
    } 

    return 0;
}

