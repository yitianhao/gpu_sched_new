#ifndef _MYEXPSET
#define _MYEXPSET
#include <iostream>
#include <libmemcached/memcached.h>
using namespace std;

const char *key= "current_process";
const char *v_empty= "-1";
const char *v_0= "0";
const char *v_1= "1";

extern "C" void sendToCentral(const char* controlName, const char *value)
{
    //connect to server 
    memcached_server_st *servers = NULL;
    memcached_st *memc;
    memcached_return rc;
    
    memc= memcached_create(NULL);

    servers= memcached_server_list_append(servers, "localhost", 11211, &rc);
    rc= memcached_server_push(memc, servers);

    if (rc != MEMCACHED_SUCCESS)
    {
      fprintf(stderr,"Couldn't add server: %s\n",memcached_strerror(memc, rc));
      return;
    }

    //send to server 
    rc= memcached_set(memc, controlName, strlen(controlName), value, strlen(value), (time_t)0, (uint32_t)0);

    if (rc == MEMCACHED_SUCCESS)
      fprintf(stderr,"Send %s successfully\n", controlName);
    else
      fprintf(stderr,"Couldn't send %s: %s\n", controlName, memcached_strerror(memc, rc));
    
}


extern "C" void requestFromCentral(const char *controlName, char **result, size_t *resultSize)
{
    //connect to server 
    memcached_server_st *servers = NULL;
    memcached_st *memc;
    memcached_return rc;
    
    memc= memcached_create(NULL);

    servers= memcached_server_list_append(servers, "localhost", 11211, &rc);
    rc= memcached_server_push(memc, servers);

    if (rc != MEMCACHED_SUCCESS)
    {
      fprintf(stderr,"Couldn't add server: %s\n",memcached_strerror(memc, rc));
      return;
    }

    // request from server
    uint32_t returnFlag;
    *result = memcached_get(memc, controlName, strlen(controlName), resultSize, &returnFlag, &rc);

    if (rc == MEMCACHED_SUCCESS)
      fprintf(stderr,"Request %s successfully: %s\n", controlName, *result);
    else
      fprintf(stderr,"Couldn't request %s: %s\n",controlName, memcached_strerror(memc, rc));
}

#endif