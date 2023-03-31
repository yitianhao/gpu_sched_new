#ifndef _LOG_TRACE_H
#define _LOG_TRACE_H
#include <chrono>
#include <fstream>
#include <sstream>

const char metrices_file[] = "/var/lib/alnair/workspace/metrics.log";
static void log_api_call(const char *symbol) 
{
    std::chrono::time_point<std::chrono::system_clock> now =
		std::chrono::system_clock::now();  
    auto duration = now.time_since_epoch();
    std::ofstream fmet; 
    fmet.open(metrices_file, std::fstream::app);  
    //print timestamp at nano seconds when a cuda API is called
    fmet << duration.count() << ',' << symbol << std::endl;
    fmet.close();
}  
/************************************************/
#include <execinfo.h>
static void print_trace (void)
{
  void *array[10];
  char **strings;
  int size, i;

  size = backtrace (array, 10);
  strings = backtrace_symbols (array, size);
  if (strings != NULL)
  {
    fprintf(stderr, "Obtained %d stack frames.\n", size);
    for (i = 0; i < size; i++)
      fprintf(stderr, "%s\n", strings[i]);
  }

  free (strings);
}
/************************************************/
#endif
