CXX = g++

OBJ_DIR := build/obj
LIB_DIR := build/lib
BIN_DIR := build/bin
_builddirs := $(shell mkdir -p $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR))

CUDA_BIN_DIR := $(shell dirname `which nvcc`)
CUDA_INCLUDE_DIR := $(shell readlink -f $(CUDA_BIN_DIR)/../include)

# compile flags for experimenting:
# -D_DEBUG_FILL_R
# -D_RECORD_LAUNCH_TIMES
# -D_CHANGE_ALLOC_TEST
# -D_RECORD_UTIL
# -D_PROFILE_ELAPSED_TIME
# -D_VERBOSE_WENQING
CXXFLAGS += -fPIC -O3 -D_RECORD_UTIL -D_DYN_ADJUST_FR -D_SCHEDULER_LOCK
# CXXFLAGS += -fPIC -O3 -D_RECORD_UTIL -D_DYN_ADJUST_FR -D_SCHEDULER_LOCK -D_VERBOSE_WENQING
# CXXFLAGS += -fPIC -O3 -D_RECORD_UTIL -D_DYN_ADJUST_FR -D_KERNEL_COUNT_WENQING
# CXXFLAGS += -fPIC -O3 -D_RECORD_UTIL -D_DYN_ADJUST_FR
# Flags for no-control CUDA kernel timestamp logging
# CXXFLAGS += -fPIC -O3 -D_RECORD_UTIL -D_DYN_ADJUST_FR -D_KERNEL_TIMESTAMP_WENQING
# CXXFLAGS += -fPIC -O3 -D_RECORD_UTIL -D_DYN_ADJUST_FR -D_VERBOSE_WENQING
#print per kernel time
# CXXFLAGS += -fPIC -O3 -D_RECORD_UTIL -D_DYN_ADJUST_FR -D_PROFILE_ELAPSED_TIME

OBJ_FILES := $(addprefix $(OBJ_DIR)/, register_client.o hooks.o)

.PHONY: lib
lib: $(OBJ_FILES) $(OBJ_DIR)/cuda_interpose.o
	$(CXX) $^ -shared -fPIC  -L/usr/local/cuda/lib64 $(LDFLAGS) -lcuda -lcudart -lnvidia-ml -o $(LIB_DIR)/libcuinterpose.so -lrt -lpthread -lmemcached

$(OBJ_DIR)/hooks.o: src/hooks.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $< -lmemcached

$(OBJ_DIR)/%.o: src/%.c
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

$(OBJ_DIR)/cuda_interpose.o: src/cuda_interpose.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) $(CXXFLAGS) -c -o $@ $^

.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/* $(BIN_DIR)/client-tool $(LIB_DIR)/libcuinterpose.so
