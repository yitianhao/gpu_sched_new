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
# -D_SYNC_QUEUE
CXXFLAGS = -fPIC -O3 -D_DYN_ADJUST_FR -D_SCHEDULER_LOCK  
CXXSYNCFLAGS = -fPIC -O3 -D_DYN_ADJUST_FR -D_SCHEDULER_LOCK -D_SYNC_QUEUE
CXXEVENTFLAGS = -fPIC -O3 -D_DYN_ADJUST_FR -D_SCHEDULER_LOCK -D_GROUP_EVENT
# Verbose for debugging
# CXXFLAGS += -D_VERBOSE_WENQING
# CXXSYNCFLAGS += -D_VERBOSE_WENQING

OBJ_FILES := $(addprefix $(OBJ_DIR)/, register_client.o hooks.o)
OBJ_FILES_SYNC := $(addprefix $(OBJ_DIR)/, register_client.o hooks_sync.o)
OBJ_FILES_EVENT := $(addprefix $(OBJ_DIR)/, register_client.o hooks_event.o)
all : lib libsync libevent 
.PHONY: all 

lib: $(OBJ_FILES) $(OBJ_DIR)/cuda_interpose.o
	$(CXX) $^ -shared -fPIC  -L/usr/local/cuda/lib64 $(LDFLAGS) -lcuda -lcudart -lnvidia-ml -o $(LIB_DIR)/libcuinterpose.so -lrt -lpthread

libsync: $(OBJ_FILES_SYNC) $(OBJ_DIR)/cuda_interpose.o
	$(CXX) $^ -shared -fPIC  -L/usr/local/cuda/lib64 $(LDFLAGS) -lcuda -lcudart -lnvidia-ml -o $(LIB_DIR)/libcuinterpose_sync.so -lrt -lpthread

libevent: $(OBJ_FILES_EVENT) $(OBJ_DIR)/cuda_interpose.o
	$(CXX) $^ -shared -fPIC  -L/usr/local/cuda/lib64 $(LDFLAGS) -lcuda -lcudart -lnvidia-ml -o $(LIB_DIR)/libcuinterpose_event.so -lrt -lpthread

$(OBJ_DIR)/hooks.o: src/hooks.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) -I ../minor/boost_1_61_0 $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

$(OBJ_DIR)/hooks_sync.o: src/hooks.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) -I ../minor/boost_1_61_0 $(CXXSYNCFLAGS) $(CPPFLAGS) -c -o $@ $<

$(OBJ_DIR)/hooks_event.o: src/hooks.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) -I ../minor/boost_1_61_0 $(CXXEVENTFLAGS) $(CPPFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: src/%.c
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

$(OBJ_DIR)/cuda_interpose.o: src/cuda_interpose.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) $(CXXFLAGS) -c -o $@ $^

.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/* $(BIN_DIR)/client-tool $(LIB_DIR)/libcuinterpose.so $(LIB_DIR)/libcuinterpose_sync.so $(LIB_DIR)/libcuinterpose_event.so

