CXX = g++
CXXFLAGS = -lrt -lpthread
INCLUDE_DIR = ../minor/boost_1_61_0

all: expcontroller expset.o libgeek.so

.PHONY: all
expcontroller: exp.cpp
	$(CXX) -g -I $(INCLUDE_DIR) $< $(CXXFLAGS) -o $@

expset.o: expset.cc
	$(CXX) -c -fPIC -I $(INCLUDE_DIR) $< $(CXXFLAGS) -o $@

libgeek.so: expset.o
	$(CXX) -shared -fPIC -Wl,-soname,$@ -I $(INCLUDE_DIR) $< $(CXXFLAGS) -o $@

.PHONY: clean
clean:
	rm expcontroller expset.o libgeek.so
