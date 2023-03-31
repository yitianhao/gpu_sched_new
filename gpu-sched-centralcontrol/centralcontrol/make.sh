#g++ -c -fPIC -I /home/yhao/gpu-sched/minor/boost_1_61_0 exp.cpp -lrt -lpthread -o exp.o
g++ -o controller controller.cpp -lmemcached -lrt -lpthread 
g++ -c -fPIC -o sendReceive.o sendReceive.cc -lrt -lmemcached
#g++ -g -I /home/yhao/gpu-sched/minor/boost_1_61_0 expset.cc -lrt -lpthread -o expset
g++ -shared -fPIC -Wl,-soname,libgeek.so -o libgeek.so sendReceive.o -lrt -lmemcached
