#g++ -c -fPIC -I /home/yhao/gpu-sched/minor/boost_1_61_0 exp.cpp -lrt -lpthread -o exp.o
g++ -g -I ../minor/boost_1_61_0 exp.cpp -lrt -lpthread -o expcontorller 
g++ -c -fPIC -I ../minor/boost_1_61_0 expset.cc -lpthread -o expset.o -lrt 
#g++ -g -I /home/yhao/gpu-sched/minor/boost_1_61_0 expset.cc -lrt -lpthread -o expset
g++ -shared -fPIC -Wl,-soname,libgeek.so -I ../minor/boost_1_61_0 -o libgeek.so -lpthread expset.o -lrt
