# import the module
from ctypes import cdll
  
# load the library
lib = cdll.LoadLibrary('./libgeek.so')
 
lib.setMem(1)
