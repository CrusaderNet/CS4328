import numba
from random import *
import math
from math import sqrt

def go_fast(pi): 
    inside = 0
    inputs =10
    for i in range(0,inputs):
        x=random()
        y=random()
        if sqrt(x*x+y*y)<=1:
            inside+=1
    pi=4*inside/inputs 
    return pi

pi = math.pi      
go_fast(pi)
print(pi)
