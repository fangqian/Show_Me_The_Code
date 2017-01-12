import numpy as np
from sympy import *

x,y,m = symbols("x y m")

f1 = x + y
f2 = -x**2 + y

f3 = x 

f4 = f1 + f2 + f3
#f4 = -x**2 + 2*x + 2*y

f5 = f1 + m*(x**2 + (-x**2 + y)**2)

print(diff(f5,x))
print(diff(f5,y)) 