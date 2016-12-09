# -*- coding:utf-8 -*-

# 求解二元一次方程组
# 如：
"""
-x + y = 1
3x + y = 4
"""
# step1:将未知数的系数放入一个二维列表[[-1,1],[3,1]]
# step2:将等式右边的值放入一个一维列表[1,4]

import numpy as np

a = np.array([[-1,1],[3,1]])
b = np.array([1,4])

print np.linalg.solve(a,b)
