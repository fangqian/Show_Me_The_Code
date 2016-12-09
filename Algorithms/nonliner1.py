# -*- coding:utf-8 -*-
import numpy as np
import numpy
import matplotlib.pyplot as plt
from numba import jit

# 线搜索算法框架
# 输入 (目标函数，梯度函数，起始点，方向算法，步长算法，方向算法的参数，步长算法的参数，
#       终止条件的 epsilon，终止条件的迭代次数)
# 注意：如果 epsilon 和 iterations 都不填的话，你就安静地坐下来，等待时间的洪流带你去往彼方吧。
# 输出一个优化的搜寻路径
@jit
def line_search(f, grad, init, direction_algo, steplength_algo, \
                direction_params=None, steplength_params=None, epsilon=0, iterations=float('inf')):
    sequence = [init]         # 起始点
    x = init
    prev_val = float('inf')    # 正无穷
    counter = 0
    while abs(f(x)-prev_val)>epsilon and counter<iterations:
        p = direction_algo(f,grad,x,direction_params)
        a = steplength_algo(f,grad,x,p,steplength_params)
       # print(p,a)
        prev_val = f(x)
        counter += 1
       # print(x[0],x[1],grad(x),f(x))
        x = x+a*p
        sequence.append(x)    # 遍历的点坐标
    return(sequence)
	
# 梯度下降方向算法
# 输入 (目标函数，梯度函数，x)
# 输出在 x 的负梯度向量
def grad_descent(f, grad, x, params=None):
    print(grad(x),np.sqrt(grad(x).dot(grad(x))))
    # print(grad(x)[0]/np.sqrt(grad(x).dot(x)),grad(x)[1]/np.sqrt(grad(x).dot(x)))
    return -grad(x)
	
# 牛顿方向算法
# 输入 (目标函数，梯度函数，x，(海塞矩阵函数,))
# 输出（如果海塞矩阵是正定的）牛顿方向，或者负梯度
def newton_method(f, grad, x, params):
    hess = params[0](x)
    if is_pos_def(hess):
        return -(numpy.linalg.inv(hess)).dot(grad(x))
    else:
        return -grad(x)
		
# 判断一个矩阵是否正定
# 输入矩阵
# 输出 Boolean
def is_pos_def(mat):
    return(np.all(np.linalg.eigvals(mat) > 0) and np.all(mat==mat.T))
	
# 回溯步长算法
# 输入 (目标函数，梯度函数，起始点，方向，参数)
# 参数是 (初始 alpha，每次迭代的缩减倍数，Armijo 条件中的 c1)
# 返回一个合适的步长
@jit
def backtrack(f,grad,x,p, params):
    (alpha, rho, c) = params
    a = alpha
    while f(x+a*p)>f(x)+c*a*p.dot(grad(x)):
        a = a*rho
    return a

def f3(x):
    return (x[0])**2 + (x[1])**2 - x[0]*x[1] - 10*x[0] - 4*x[1] + 60

def grad3(x):
    return np.array([2*x[0]-x[1]-10, 2*x[1]-x[0]-4])
	
result1 = line_search(f3, grad3, np.array([0.0,0.0]), grad_descent, backtrack,None,\
                      (1,0.5,0.0001),epsilon=0.1)

# print(result1)
# print(f3(result1[-1]))

						   

