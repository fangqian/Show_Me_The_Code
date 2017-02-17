# -*- coding:utf-8 -*-
import numpy as np
import math
from sympy import *
from optparse import OptionParser
import numpy.linalg as LA
from numba import jit

# 线搜索算法框架
# 输入 (目标函数，梯度函数，起始点，方向算法，步长算法，方向算法的参数，步长算法的参数，
#       终止条件的 epsilon，终止条件的迭代次数)
# 注意：如果 epsilon 和 iterations 都不填的话，你就安静地坐下来，等待时间的洪流带你去往彼方吧。
# 输出一个优化的搜寻路径
@jit
def line_search(f, fun, grad, init, direction_algo, steplength_algo, \
                direction_params=None, steplength_params=None, epsilon=0, iterations=float('inf')):
    sequence = [init]         # 起始点
    data = init
    prev_val = float('inf')    # 正无穷
    counter = 0
    while abs(fun(f,data)-prev_val)>epsilon and counter<iterations:
        print("=============")
        p = direction_algo(f,grad,data,hess)
        s = steplength_algo(f,grad,data)
        prev_val = fun(f,data)
        counter += 1
        data = data+float(s)*p
        sequence.append(data)    # 遍历的点坐标
    print(sequence)
    return(sequence)
	
# 梯度下降方向算法
# 输入 (目标函数，梯度函数，x)
# 输出在 x 的负梯度向量
def grad_descent(f, grad, data, hess):
    print("zzzzzzzz")
    print(-(grad(f,data))/(LA.norm(grad(f,data),2)))
    print("zzzzzzzz")

    return -(grad(f,data))/(np.linalg.norm(grad(f,data)))
	
# 牛顿方向算法
# 输入 (目标函数，梯度函数，x，(海塞矩阵函数,))
# 输出（如果海塞矩阵是正定的）牛顿方向，或者负梯度
def newton_method(f, grad, x, hess):
    hess = hess(f,x)
    if is_pos_def(hess):
        return -(np.linalg.inv(hess)).dot(grad(f,x))
    else:
        return -grad(x)
		
# 判断一个矩阵是否正定
# 输入矩阵
# 输出 Boolean
def is_pos_def(mat):
    return(np.all(np.linalg.eigvals(mat) > 0) and np.all(mat==mat.T))
	
# # 回溯步长算法
# # 输入 (目标函数，梯度函数，起始点，方向，参数)
# # 参数是 (初始 alpha，每次迭代的缩减倍数，Armijo 条件中的 c1)
# # 返回一个合适的步长
# @jit
# def backtrack(f,grad,x,p, params):
#     (alpha, rho, c) = params
#     a = alpha
#     while f(x+a*p)>f(x)+c*a*p.dot(grad(x)):
#         a = a*rho
#     return a


def grad(f,data):
    return np.array([float(diff(f,x).subs({x:data[0],y:data[1]})), float(diff(f,y).subs({x:data[0],y:data[1]}))])

# 最佳步长法：
# 梯度 × 梯度的转置 × 梯度的模
#————————————————————————————————
# 梯度 × f(x)的海瑟矩阵 × 梯度的转置
def best_track(f,grad,data):
    # print("yyyyyy")
    # print(grad(f,data))
    # print(grad(f,data).dot(np.mat(grad(f,data)).T).dot(np.linalg.norm(grad(f,data))))/(grad(f,data).dot(hess(f,data)).dot(np.mat(grad(f,data)).T))

    tidu = grad(f,data)
    tidu.shape = (1,2)
    print(tidu)
    print("XXXXXX")
    print(tidu.dot(np.mat(tidu).T).dot(LA.norm(tidu)))/(tidu.dot(hess(f,data)).dot(np.mat(tidu).T))
    return (tidu.dot(np.mat(tidu).T).dot(LA.norm(tidu)))/(tidu.dot(hess(f,data)).dot(np.mat(tidu).T))



def fun(f,data):
    return f.subs({x:data[0],y:data[1]})


    
    
def hess(f,data):
    return np.array([[diff(f,x,x),diff(f,x,y)], [diff(f,y,x),diff(f,y,y)]])




if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--function',
                         dest='func',
                         help='function name that want to be solved',
                         type='string',
                         default=None)

    optparser.add_option('-e', '--epsilon',
                         dest='epsilon',
                         help='epsilon  allowed',
                         default=0.01,
                         type='float')
    optparser.add_option('-n', '--number',
                         dest='number',
                         help='numbers of independent variable',
                         default=2,
                         type='int')
    optparser.add_option('-s','--start_point',
                         dest='start_point',
                         help='start point coordinates',
                         default=[2,2],
                         type="string")
    
    (options, args) = optparser.parse_args()
  #  print(options,args)
    
    x,y = symbols("x y")

    f = eval(options.func)
  #  print(type(f),f)
    data = np.array([0.0,0.0])
  #  print("**********************")
  #  print(np.array([diff(f,x).subs({x:data[0],y:data[1]}), diff(f,y).subs({x:data[0],y:data[1]})]))
 #   print("**********************")

    e = options.epsilon
  #  print(type(e),e)

    n = options.number
   # print(type(n),n)

    s = eval(options.start_point)
   # print(type(s),s)
 

    # (args) = symbols(str(args[0],args[1]))
    #x,y = symbols("x y")
    # a = eval(f)

    # def fun(f,data):
    #     return f.subs({x:data[0],y:data[1]})

    # def grad(a,data):
    #     return np.array([diff(f,x).subs({x:data[0],y:data[1]}), diff(f,y).subs({x:data[0],y:data[1]})])
        
    # def hess(a,data):
    #     return np.array([[diff(f,x,x),diff(f,x,y)], [diff(f,y,x),diff(f,y,y)]])

    point = np.array([0.0,0.0])

    print(fun(f,point),grad(f,point),hess(f,point))
    result = line_search(f, fun, grad, np.array(s), grad_descent, best_track,direction_params=None, steplength_params=None,epsilon=0.1)
