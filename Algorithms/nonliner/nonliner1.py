# -*- coding:utf-8 -*-
import numpy as np
from sympy import *
from optparse import OptionParser
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
        print(p,a)
        print("++++")
        print(type(a))
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
    print(x)
    print(-grad(x)/np.linalg.norm(grad(x)))
    #print(grad(x),np.sqrt(grad(x).dot(grad(x))))
    # print(grad(x)[0]/np.sqrt(grad(x).dot(x)),grad(x)[1]/np.sqrt(grad(x).dot(x)))
    return -grad(x)/np.linalg.norm(grad(x))
	
# 牛顿方向算法
# 输入 (目标函数，梯度函数，x，(海塞矩阵函数,))
# 输出（如果海塞矩阵是正定的）牛顿方向，或者负梯度
def newton_method(f, grad, x, params):
    hess = params[0](x)
    if is_pos_def(hess):
        return -(np.linalg.inv(hess)).dot(grad(x))
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

def hess3(x):
    return np.array([[2,-1],[-1,2]])

def hess4(x):
    return np.array([[2,0],[0,50]])
# 最佳步长法：
# 梯度 × 梯度的转置 × 梯度的模
#————————————————————————————————
# 梯度 × f(x)的海瑟矩阵 × 梯度的转置
def best_track(grad,x):
    print("****")
    print(grad(x))
    print(grad(x).dot(np.mat(grad(x)).T).dot(np.linalg.norm(grad(x))))/(grad(x).dot(hess3(x)).dot(np.mat(grad(x)).T))
    return (grad(x).dot(np.mat(grad(x)).T).dot(np.linalg.norm(grad(x))))/(grad(x).dot(hess3(x)).dot(np.mat(grad(x)).T))


def f3(x):
    return (x[0])**2 + (x[1])**2 - x[0]*x[1] - 10*x[0] - 4*x[1] + 60

def grad3(x):
    return np.array([2*x[0]-x[1]-10, 2*x[1]-x[0]-4])

# def hess3(x):
#     return np.array([[2,0],[0,50]])


	
# result1 = line_search(f3, grad3, np.array([0.0,0.0]), grad_descent, backtrack,None,\
#                       (1,0.5,0.0001),epsilon=0.1)

def line_search2(f, grad, init, direction_algo, steplength_algo, \
                direction_params=None, steplength_params=None, epsilon=0, iterations=5):
    sequence = [init]         # 起始点
    x = init
    print(type(x),x)
    prev_val = float('inf')    # 正无穷
    counter = 0
    while abs(f(x)-prev_val)>epsilon and counter<iterations:
        p = direction_algo(f,grad,x,direction_params)
        a = steplength_algo(grad,x)
        print("++++")
        print(p,a)
        print(type(a))
        print(type(a*p))
        prev_val = f(x)
        counter += 1
        x=x+float(a)*p
        # print(a)

    
        # x = float(a)*p
        
        sequence.append(x)    # 遍历的点坐标
        print("\n")
        print(f(x))
    for i in sequence:
        print(i)
    return(sequence)
result2 = line_search2(f3, grad3, np.array([0.0,0.0]), grad_descent, best_track,direction_params=None, steplength_params=None,epsilon=0.1)


def f4(x):
    return (x[0])**2 + 25*(x[1])**2

def grad4(x):
    return np.array([2*x[0], 50*x[1]])



#result3 = line_search2(f4, grad4, np.array([2.0,2.0]), grad_descent, best_track,direction_params=None, steplength_params=None,epsilon=0.1)



# def main():


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
                         type="string")
    
    (options, args) = optparser.parse_args()
    print(options,args)
    
    f = options.func
    print(type(f),f)

    e = options.epsilon
    print(type(e),e)

    n = options.number
    print(type(n),n)

    s = options.start_point
    print(type(s),eval(s))
 

    # (args) = symbols(str(args[0],args[1]))
    x,y = symbols("x y")
    a = eval(f)
    print(a.subs({x:1,y:1}))
    b=diff(a,x)
    print(b.subs({x:1,y:2}))
    print(diff(a,x))
    print(type(a),a)

    def fun(a,data):
        # x,y = symbols("x y")
        # x,y = data[0],data[1]
        print(x,y)
        print(type(a),a)
        print(dir(a))
        #print(help(a.subs))
        print(a.subs({x:1,y:2}))
        print(a.evalf(subs={x:1,y:2}))
        #re = a._matches_commutative(a){x_: x, y_: y}
        #print(type(re),re)
        return a.subs({x:data[0],y:data[1]})

    def grad(a,data):
        return np.array([diff(a,x).subs({x:data[0],y:data[1]}), diff(a,y).subs({x:data[0],y:data[1]})])
        
    #(a+b*c)._matches_commutative(x+y*z, repl_dict={a: x}, evaluate=True) {a_: x, b_: y, c_: z}
    def hess(a,data):
        return np.array([[diff(a,x,x),diff(a,x,y)], [diff(a,y,x),diff(a,y,y)]])

    point = np.array([0.0,0.0])

    print(fun(a,point),grad(a,point),hess(a,point))
    # result = line_search2(fun(a), grad(a), np.array([0.0,0.0]), grad_descent, best_track,direction_params=None, steplength_params=None,epsilon=0.1)
