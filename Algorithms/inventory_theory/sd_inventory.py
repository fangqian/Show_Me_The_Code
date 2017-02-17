# -*- coding:utf-8 -*-
'''
Description : Check the data input 
require     : windows Anaconda-2.3.0
author      : qiangu_fang@163.com
usage  $ python sd_inventory.py -d -a ...
'''
import os
import sys
import bisect
import logging
import math
from math import exp,factorial
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton
from scipy.optimize import fsolve
from optparse import OptionParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(BASE_DIR)

inventory_logger = logging.getLogger('SD_API.Method.sd_queue')
inventory_logger.setLevel(logging.INFO)
fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_Inventory.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
inventory_logger.addHandler(fh)


dicts = {"best_order":"最优订货量", "inventory_cost":"单位时间内的存储费", "order_cost":"单位时间内的订货费", 
         "total_fee":"总费用", "max_inventory":"最大存储水平","ave_inventory":"平均存储水平", 
         "advance_point":"再订货点", "times":"订货/生产次数", "days":"周期(每单位时间)", "shortage_cost":"单位时间内的缺货费", 
         "best_product":"最佳生产量","product_cost":"单位时间内的生产费", "max_shortage":"最大缺货量", "ave_shortage":"平均缺货量",
         "C0":"供过于求时单位产品总成本", "Cu":"供不应求时单位产品的总成本", "SL":"最优服务水平","fQ":"最优期望收益",
         "cQ":"最优期望成本","EX":"需求均值","DX":"需求标准差",
         "B":"单位产品期望缺货成本","D":"需求的均值", "dl":"提前期内的平均需求量", "s":"再订货点", 
         "SS":"安全存储量", "shortage_percent":"缺货概率", "standard":"提前期内需求的标准差"}


def EOQ(d, c, m, x, y):

    best_order = math.sqrt(2.0*d*c/m)
    order_cost = inventory_cost = math.sqrt(d*m*c/2.0)
    max_inventory = best_order
    ave_inventory = max_inventory/2.0
    total_fee = math.sqrt(2.0*d*c*m)
    advance_point = x*d/y
    days = y/(d/best_order)
    times = math.sqrt(m*d/(2.0*c))
    return(["best_order","inventory_cost","order_cost","total_fee","max_inventory","ave_inventory","advance_point","times","days"],
    	    [best_order, inventory_cost,order_cost,total_fee,max_inventory, ave_inventory,advance_point, times, days])

def EPQ(d, a, b, m, x, y):

    best_product = math.sqrt(2*d*b/((1-d/a)*m))
    product_cost = inventory_cost = b * d/best_product
    total_fee = product_cost + inventory_cost
    max_inventory = math.sqrt(2*d*b*(1-d/a)/m)
    ave_inventory = max_inventory/2.0
    advance_point = x*d/y
    days = y/(d/best_product)
    times = d/best_product

    return(["best_product", "product_cost", "inventory_cost", "total_fee", "max_inventory",
            "ave_inventory","advance_point", "times", "days"],
           [best_product, product_cost, inventory_cost, total_fee, max_inventory,
            ave_inventory, advance_point, times, days])

def AEOQ(d, c, e, m, x, y):

    best_order = math.sqrt(2.0*d*c*(m+e)/(e*m))
    max_shortage = best_order*(m/(m+e))
    ave_shortage = 1.0/2.0 *best_order* (m/m+e)**2 
    shortage_cost = max_shortage**2*e/(2*best_order)
    order_cost = d/best_order*c
    inventory_cost = m*(best_order-max_shortage)**2/(2*best_order)
    total_fee = order_cost + inventory_cost + shortage_cost
    max_inventory = best_order-max_shortage
    ave_inventory = (best_order-max_shortage)**2/(best_order*2.0)
    advance_point = x*d/y - max_shortage
    days = y/(d/best_order)
    times = d/best_order
    
    return(["best_order", "inventory_cost", "order_cost", "shortage_cost", "total_fee", "max_inventory",
            "ave_inventory", "max_shortage", "ave_shortage", "advance_point", "times", "days"],
           [best_order, inventory_cost, order_cost, shortage_cost, total_fee, max_inventory,
            ave_inventory, max_shortage, ave_shortage, advance_point, times, days])

def AEPQ(d, a, b, e, m, x, y):
    
    best_product = math.sqrt(2.0*d*b*(m+e)/(m*e*(1-d/a)))
    shortage_cost = (e*best_product*(1-d/a)*(m/(m+e))**2)/2
    inventory_cost = (m*best_product*(1-d/a)*(e/(m+e))**2)/2
    product_cost = b*d/best_product
    
    max_shortage = math.sqrt(2*b*d*m*(1-d/a)/(e*(m+e)))
    total_fee = math.sqrt(2*d*m*e*b*(1-d/a)/(m+e))
 
    ave_inventory = (best_product*(1-d/a)-max_shortage)**2/(2*best_product*(1-d/a))
    ave_shortage = max_shortage**2/(2*best_product*(1-d/a))
    max_inventory = ave_inventory*2+max_shortage
    days = y*best_product/d 
    times = d/best_product

    advance_point = x*d/y - max_shortage

    return(["best_product", "inventory_cost", "shortage_cost", "product_cost", "total_fee",
            "max_inventory", "ave_inventory", "max_shortage", "ave_shortage", "advance_point", "times", "days"],
            [best_product, inventory_cost, shortage_cost, product_cost, total_fee, max_inventory, ave_inventory,
             max_shortage, ave_shortage, advance_point, times, days])


# example 1 in yfj's ppt No.19 page
def EODQ(d, c, m, price, number, x, y):

    best_order = math.sqrt(2*c*d/m)
    position = bisect.bisect_left(number,best_order)

    def cost(d, c, m, price,number):
    	return c*d/number + price*d + 1/2.0*m*number

    best_order_cost = cost(d,c,m,price[position],best_order)

    costs = [cost(d, c, m, price[i+1], number[i]) for i in range(len(number))]
    min_cost = min(costs)
    index = costs.index(min_cost)

    if best_order_cost < min_cost:
        total_fee = best_order_cost
        best_order = best_order
    else:
    	total_fee = min_cost
    	best_order = number[index]
    
    inventory_cost = 1/2.0*m*best_order
    shortage_cost = 0
    order_cost = total_fee - inventory_cost

    max_inventory = best_order
    ave_inventory = best_order/2.0

    advance_point = x*d/y
    days = y/(d/best_order)
    times = d/best_order
    
    return(["best_order",  "inventory_cost", "order_cost", "total_fee", "max_inventory", 
            "ave_inventory", "advance_point", "times", "days"],
           [best_order,  inventory_cost, order_cost, total_fee,  max_inventory, 
            ave_inventory, advance_point, days, times, days])


# example 2 in yfj's ppt No.20 page
def AEODQ(d, c, e, m, price, number, x, y):

    best_order = math.sqrt(2*c*d*(m+e)/(m*e))
    position = bisect.bisect_left(number, best_order)

    def cost(d, c, e, m, price,number):
    	return c*d/number + price*d + 1/2.0*m*number*e/(m+e)

    best_order_cost = cost(d, c, e, m, price[position], best_order)
    costs = [cost(d, c, e, m, price[i+1], number[i]) for i in range(len(number))]

    min_cost = min(costs)
    index = costs.index(min_cost)

    if best_order_cost < min_cost:
        total_fee = best_order_cost
        best_order = best_order
    else:
    	total_fee = min_cost
    	best_order = number[index]

    inventory_cost = 1/2.0*m*best_order*(e/(m+e))**2
    shortage_cost = 1/2.0*e*best_order*(m/(m+e))**2
    order_cost = total_fee - inventory_cost - shortage_cost

    max_inventory = e*best_order/(m+e)
    ave_inventory = 1/2.0*best_order*(e/(m+e))**2
    max_shortage = m/(m+e)*best_order
    ave_shortage = 1/2.0*best_order*(m/(m+e))**2

    advance_point = x*d/y - max_shortage
    days = y/(d/best_order)
    times = d/best_order

    return(["best_order", "max_shortage", "ave_shortage", "shortage_cost", "max_inventory", "ave_inventory",
            "inventory_cost","order_cost","total_fee", "advance_point", "times", "days"],
    	   [best_order, max_shortage, ave_shortage, shortage_cost, max_inventory, ave_inventory, inventory_cost,order_cost,
    	     total_fee, advance_point, times, days])

# example 3 in yfj's ppt No.21
def EOPQ(d, a, c, m, price, number, x, y):

    best_order = math.sqrt(2*c*d*a/(m*(a-d)))
    position = bisect.bisect_left(number, best_order)

    def cost(d, a, c, m, price, number):
        return c*d/number + price*d + 1/2.0*m*number*(a-d)/a

    best_order_cost = cost(d, a, c, m, price[position], best_order)
    costs = [cost(d, a, c, m, price[i+1], number[i]) for i in range(len(number))]

    min_cost = min(costs)
    index = costs.index(min_cost)
    
    if best_order_cost < min_cost:
        total_fee = best_order_cost
        best_order = best_order
    else:
    	total_fee = min_cost
    	best_order = number[index]

    inventory_cost = 1/2.0*m*best_order*(a-d)/a
    shortage_cost = 0
    order_cost = total_fee - inventory_cost

    max_inventory = (a-d)*best_order/a
    ave_inventory = 1/2.0*best_order*(a-d)/a
    max_shortage = 0
    ave_shortage = 0

    advance_point = x*d/y
    days = y/(d/best_order)
    times = d/best_order

    return(["best_order", "inventory_cost","order_cost", "total_fee", "max_inventory","ave_inventory",
    	    "advance_point", "times", "days"],
           [best_order, inventory_cost, order_cost, total_fee, max_inventory, ave_inventory,
            advance_point, times, days])

# example 4 in yfj's ppt No.22
def AEOPQ(d, a, c, e, m, price, number, x, y):

    best_order = math.sqrt(2*c*d*(m+e)*a/(m*e*(a-d)))
    position = bisect.bisect_left(number, best_order)

    def cost(d, a, c, e, m, price, number):
    	return c*d/number + price * d + 1/2.0 *m *number *e* (a-d)/(a*(m+e))

    best_order_cost = cost(d, a, c, e, m, price[position], best_order)
    costs = [cost(d, a, c, e, m, price[i+1], number[i]) for i in range(len(number))]

    min_cost = min(costs)

    index = costs.index(min_cost)
    
    if best_order_cost < min_cost:
        total_fee = best_order_cost
        best_order = best_order
    else:
    	total_fee = min_cost
    	best_order = number[index]

    inventory_cost = 1/2.0*m*best_order*(a-d)/a*(e/(m+e))**2
    shortage_cost = 1/2.0*e*best_order*(a-d)/a*(m/(m+e))**2
    order_cost = total_fee - inventory_cost - shortage_cost

    max_inventory = e*best_order/(m+e)*(a-d)/a
    ave_inventory = 1/2.0*best_order*(a-d)/a*(e/(m+e))**2
    max_shortage = m/(m+e)*best_order*(a-d)/a
    ave_shortage = 1/2.0*best_order*(a-d)/a*(m/(m+e))**2

    advance_point = x*d/y - max_shortage
    days = y/(d/best_order)
    times = d/best_order

    return(["best_order", "max_shortage","ave_shortage", "inventory_cost", "shortage_cost","order_cost", "total_fee",
            "max_inventory", "ave_inventory", "advance_point", "times", "days"],
            [best_order, max_shortage, ave_shortage, inventory_cost, shortage_cost, order_cost, total_fee,
             max_inventory, ave_inventory, advance_point, times, days])

def RandomRequestPossion(a,c, p, s, b, h, l, number, frequency):
    """
       c: price, p: sell_price, s: back_price  b: shortage_cost, h: inventory_cost, l:lambda
    """
    C0 = c-s+h
    Cu = p-c+b
    SL = Cu/(Cu+C0)	

    if l:
    	i = 0
        number = [i]
        sum_p = (l**i)/factorial(i)*(exp(-l))
        frequency = [sum_p]
        while sum_p<=0.9999:
        	i += 1
        	number.append(i)
        	pi = (l**i)*exp(-l)/factorial(i)
        	sum_p += (l**i)*exp(-l)/factorial(i)
        	frequency.append(pi)

    Sum = 0
    j = -1
    for i in frequency:
        Sum += i
        j+=1
        if Sum >= SL:
            best_order, index = number[j],j
            break
    if l:
    	EX = l
    	DX = math.sqrt(l)
    else:
    	EX = sum([number[i] * frequency[i] for i in range(len(number))])
    	DX = math.sqrt(sum([ frequency[i] * (number[i]-EX)**2 for i in range(len(number))]))

    cQ =  C0*sum([(best_order-number[i])*frequency[i] for i in range(index+1)]) + Cu*sum([(number[i]-best_order)*frequency[i] for i in range(index+1, len(frequency))]) + a
    fQ = sum([((p-s+h)*number[i] - C0*best_order)*frequency[i] for i in range(index+1)]) + sum([(Cu*best_order-b*number[i])*frequency[i] for i in range(index+1, len(frequency))]) - a
    
    return(["best_order","SL", "fQ", "cQ", "EX","DX","C0", "Cu"],
    	   [best_order, SL, fQ, cQ, EX, DX, C0, Cu])

def RandomRequestUniform(a,c, b, p, s, h, distribution):
    """
       a: order_cost, c: price, p: sell_price, b:shortage_cost, s: back_price, h:inventory_cost, distribution: uniform
    """
    C0 = c-s+h
    Cu = p-c+b
    SL = Cu/(Cu+C0)

    diff_ave = (distribution[-1]-distribution[0])/2.0
    add_ave  = (distribution[-1]+distribution[0])/2.0

    EX = add_ave
    DX = math.sqrt(1/12.0 * (2*diff_ave)**2)
    
    best_order = distribution[0] + SL*diff_ave*2.0
    fQ = (Cu+C0)*(SL*distribution[0]+diff_ave*SL**2) - b*add_ave - a
    cQ = Cu*add_ave - (C0+Cu)*(SL*distribution[0]+diff_ave*SL**2) + a

    return(["best_order","SL", "fQ", "cQ", "EX","DX","C0", "Cu"],
    	   [best_order, SL, fQ, cQ, EX, DX, C0, Cu])

def RandomRequestExponential(a,c, p, s, b, h, position, scale):
    C0 = c-s+h
    Cu = p-c+b
    SL = Cu/(Cu+C0)
    EX = position + 1.0/scale
    DX = 1.0/scale

    best_order = position - math.log(1-SL)/scale
    fQ = (Cu+C0) * ((position+1.0/scale)*SL + (1-SL)*math.log(1-SL)/scale) - b*(position+1.0/scale) - a
    cQ = Cu * (position+1.0/scale) - (Cu+C0)*((position+1.0/scale)*SL+(1-SL)*math.log(1-SL)/scale) + a

    return(["best_order","SL",  "fQ", "cQ", "EX","DX","C0", "Cu"],
    	   [best_order, SL, fQ, cQ, EX, DX, C0, Cu])

def RandomRequestNormal(a,c, p, s, b, h, average, standard_deviation):
    C0 = c-s+h
    Cu = p-c+b
    SL = Cu/(Cu+C0)
    EX = average
    DX = standard_deviation

    best_order = norm.ppf(SL, loc=average, scale=standard_deviation)
    fQ = best_order*Cu - (Cu+C0)*best_order*SL + (Cu+C0)*(average*SL-standard_deviation*norm.pdf((best_order-average)/standard_deviation, loc=0, scale=1)) - b*average - a  
    cQ = -(Cu+C0)*(average*SL-standard_deviation*norm.pdf((best_order-average)/standard_deviation, loc=0, scale=1)) + Cu*average + a
    
    return(["best_order","SL",  "fQ", "cQ", "EX","DX","C0", "Cu"],
    	   [best_order, SL, fQ, cQ, EX, DX, C0, Cu])

def s_Q_Uniform(a, c, h, p, e, q, o, B1, w, x, y, uniform):
    B = p*e + q*o
    D = (uniform[0]+uniform[-1])/2.0
    dl = D*x/y
    
    i = float("inf")
    best_order = math.sqrt(2.0*D*a/h)
    while w<i:
       s = 2.0*dl*(1-(h*best_order-B1*D/(2*dl))/(q*h*best_order+B*D))

       i = math.sqrt(2*D*(a+B*(dl+(s**2)/(4.0*dl)-s) + B1*(1-s/(2.0*dl)))/h) - best_order
       best_order = math.sqrt(2*D*(a+B*(dl+(s**2)/(4.0*dl)-s) + B1*(1-s/(2.0*dl)))/h)
    
    s = 2.0*dl*(1-(h*best_order-B1*D/(2*dl))/(q*h*best_order+B*D))
    SS = s - dl
    ave_shortage = dl + (s**2)/(2*2*dl) - s
    shortage_percent = 1 - s/(2*dl)

    order_cost = a*D/best_order + c*D
    inventory_cost = h*(best_order/2.0+s-dl+q*ave_shortage)
    shortage_cost = (B*D*ave_shortage+shortage_percent*B1*D)/best_order
    total_fee = order_cost + inventory_cost + shortage_cost

    return(["s", "best_order", "SS", "ave_shortage", "shortage_percent",
    	    "order_cost", "inventory_cost", "shortage_cost", "total_fee"],
    	    [s, best_order, SS,ave_shortage, shortage_percent,
    	     order_cost, inventory_cost, shortage_cost, total_fee])

def s_Q_Exponential(a, c, h, p, e, q, o, B1, w, x, y, l):
    B = p*e + q*o
    D = l
    dl = D*x/y

    i = float("inf")
    best_order = math.sqrt(2.0*D*a/h)

    while w<i:   
       s = -dl*math.log(h*best_order/(q*h*best_order+B*D+B1*D/dl))
       i = math.sqrt( 2*D*(a+B*dl*exp(-s/dl)+B1*exp(-s/dl))/h ) - best_order
       best_order = math.sqrt( 2*D*(a+B*dl*exp(-s/dl)+B1*exp(-s/dl))/h)

    s = -dl*math.log(h*best_order/(q*h*best_order+B*D+B1*D/dl))
    SS = s - dl
    ave_shortage = dl*exp(-s/dl)
    shortage_percent = exp(-s/dl)

    order_cost = a*D/best_order + c*D
    inventory_cost = h*(best_order/2.0+s-dl+q*ave_shortage)
    shortage_cost = (B*D*ave_shortage+shortage_percent*B1*D)/best_order
    total_fee = order_cost + inventory_cost + shortage_cost

    return(["s", "best_order", "SS", "ave_shortage", "shortage_percent",
    	    "order_cost", "inventory_cost", "shortage_cost", "total_fee"],
    	    [s, best_order, SS,ave_shortage, shortage_percent,
    	     order_cost, inventory_cost, shortage_cost, total_fee])

def s_Q_Normal(a, c, h, p, e, q, o, B1, w, x, y, average, standard_deviation):
    B = p*e + q*o
    D = average
    dl = x*average/y
    i = float("inf")
    best_order = math.sqrt(2.0*D*a/h)

    standard = standard_deviation*math.sqrt(x/y)
    
    I = [0]
    while w<i:
        def f(s,h,best_order,B1,D,B,q):
            return norm.cdf(s,loc=dl, scale=standard) - 1 + (h*best_order-B1*D*norm.pdf(s,loc=dl,scale=standard))/(q*h*best_order
            	+B*D)

        s = fsolve(f,x0=0, args=(h,best_order,B1,D,B,q))[0]
        i = math.sqrt(2*D/h*(a+B*(standard*norm.pdf((s-dl)/standard, loc=0, scale=1)+(dl-s)*(1-norm.cdf(s, loc = dl, scale= standard))))) - best_order
        I.append(i)
        
        best_order = math.sqrt(2*D/h*(a+B*(standard*norm.pdf((s-dl)/standard, loc=0, scale=1)+(dl-s)*(1-norm.cdf(s, loc = dl, scale= standard))) + B1*1-norm.cdf(s,loc=dl, scale=standard)))
        

        if len(I)>2:
		    if abs(I[-2]-I[-1]) <= 0.000001:
		       i = w
	s = fsolve(f,x0=0, args=(h,best_order,B1,D,B,q))[0]	    
    SS = s - dl
    ave_shortage = standard*norm.pdf((s-dl)/standard, loc=0, scale=1)+(dl-s)*(1-norm.cdf(s, loc = dl, scale= standard))
    shortage_percent = 1-norm.cdf(s,loc=dl, scale=standard)

    order_cost = a*D/best_order + c*D
    inventory_cost = h*(best_order/2.0+s-dl+q*ave_shortage)
    shortage_cost = (B*D*ave_shortage+shortage_percent*B1*D)/best_order
    total_fee = order_cost + inventory_cost + shortage_cost

    return(["s", "best_order", "SS", "ave_shortage", "shortage_percent",
    	    "order_cost", "inventory_cost", "shortage_cost", "total_fee"],
    	    [s, best_order, SS,ave_shortage, shortage_percent,
    	     order_cost, inventory_cost, shortage_cost, total_fee])

def main(method, d, a, b, c, e, m, o, s, t, j, w, x, y, price, number):
    if method == "EOQ":
    	summary = EOQ(d, c, m, x, y)

    elif method == "EPQ":
    	summary = EPQ(d, a, b, m, x, y)

    elif method == "AEOQ":
    	summary = AEOQ(d, c, e, m, x, y)

    elif method == "AEPQ":
        summary = AEPQ(d, a, b, e, m, x, y)

    elif method == "EODQ":
    	summary = EODQ(d, c, m, price, number, x, y)

    elif method == "EOPQ":
    	summary = EOPQ(d, a, c, m, price, number, x, y)
    
    elif method == "AEODQ":
    	summary = AEODQ(d, c, e, m, price, number, x, y)

    elif method == "AEOPQ":
    	summary = AEOPQ(d, a, c, e, m, price, number, x, y)

    elif method == "RandomRequestPossion":
    	summary =  RandomRequestPossion(a, c, d, s, b, j, t, price, number)

    elif method == "RandomRequestUniform":
    	summary = RandomRequestUniform(a, c, b, d, s, j, price)

    elif method == "RandomRequestExponential":
    	summary = RandomRequestExponential(a,c, d, s, b, j, o, t)

    elif method == "RandomRequestNormal":
    	summary = RandomRequestNormal(a,c, d, s, b, j, o, t)

    elif method == "s_Q_Uniform":
        summary = s_Q_Uniform(a, c, j, d, e, s, o, t, w, x, y, price)

    elif method == "s_Q_Exponential":
    	summary = s_Q_Exponential(a, c, j, d, e, s, o, t, w, x, y, b)

    elif method == "s_Q_Normal":
    	summary = s_Q_Normal(a, c, j, d, e, s, o, t, w, x, y, b, m)

    else: inventory_logger.info("Method error, not %s method in this program"%method)

    return summary

if __name__ == "__main__":
    optparser = OptionParser()

    optparser.add_option('-d', '--need',
                         dest='need',
                         help='need per year',
                         default=None,
                         type="float")

    optparser.add_option('-a', '--production',
                         dest='production',
                         help='production per year',
                         default=None,
                         type="float")

    optparser.add_option('-b', '--prepare_cost',
                         dest='prepare',
                         help='prepare cost per year on production',
                         default=None,
                         type="float")


    optparser.add_option('-c', '--order_cost',
                         dest='cost',
                         help='order cost',
                         default=None,
                         type="float")

    optparser.add_option('-e', '--shortage_cost',
                         dest='shortage',
                         help='Shortage costs',
                         default=None,
                         type="float")

    optparser.add_option('-p', '--percent',
                         dest='percent',
                         help='inventory cost percent',
                         default=None,
                         type="float")

    optparser.add_option('-q', '--inventory_cost',
                     dest='per_invent_cost',
                     help='per production cost',
                     default=None,
                     type="float")

    optparser.add_option('-m', '--inventory_fee',
                     dest='inven_fee',
                     help='inventory fee, equal p * q',
                     default=None,
                     type="float")

    optparser.add_option('-o',
                     dest='o',
                     help='',
                     default=None,
                     type="float")

    optparser.add_option('-s',
                     dest='B1',
                     help='',
                     default=None,
                     type="float")

    optparser.add_option('-t', 
                     dest='t',
                     help='p s_Q_Normal',
                     default=None,
                     type="float")

    optparser.add_option('-j',
                     dest='j',
                     help='',
                     default=None,
                     type="float")

    optparser.add_option('-w',
                     dest='w',
                     help='',
                     default=None,
                     type="float")

    optparser.add_option('-x', '--advance_order_days',
                     dest='advance_days',
                     help='advance order days before shortage',
                     default=None,
                     type="float")

    optparser.add_option('-y', '--year_work_day',
                     dest='year_days',
                     help='working days in a year',
                     default=None,
                     type="float")

    optparser.add_option('--price',
                     dest='price',
                     help='discount price list',
                     default=None,
                     type="string")

    optparser.add_option('--number',
                     dest='number',
                     help='discount need number',
                     default=None,
                     type="string")

    optparser.add_option('-z', '--model',
                         dest='model',
                         help='model',
                         default=None,
                         type="string")

    (options, args) = optparser.parse_args()
    
    inventory_logger.info("Start,getting parameters")
    method = options.model
    d = options.need
    a = options.production
    b = options.prepare
    c = options.cost
    e = options.shortage
    m = options.inven_fee
    p = options.percent
    q = options.per_invent_cost
    o = options.o
    s = options.B1
    t = options.t
    j = options.j
    w = options.w
    x = options.advance_days
    y = options.year_days
    price = options.price
    number = options.number
    if price:
        price = eval(price)
    if number:
        number = eval(number)

    if p and q:
        m = p*q/100.0
 
    name,value = main(method, d, a, b, c, e, m, o, s, t, j, w, x, y, price, number)
    
    inventory_logger.info("Computing...")
   
    full_name = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

    inventory_logger.info("Computing done, save data into file")
    result_name = full_name +os.path.sep+"pythonFiles"+os.path.sep+"sd_inventory_result.txt"

    f = open(result_name, "w+")
 
    for i in xrange(len(name)):
        f.write(dicts[name[i]]+"\t"+str(float("%0.4f"%value[i]))+"\n")
    f.close()

