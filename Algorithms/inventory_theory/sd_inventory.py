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
#import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton
from scipy.optimize import fsolve
from optparse import OptionParser

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PAR_DIR = os.path.dirname(BASE_DIR)

# queue_logger = logging.getLogger('SD_API.Method.sd_queue')
# queue_logger.setLevel(logging.INFO)
# fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_Queue.log")
# fh.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
# fh.setFormatter(formatter)
# queue_logger.addHandler(fh)

"""
1.经济订货批量模型: EOQ

2.经济生产批量模型: EPQ

3.允许缺货的经济订货批量模型: AEOQ

4.允许缺货的经济生产批量模型: AEPQ

5.折扣: EODQ (不允许缺货，供应速率无穷大)

6.折扣（不允许缺货，供应速率一定）EOPQ

7.折扣（允许缺货，供应速率无穷大)AEODQ

8.折扣（允许缺货，供应速率一定)AEOPQ

9.随机需求的单一周期存贮模型: RR

10.随机需求的订货批量--在订货点模型
"""
"""
需求量/每年：d   必须
每年的生产量：a
生产准备费: b
订货费: c    必须
缺货费: e 
成本&百分比: 存贮物品成本q, 百分比p
存贮费:m 或者 m=p*q
年总工作日:y
订货提前期:x
"""

"""
最优订货量 max_need

每年存贮成本 cost_per_year

每年订货成本

成本总计

最大存贮水平

平均存贮水平

再订货点

每年订货次数

周期(天数）
"""
def EOQ(d, c, p, q, m, x, y):
    if p and q:
        m = p*q/100.0

    best_order = math.sqrt(2.0*d*c/m)
    order_cost = inventory_cost = math.sqrt(d*m*c/2.0)
    max_inventory = best_order
    ave_inventory = max_inventory/2.0
    total_fee = math.sqrt(2.0*d*c*m)
    advance_point = x*d/y
    days = y/(d/best_order)
    times = math.sqrt(m*d/(2.0*c))
    print(best_order,inventory_cost,order_cost,total_fee,max_inventory,ave_inventory,advance_point,times,days)

def EPQ(d, a, b, p, q, m, x, y):
    if p and q:
        m = p*q/100.0

    best_product = math.sqrt(2*d*b/((1-d/a)*m))
    product_cost = inventory_cost = b * d/best_product
    total_fee = product_cost + inventory_cost
    max_inventory = math.sqrt(2*d*b*(1-d/a)/m)
    ave_inventory = max_inventory/2.0
    advance_point = x*d/y
    days = y/(d/best_product)
    times = d/best_product

    print(best_product, product_cost, inventory_cost, total_fee, max_inventory, ave_inventory, advance_point, times, days)

def AEOQ(d, c, e, p, q, m, x, y):
    if p and q:
        m = p*q/100.0

    best_order = math.sqrt(2.0*d*c*(m+e)/(e*m))
    max_shortage = best_order*(m/(m+e))
    shortage_cost = max_shortage**2*e/(2*best_order)
    order_cost = d/best_order*c
    inventory_cost = m*(best_order-max_shortage)**2/(2*best_order)
    total_fee = order_cost + inventory_cost + shortage_cost
    max_inventory = best_order-max_shortage
    ave_inventory = (best_order-max_shortage)**2/(best_order*2.0)
    advance_point = x*d/y - max_shortage
    days = y/(d/best_order)
    times = d/best_order
    
    print(best_order, inventory_cost, order_cost, shortage_cost, total_fee, max_inventory, ave_inventory, max_shortage, advance_point, times, days)

def AEPQ(d, a, b, e, p, q, m, x, y):
    if p and q:
    	m = p*q/100.0
    
    best_product = math.sqrt(2*d*b*(m+e)/(m*e*(1-d/a)))
    shortage_cost = (e*best_product*(1-d/a)*(m/(m+e))**2)/2
    inventory_cost = (m*best_product*(1-d/a)*(e/(m+e))**2)/2
    product_cost = b*d/best_product
    
    max_shortage = math.sqrt(2*b*d*m*(1-d/a)/(e*(m+e)))
    total_fee = math.sqrt(2*d*m*e*b*(1-d/a)/(m+e))
 
    ave_inventory = (best_product*(1-d/a)-max_shortage)**2/(2*best_product*(1-d/a))
    ave_shortage = max_shortage**2/(2*best_product*(1-d/a))
    max_inventory = ave_inventory*2+max_shortage
    days = y*best_product/d 

    advance_point = x*d/y - max_shortage
    #times = d/best_product
    print(best_product, inventory_cost, shortage_cost, product_cost,ave_inventory, total_fee,max_inventory,max_shortage, advance_point, ave_shortage,days)


# example 1 in yfj's ppt No.19 page
def EODQ(d, c, p, q, m, price, number, x, y):
    if p and q:
		m = p * q

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
    print("best_order, total_fee, inventory_cost, order_cost,advance_point, max_inventory, ave_inventory,days, times")
    print(best_order, total_fee, inventory_cost, order_cost,advance_point, max_inventory, ave_inventory,days, times)


# example 2 in yfj's ppt No.20 page
def AEODQ(d, c, e, p, q, m, price, number, x, y):
    if p and q:
    	m = p * q

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

    print(best_order, total_fee,max_shortage, shortage_cost,ave_shortage, inventory_cost, order_cost,advance_point, max_inventory, ave_inventory, days, times)

# example 3 in yfj's ppt No.21
def EOPQ(d, a, c, p, q, m, price, number, x, y):
    if p and q:
    	m = p * q
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

    print(best_order, total_fee, inventory_cost, order_cost,advance_point, max_inventory, ave_inventory,days, times)

# example 4 in yfj's ppt No.22
def AEOPQ(d, a, c, e, p, q, m, price, number, x, y):
    if p and q:
        m = p * q
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

    print(best_order, total_fee,max_shortage, ave_shortage, inventory_cost, order_cost,advance_point, max_inventory,ave_inventory, days, times)



# EOQ(100, 20, 20, 3, 0, 3.0, 250.0)
# EPQ(200.0, 300.0, 35.0, 30.0, 4.0, 0, 20.0, 250.0)
# AEOQ(200.0, 40.0, 20.0, 20.0, 10.0, 0, 10.0, 250.0)
#AEPQ(100.0, 130.0, 20.0, 10.0, 20.0, 7.0, 0, 10.0, 250.0)

# EODQ(500.0, 50, 0, 0, 20, [40, 39, 38, 37], [100.0,200.0,300.0], 10, 250)

# AEODQ(500.0, 50.0, 30.0, 0, 0, 20.0, [40, 39, 38, 37], [100, 200, 300], 10, 250)

# EOPQ(500.0, 600.0, 50.0, 0.0, 0.0, 20.0, [40, 39, 38, 37], [100, 200, 300], 10, 250)

# AEOPQ(500.0, 600.0, 50.0, 30.0, 0, 0, 20.0, [40, 39, 38, 37], [100, 200, 300], 10, 250)


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

    order_cost =  C0*sum([(best_order-number[i])*frequency[i] for i in range(index+1)]) + Cu*sum([(number[i]-best_order)*frequency[i] for i in range(index+1, len(frequency))])
    total_fee = sum([((p-s+h)*number[i] - C0*best_order)*frequency[i] for i in range(index+1)]) + sum([(Cu*best_order-b*number[i])*frequency[i] for i in range(index+1, len(frequency))])
    print(C0, Cu, SL, best_order, order_cost+a, total_fee-a)

#RandomRequestPossion(300,0.7, 1.0, 0.1, 0, 0, 0, [60, 100, 140, 170, 190, 210, 230, 240], [0.04, 0.1, 0.12, 0.14, 0.16, 0.2, 0.14, 0.1])
#RandomRequestPossion(100.0, 100.0, 0.0, 180.0, 0.0, 5.0,0,0)
#RandomRequestPossion(3.0,10.0,2.0,2.0,0,2.0,0,0)

def RandomRequestUniform(a,c, b, p, s, h, distribution):
    """
       a: order_cost, c: price, p: sell_price, b:shortage_cost, s: back_price, h:inventory_cost, distribution: uniform
    """
    C0 = c-s+h
    Cu = p-c+b
    SL = Cu/(Cu+C0)
    print(C0,Cu,SL)

    diff_ave = (distribution[-1]-distribution[0])/2.0
    add_ave  = (distribution[-1]+distribution[0])/2.0
    
    best_order = distribution[0] + SL*diff_ave*2.0
    order_cost = (Cu+C0)*(SL*distribution[0]+diff_ave*SL**2) - b*add_ave
    total_fee = Cu*add_ave - (C0+Cu)*(SL*distribution[0]+diff_ave*SL**2)

    print(SL, best_order, order_cost+a, total_fee-a)

#RandomRequestUniform(27.0, 10.0, 30.0, 15.0, 0, [100.0,200.0])

def RandomRequestExponential(a,c, p, s, b, h, position, scale):
    C0 = c-s+h
    Cu = p-c+b
    SL = Cu/(Cu+C0)
    print(C0,Cu,SL)

    best_order = position - math.log(1-SL)/scale
    order_cost = (Cu+C0) * ((position+1.0/scale)*SL + (1-SL)*math.log(1-SL)/scale) - b*(position+1.0/scale)
    total_fee = Cu * (position+1.0/scale) - (Cu+C0)*((position+1.0/scale)*SL+(1-SL)*math.log(1-SL)/scale)

    print(SL, best_order, order_cost+a, total_fee-a)
#RandomRequestExponential(800.0, 1440.0, 400.0, 0, 0, 0, 0.0125)


def RandomRequestNormal(a,c, p, s, b, h, average, standard_deviation):
    C0 = c-s+h
    Cu = p-c+b
    SL = Cu/(Cu+C0)

    best_order = norm.ppf(SL, loc=average, scale=standard_deviation)
    order_cost = best_order*Cu - (Cu+C0)*best_order*SL + (Cu+C0)*(average*SL-standard_deviation*norm.pdf((best_order-average)/standard_deviation, loc=0, scale=1)) - b*average
    # print(best_order*Cu, (Cu+C0)*best_order*SL, (Cu+C0)*(average*SL-standard_deviation*norm.pdf((best_order-average)/standard_deviation)), b*average)
    
    total_fee = -(Cu+C0)*(average*SL-standard_deviation*norm.pdf((best_order-average)/standard_deviation, loc=0, scale=1)) + Cu*average
    print(C0, Cu, SL, best_order, order_cost+a, total_fee-a)

#RandomRequestNormal(100.0, 250.0, 80.0, 10.0, 0, 900.0, 110.0)

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
    
    SS = s - dl
    ave_shortage = dl + (s**2)/(2*2*dl) - s
    shortage_percent = 1 - s/(2*dl)

    order_cost = a*D/best_order + c*D
    total_inventory_cost = h*(best_order/2.0+s-dl+q*ave_shortage)
    shortage_cost = (B*D*ave_shortage+shortage_percent*B1*D)/best_order
    total_fee = order_cost + total_inventory_cost + shortage_cost
    print(B, D, dl, s, best_order, SS, ave_shortage, shortage_percent,order_cost, total_inventory_cost, shortage_cost, total_fee)

# s_Q_Uniform(1500.0, 100000.0, 2000.0, 0.8, 1000.0, 0.2, 6000.0, 0, 0.001, 10.0, 365.0, [0,1200])

def s_Q_Exponential(a, c, h, p, e, q, o, B1, w, x, y, l):
    B = p*e + q*o
    D = l
    dl = D*x/y

    i = float("inf")
    best_order = math.sqrt(2.0*D*a/h)

    while w<i:
       s = -dl*math.log(h*best_order/(q*h*best_order+B*D+B1*D/dl))
       i = math.sqrt( 2*D*(a+B*dl*exp(-s/dl)+B1*exp(-s/dl))/h ) - best_order
       best_order = math.sqrt( 2*D*(a+B*dl*exp(-s/dl)+B1*exp(-s/dl))/h )

    SS = s - dl
    ave_shortage = dl*exp(-s/dl)
    shortage_percent = exp(-s/dl)

    order_cost = a*D/best_order + c*D
    total_inventory_cost = h*(best_order/2.0+s-dl+q*ave_shortage)
    shortage_cost = (B*D*ave_shortage+shortage_percent*B1*D)/best_order
    total_fee = order_cost + total_inventory_cost + shortage_cost
    print(B, D, dl, s, best_order, SS, ave_shortage, shortage_percent,order_cost, total_inventory_cost, shortage_cost, total_fee)

# s_Q_Exponential(500.0, 100000.0, 2000.0, 0.8, 1000, 0.2, 6000.0, 0, 0.001, 10.0, 365.0, 600.0)

def s_Q_Normal(a, c, h, p, e, q, o, B1, w, x, y, average, standard_deviation):
    B = p*e + q*o
    D = average
    dl = x*average/y
    print(dl)
    i = float("inf")
    best_order = math.sqrt(2.0*D*a/h)

    standard = standard_deviation*math.sqrt(x/y)

    while w<i:
        def f(s,h,best_order,B1,D,B,q):
            return norm.cdf(s,loc=dl, scale=standard) - 1 + (h*best_order-B1*D*norm.pdf(s,loc=dl,scale=standard))/(q*h*best_order
            	+B*D)

        s = fsolve(f,x0=0, args=(h,best_order,B1,D,B,q))[0]
        i = math.sqrt(2*D/h*(a+B*(standard*norm.pdf((s-dl)/standard, loc=0, scale=1)+(dl-s)*(1-norm.cdf(s, loc = dl, scale= standard))))) - best_order
        best_order = math.sqrt(2*D/h*(a+B*(standard*norm.pdf((s-dl)/standard, loc=0, scale=1)+(dl-s)*(1-norm.cdf(s, loc = dl, scale= standard)))))

    
    SS = s - dl
    ave_shortage = standard*norm.pdf((s-dl)/standard, loc=0, scale=1)+(dl-s)*(1-norm.cdf(s, loc = dl, scale= standard))
    shortage_percent = 1-norm.cdf(s,loc=dl, scale=standard)

    order_cost = a*D/best_order + c*D
    total_inventory_cost = h*(best_order/2.0+s-dl+q*ave_shortage)
    shortage_cost = (B*D*ave_shortage+shortage_percent*B1*D)/best_order
    total_fee = order_cost + total_inventory_cost + shortage_cost
    print(B, D, standard,dl, s, best_order, SS, ave_shortage, shortage_percent,order_cost, total_inventory_cost, shortage_cost, total_fee)

s_Q_Normal(500.0, 100000.0, 2000.0, 0.8, 1000.0, 0.2, 6000.0, 0, 0.05, 10.0, 365.0, 600.0, 100.0)
# if __name__ == "__main__":
#     optparser = OptionParser()

#     optparser.add_option('-d', '--need',
#                          dest='need',
#                          help='need per year',
#                          default=None,
#                          type="float")

#     optparser.add_option('-a', '--production',
#                          dest='production',
#                          help='production per year',
#                          default=None,
#                          type="float")

#     optparser.add_option('-b', '--prepare_cost',
#                          dest='prepare',
#                          help='prepare cost per year on production',
#                          default=None,
#                          type="float")


#     optparser.add_option('-c', '--order_cost',
#                          dest='cost',
#                          help='order cost',
#                          default=None,
#                          type="float")

#     optparser.add_option('-e', '--shortage_cost',
#                          dest='shortage',
#                          help='Shortage costs',
#                          default=None,
#                          type="float")

#     optparser.add_option('-p', '--percent',
#                          dest='percent',
#                          help='inventory cost percent',
#                          default=None,
#                          type="float")

#     optparser.add_option('-q', '--inventory_cost',
#                      dest='per_invent_cost',
#                      help='per production cost',
#                      default=None,
#                      type="float")

#     optparser.add_option('-m', '--inventory_fee',
#                      dest='inven_fee',
#                      help='inventory fee, equal p * q',
#                      default=None,
#                      type="float")

#     optparser.add_option('-x', '--advance_order_days',
#                      dest='advance_days',
#                      help='advance order days before shortage',
#                      default=None,
#                      type="float")

#     optparser.add_option('-y', '--year_work_day',
#                      dest='year_days',
#                      help='working days in a year',
#                      default=None,
#                      type="float")

#     optparser.add_option('-z', '--model',
#                          dest='model',
#                          help='model',
#                          default=None,
#                          type="string")

#     (options, args) = optparser.parse_args()
    
#     queue_logger.info("Start,getting parameters")
#     model = options.model
#     d = options.need
#     a = options.production
#     b = options.prepare
#     c = options.cost
#     e = options.shortage
#     m = options.mean
#     cn = options.cu_number
