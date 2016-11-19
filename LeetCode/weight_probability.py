#coding=utf-8 

import bisect
import random

class WeightRandom:
  def __init__(self, items):
    weights = [w for _,w in items]
    self.goods = [x for x,_ in items]
    self.total = sum(weights)
    self.acc = list(self.accumulate(weights))
  
  def accumulate(self, weights):#累和.如accumulate([10,40,50])->[10,50,100] 
    cur = 0
    for w in weights:
      cur = cur+w
      yield cur
  
  def __call__(self): 
    return self.goods[bisect.bisect_right(self.acc , random.uniform(0, self.total))]
  
wr = WeightRandom([('iphone', 0), ('ipad', 0), ('itouch', 90)])
print(wr())