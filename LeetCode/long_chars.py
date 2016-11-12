#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 输入一串任意顺序的字符串，返回其中从小到大最长的一组字符串
# 如sdabcdfab,返回adcdf;如果有多个符合的，则返回第一个
s = input("Input a string(lowcase) please:") #if python distribution is 2.x, use raw_input
lists = []
for i in range(len(s)):
    if i+1 == len(s):break
    if s[i] > s[i+1]: lists.append(i)

lists.append(len(s))

list1 = ([-1] +lists)
x = zip(list1,lists)

result = [s[(p+1):q+1] for p,q in x]
x = sorted(result, key=len)

for i in x:
    if len(i) == len(x[-1]):
        print(i)
        break

