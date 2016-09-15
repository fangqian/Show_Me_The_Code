#!/usr/bin/env python
# -*- coding: utf-8 -*-

def quick_sort(array):
    less =[]; greater = [];
    if len(array)<=1:
        return array
    pivot = array.pop()
    for i in array:
        if i <= pivot: less.append(i)
        else: greater.append(i)
    return quick_sort(less) + [pivot] + quick_sort(greater)

a = [1,4,6,7,8,3,11,45,113]
print quick_sort(a)
