#! -*- coding:utf-8 -*-
"""
    Given an array of integers, return indices of the two numbers such that add up
    to a specific target.
    You may assume that each input would have exactly one solution

    Example:
    Given nums = [2, 7, 11, 15], target = 9,
    Because nums[0] + nums[1] = 2 + 7 = 9,
    return [0, 1].

    需找出一个列表中是否有两个数的和为一个给定的数，并返回这两个数的下标。
    需要用到python里面的字典（相当于hash表），判断第i个数前面是否有一个数的值为target - num[i]。
"""

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d = {}
        for i in range(len(nums)):
            find = target  - nums[i]
            if d.get(find, None) == None:
                d[nums[i]] = i
            else:
                return [d[find], i]

class Solution1:
    # @return a tuple, (index1, index2)
    def twoSum(self, num, target):
        tmp = {}
        for i in range(len(num)):
            if target - num[i] in tmp:
               return([tmp[target - num[i]], i] )
            else:
               tmp[num[i]] = i;

nums = [2, 5, 2, 11, 15]
target = 4
a=Solution()
print a.twoSum(nums, target)

b=Solution1()
print b.twoSum(nums, target)