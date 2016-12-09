# -*- coding:UTF-8 -*-

"""
Algorithm Pseudocode

para:
@A array wanted to sort

INSERTION-SORT(A)
    for j = 2 to A.length
       key = A[j]
       //Insert A[j] into the sorted sequence A[1...j-1]
       i = j-1
       while i >0 and A[i]>key
           A[i+1] = A[i]
           i = i-1
       A[i+1] = key
"""

# def insert_sort(A):
#     for j in range(1,len(A)):
#         key = A[j]
#         i = j - 1
#         while i>=0 and A[i]>key:
#             A[i+1] = A[i]
#             i = i-1
#         A[i+1] = key
#         print(A)
#     return A
# A = [5,2,4,6,1,3]
# insert_sort(A)



"""
Algorithm Psuedocode

para
@A array wanted to sort

MERGE(A,p,q,r)
    n1 = q - p + 1
    n2 = r -q
    let L[1..n1+1] and R[1..n2+1] be new arrays
    for i = 1 to n1
        L[i] = A[p+i-1]
    for j = 1 to n2
        R[j] = A[q+j]

    L[n1+1] = &
    R[n2+1] = &
    i = 1
    j = 1
    for k = p to r
        if L[i] <= R[j]
           A[k] = L[i]
           i = i+1
        else A[k] = R[j]
             j = j+1

 MERGE-SORT(A,p,r)
    if p<r:
        q = [(p+r)/2]
        MERGE-SORT(A,p,q)
        MERGE-SORT(A,q+1,r)
        MERGE(A,p,q,r)
"""

 
import sys  
   
def merge(nums, first, middle, last): 
    print("****",first,middle,last) 
    ''''' merge '''  
    # 切片边界,左闭右开并且是了0为开始  
    lnums = nums[first:middle+1]   
    rnums = nums[middle+1:last+1]  
    lnums.append(sys.maxint)  
    rnums.append(sys.maxint)  
    l = 0  
    r = 0  
    for i in range(first, last+1):  
        if lnums[l] < rnums[r]:  
            nums[i] = lnums[l]  
            l+=1  
        else:  
            nums[i] = rnums[r]  
            r+=1
    print(nums)

def merge_sort(nums, first, last):
    print("====") 
    ''''' merge sort 
    merge_sort函数中传递的是下标，不是元素个数 
    '''  
    if first < last:  
        middle = (first + last)/2  
        # print(middle)
        merge_sort(nums, first, middle)

        # print("XXXX")  
        merge_sort(nums, middle+1, last) 
        # print("YYYY") 
        # print(nums)
        merge(nums, first, middle,last) 
   
if __name__ == '__main__':  
    nums = [10,8,4,-1,2,6,7,3]  
    # print 'nums is:', nums  
    merge_sort(nums, 0, 7)  
    print 'merge sort:', nums

