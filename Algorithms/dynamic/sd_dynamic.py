# -*- coding:utf-8 -*-

from itertools import product

# 0-1 knapsack
def knapsack(w, v, c):                         # Returns solution matrices
    n = len(w)                                 # Number of available items
    m = [[0]*(c+1) for i in range(n+1)]        # Empty max-value matrix
    P = [[False]*(c+1) for i in range(n+1)]    # Empty keep/drop matrix
    for k in range(1,n+1):                     # We can use k first objects
        i = k-1                                # Object under consideration
        for r in range(1,c+1):                 # Every positive capacity
            m[k][r] = drop = m[k-1][r]         # By default: drop the object
            if w[i] > r: continue              # Too heavy? Ignore it
            keep = v[i] + m[k-1][r-w[i]]       # Value of keeping it
            m[k][r] = max(drop, keep)          # Best of dropping and keeping
            P[k][r] = keep > drop              # Did we keep it?
    return m, P                                # return full results

# unbounded knapsack
def unbounded_knapsack(w, v, c):
    m = [0]
    for r in range(1,c+1):
        val = m[r-1]
        for i, wi in enumerate(w):
            if wi > r: continue
            val = max(val, v[i] + m[r-wi])
        m.append(val)
    print m
    return m[c]

# multiple kanspack
def multiple_knapsack(n,c,w,v,amount):  
    res=[[0 for j in range(c+1)] for i in range(n+1)]

    for i in range(1,n+1):  
        for j in range(0,c+1):  
            res[i][j]=res[i-1][j]
            for k in xrange(0,min(j/w[i-1],amount[i-1])+1):
                if j>=k*w[i-1] and res[i][j]<=res[i-1][j-k*w[i-1]]+k*v[i-1]:  
                    res[i][j]=res[i-1][j-k*w[i-1]]+k*v[i-1]
    print res[n][c]                       
    return res[n][c] 



n=6  
c=16
w=[3,4,5,6,7,7]  
v=[6,7,8,9,10,10]
amount=[1,1,1,1,1,1]

multiple_knapsack(n,c,w,v,amount)

# w = [3,2,5]
# v = [60,40,60]
# c = 12

# unbounded_knapsack(w,v,c)

# w = [0,3,4,5,6,7,7]
# v = [0,6,7,8,9,10,10]
# c = 16
# m, P = knapsack(w, v, c)

# n = len(w)
# max_value = m[n][c]


# # show
# lists = product(range(2),repeat=n)
# for x in lists:
#     temp = []
#     # z = []
#     # results = []
#     # sumv = 0
#     for i,j in enumerate(x):
#         if j != 0: 
#             temp.append(v[i])
#     if x[0] != 1 and max_value==sum(temp):
#         print x