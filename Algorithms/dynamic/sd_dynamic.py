def extended_bottom_up_cut_rod(p,n):
    r = [0 for i in range(n+1)]
    s = [0 for i in range(n+1)]
    r[0] = 0

    for j in range(1, n+1):
        q = -float("inf")
        for i in range(1,j+1):
            #print(p[i] + r[j-i])
            if q < p[i] + r[j-i]:
                q = p[i] + r[j-i]
               # print(q)
                s[j] = i
            r[j] = q
    print(r,s)
    return(r,s)

def show(p,n):
    r,s = extended_bottom_up_cut_rod(p,n)
    while n > 0:
        print(s[n])
        n = n-s[n]

show([0,1,5,8,9,10,17,17,20,24,30],10)
show([0,6,3,5,4,6],5)