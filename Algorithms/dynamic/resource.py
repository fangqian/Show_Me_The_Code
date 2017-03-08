# -*- coding:utf-8 -*-

    # #数据1
n=3  #项目数
c=8  #资源总数
m=5  #可选择的投资策略
r=[0,2,4,6,8]
v=[[0,8,15,30,38],[0,9,20,35,40],[0,10,28,35,43]] #每种投资策略下每个项目可获得的收益

f=[[0 for i in range(m)] for j in range(n)]
p=[[0 for i in range(m)] for j in range(n)]
#j表示的是项目数，i表示的资源数，f[i][j]表示用将资源i投入到项目j所得到的最大获利，
#p[i][j]表示获得最优解时第j个项目使用的资源数的下标
for j in range(0,n):
	for i in range(0,m):
		for k in range(0,i+1):
			if f[j][i]<f[j-1][i-k]+v[j][k]:
				p[j][i]=k
				f[j][i]=f[j-1][i-k]+v[j][k]
print f
print p
print ('The maximizing revenue is:',f[n-1][m-1])
k=m-1
for j in reversed(range(0,n)):
	print ('The project is:',j+1,';The investment is:',r[p[j][k]])
	k=k-p[j][k]

