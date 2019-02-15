import numpy as np
import matplotlib.pyplot as plt
t=np.linspace(0,2,100)
n1=np.zeros((2,100))
n2=np.zeros((2,100))
p=np.zeros((2,100))

for i in range(100):
  n1[0,i]=t[i]
  n2[0,i]=1
  p[0,i]=3*(t[i])
  n1[1,i]=2
  n2[1,i]=2*(t[i])
  p[1,i]=3
  t1=np.array([n1[0,i],n1[1,i]])
  t2=np.array([n2[0,i],n2[1,i]])
  P=np.array([p[0,i],p[1,i]])

A=np.vstack([t1,t2]).T
x=np.matmul(np.linalg.inv(A),P)

plt.plot(x[0],x[1],'o')
plt.grid()
plt.axis('equal')
plt.show()
