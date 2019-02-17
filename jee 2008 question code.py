import numpy as np
import matplotlib.pyplot as plt

def dvec(A,B):
    return B-A

def nvec(A,B):
    return np.dot(omat,dvec(A,B))

def midpt(A,B):
    return (A+B)/2.0
   
def line_int(n1,p1,n2,p2):
    A = np.linalg.inv(np.vstack((n1,n2)))
    p = np.zeros((2, 1))
    p[0] = p1
    p[1] = p2
    return np.matmul(A,p)

omat = np.array([[0,-1],[1,0]])
   
A = np.array([4,1])
B = np.array([6,5])


C = midpt(A,B)


n1 = np.array([4,1])
n2 = dvec(A,B)
p1 = 16
p2 = np.dot(n2,C)

O = line_int(n1,p1,n2,p2)
print(O)
a= O[0,0]
b= O[1,0]
D = np.array([a, b])


r = np.linalg.norm(D - A)
print(r)

len = 1000
t = np.linspace(0, 2*np.pi, 1000)
x = np.zeros((len, 2))

for i in range(len):
	cir = np.array([a + r*np.cos(t[i]), b + r*np.sin(t[i])])
	x[i, :] = cir

plt.plot(x[:, 0], x[:, 1], label = '$Circle$')

K = np.array([4,1])
N = np.matmul(omat,K)
k = np.linalg.norm(N)
y = np.zeros((2, len))
l = np.linspace(-(r+1)/k,(r+1)/k,len)
x_O = np.zeros((2, len))
for j in range(len):
	line = D + l[j]*N
	x_O[:,j] = line.T

plt.plot(x_O[0, :], x_O[1, :], label = '$Given - line$')

plt.plot(4,1,'o')
plt.text(4.1, 1.1, 'A')
plt.plot(6,5,'o')
plt.text(6.1, 5.1, 'B')
plt.text(-0.43,- 2.7, 'radius = 3.16')

plt.plot(O[0,0],O[1,0],'o')
plt.text(D[0] - 0.5, D[1] + 0.5, '$Centre$')

plt.axis('equal')
plt.title('Plot of the circle')
plt.legend(loc = 'best')

plt.grid()
plt.savefig('Circle.png')
plt.show()
