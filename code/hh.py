import numpy as np
import matplotlib.pyplot as plt
from math import pi
from math import cos
from math import sin
M = 200 # 声子数空间
Z = 50  # 行走步数
n = 0   #初始位置
theta_1 = 3*pi/8
theta_2 = -0.5
def walk(N,n):
	up_right = np.zeros(M)
	down_left = np.zeros(M)
	aux=np.zeros(M)
	down_left[n] = 1
	for i in range(0,N):
			for j in range (0,M):    #R1
				Up = up_right[j]
				Down = down_left[j]
				up_right[j] = np.cos(theta_1/2) * Up - np.sin(theta_1/2) * Down
				down_left[j] = np.sin(theta_1/2) * Up + np.cos(theta_1/2) * Down
			for j in range (0,M):    #U1   
				c=down_left[j]
				down_left[j]=aux[j]
				aux[j]=c
			for j in range (0,M-1):   #red
				c=up_right[j]
				up_right[j]=down_left[j+1]
				down_left[j+1]=c
			for j in range (0,M):     #R2
				Aux = aux[j]
				Down = down_left[j]
				aux[j] = np.cos(theta_2/2) * Aux - np.sin(theta_2/2) * Down
				down_left[j] = np.sin(theta_2/2) * Aux + np.cos(theta_2/2) * Down
			for j in range (0,M):    #carrier
				c=down_left[j]
				down_left[j]=up_right[j]
				up_right[j]=c
			for j in range (1,M-1):  #blue
				c=aux[j]
				aux[j]=down_left[j-1]
				down_left[j-1]=c
	p=np.zeros(M)
	for i in range(0,M):
		p[i] = up_right[i]**2 + down_left[i]**2 + aux[i]**2
	return p


array=np.zeros((Z,M))
for i in range(0,Z):
	array[i]=walk(i+1,n)
plt.figure()
plt.xlim(0,Z)
plt.imshow(array)
plt.colorbar()

plt.figure()
t=np.arange(0,M)
p=walk(Z,n)
print('total pro',sum(p))
plt.title('with boundary')
plt.xlabel('phonons')
plt.ylabel('P')
plt.xlim(0,Z)
plt.bar(t,p,width=0.8)
plt.show()
