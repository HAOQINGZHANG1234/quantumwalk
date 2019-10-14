import numpy as np
import matplotlib.pyplot as plt
from math import pi
from math import cos
from math import sin

#N = 60 # N为随机行走的步数
M = 100 # 声子数空间
Z = 50
n = 1 # walker的初始位置
#010
theta_1p = 7*pi/8
theta_1m = 3*pi/8
theta_2 = pi/2
#011
# theta_1p = 3*pi/8
# theta_1m = 1*pi/16
# theta_2 = pi/2
#001
# theta_1p = 3*pi/8
# theta_1m = 7*pi/8
# theta_2 = pi/2
#000
# theta_1p = 9*pi/8 
# theta_1m = 5*pi/8
# theta_2 = pi/2
# 假定自旋向上往左走，自旋向下往右走
# up_left = np.zeros(M)
# down_right = np.zeros(M)
#down_right[n] = 1 # 初始化为声子数在n=100，自旋向下
#R1^1/2 T2 R2 T1 R1^1/2
def walk1(N,n):
	# n为walker的初始位置
	up_left = np.zeros(M)
	down_right = np.zeros(M)
	down_right[n] = 1 
	for i in range(0,N):
		for j in range (0,M):
			if j < n:

				Up = up_left[j]
				Down = down_right[j]
				up_left[j] = np.cos(theta_1m/4) * Up - np.sin(theta_1m/4) * Down
				down_right[j] = np.sin(theta_1m/4) * Up + np.cos(theta_1m/4) * Down
			else:
				
				Up = up_left[j]
				Down = down_right[j]
				up_left[j] = np.cos(theta_1p/4) * Up - np.sin(theta_1p/4) * Down
				down_right[j] = np.sin(theta_1p/4) * Up + np.cos(theta_1p/4) * Down

		up0 = up_left[0] # 	
		up_left = np.delete(up_left,0)
		up_left = np.insert(up_left,len(up_left)-1,0)

		for j in range (0,M):
			Up = up_left[j]
			Down = down_right[j]
			up_left[j] = np.cos(theta_2/2) * Up - np.sin(theta_2/2) * Down
			down_right[j] = np.sin(theta_2/2) * Up + np.cos(theta_2/2) * Down

		down_right = np.insert(down_right,0,up0)
		down_right = np.delete(down_right,len(down_right)-1)
		
		for j in range (0,M):
			if j < n:

				Up = up_left[j]
				Down = down_right[j]
				up_left[j] = np.cos(theta_1m/4) * Up - np.sin(theta_1m/4) * Down
				down_right[j] = np.sin(theta_1m/4) * Up + np.cos(theta_1m/4) * Down
			else:
				
				Up = up_left[j]
				Down = down_right[j]
				up_left[j] = np.cos(theta_1p/4) * Up - np.sin(theta_1p/4) * Down
				down_right[j] = np.sin(theta_1p/4) * Up + np.cos(theta_1p/4) * Down

	p=np.zeros(M)
	for i in range(0,M):
		p[i] = up_left[i]**2 + down_right[i]**2
	print('this is walk 1')
	return(p)
#R2^1/2 T1 R1 T2 R2^1/2

def walk2(N,n):
	# n为walker的初始位置
	up_left = np.zeros(M)
	down_right = np.zeros(M)
	down_right[n] = 1 
	up0 = 0
	for i in range(0,N):
		for j in range (0,M):
			Up = up_left[j]
			Down = down_right[j]
			up_left[j] = np.cos(theta_2/4) * Up - np.sin(theta_2/4) * Down
			down_right[j] = np.sin(theta_2/4) * Up + np.cos(theta_2/4) * Down

		down_right = np.insert(down_right,0,up0)
		down_right = np.delete(down_right,len(down_right)-1)
		
		for j in range (0,M):
			if j < n:

				Up = up_left[j]
				Down = down_right[j]
				up_left[j] = np.cos(theta_1m/2) * Up - np.sin(theta_1m/2) * Down
				down_right[j] = np.sin(theta_1m/2) * Up + np.cos(theta_1m/2) * Down
			else:
				
				Up = up_left[j]
				Down = down_right[j]
				up_left[j] = np.cos(theta_1p/2) * Up - np.sin(theta_1p/2) * Down
				down_right[j] = np.sin(theta_1p/2) * Up + np.cos(theta_1p/2) * Down
		up0 = up_left[0] 
		up_left = np.delete(up_left,0)
		up_left = np.insert(up_left,len(up_left)-1,0)
		down_right[0] = np.sqrt(down_right[0]**2 + up0**2)

		for j in range (0,M):
			Up = up_left[j]
			Down = down_right[j]
			up_left[j] = np.cos(theta_2/4) * Up - np.sin(theta_2/4) * Down
			down_right[j] = np.sin(theta_2/4) * Up + np.cos(theta_2/4) * Down

	p=np.zeros(M)
	for i in range(0,M):
		p[i] = up_left[i]**2 + down_right[i]**2
	print('this is walk 2')
	return(p)

array=np.zeros((Z,M))
for i in range(0,Z):
	array[i]=walk2(i+1,n)
# array_1=np.zeros((M,Z))
# array_1=np.transpose(array)

plt.figure()
plt.xlim(0,Z)
plt.imshow(array)
plt.colorbar()

plt.figure()
t=np.arange(0,M)
p=walk2(Z,n)
print('total pro',sum(p))
plt.title('with boundary')
plt.xlabel('phonons')
plt.ylabel('P')
plt.xlim(0,Z)
plt.bar(t,p,width=0.8)

plt.show()

# p=np.zeros(M)
# for i in range(0,M):
# 	p[i] = up_left[i]**2 + down_right[i]**2
# t=np.arange(0,M)
# print('total pro',sum(p))
# plt.title('with boundary')
# plt.xlabel('phonons')
# plt.ylabel('P')
# plt.xlim(0,50)
# plt.bar(t,p,width=0.8)
# plt.show()


