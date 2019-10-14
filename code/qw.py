import numpy as np 
import matplotlib.pyplot as plt
 
def carry(state, theta):
	'''
	carry U operatior on state
	suppose U=exp(-i*theta/2*sigmay)
	'''
	U = np.array([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]])
	final_state = np.dot(U, state)

	return final_state

def blue_carry(state, theta):
	'''
	blue carry U on state
	suppose U=exp(-i*theta/2*sigmay)
	and |0>-->|1>,phonon number plus 1
	|1>-->|0>, phonon number minus 1
	'''
	# U operator on up and down level of state
	state_up = np.vstack((state[0], np.zeros(len(state[0]))))
	state_down = np.vstack((np.zeros(len(state[0])), state[1]))
	U = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
	carry_state_up = np.dot(U, state_up)
	carry_state_down = np.dot(U, state_down)
	# after U operation, down level minus 1
	up_state = carry_state_up[0]
	down_state_minus = carry_state_up[1]
	down_state_minus_0 = state_up[0][0]  # blue carry can not drive the phonon=0 && up level
	down_state_minus = np.delete(down_state_minus, 0)
	down_state_minus = np.insert(down_state_minus, len(down_state_minus)-1, 0)
	# after U operation, up level plus 1
	up_state_plus = carry_state_down[0]
	down_state = carry_state_down[1]
	up_state_plus = np.insert(up_state_plus, 0, down_state_minus_0)
	up_state_plus = np.delete(up_state_plus, len(up_state_plus)-1)
	# probability add up and down level
	up = up_state + up_state_plus
	down = down_state_minus + down_state
	final_state = np.vstack((up, down))

	return final_state

def red_carry(state, theta):
	'''
	red carry U on state
	suppose U=exp(-i*theta/2*sigmay)
	and |0>-->|1>,phonon number minus 1
	|1>-->|0>, phonon number plus 1
	'''
	# U operator on up and down level of state
	state_up = np.vstack((state[0], np.zeros(len(state[0]))))
	state_down = np.vstack((np.zeros(len(state[0])), state[1]))
	U = np.array([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]])
	carry_state_up = np.dot(U, state_up)
	carry_state_down = np.dot(U, state_down)
	# after U operation, up level minus 1
	up_state_minus = carry_state_down[0]
	down_state = carry_state_down[1]
	up_state_minus_0 = state_down[1][0] # red carry can not drive the phonon=0 && down level
	up_state_minus = np.delete(up_state_minus,0)
	up_state_minus = np.insert(up_state_minus,len(up_state_minus)-1,0)
	# after U operation, down level plus 1
	up_state = carry_state_up[0]
	down_state_plus = carry_state_up[1]
	down_state_plus = np.insert(down_state_plus,0,up_state_minus_0)
	down_state_plus = np.delete(down_state_plus,len(down_state_plus)-1)
	# probability add up and down level
	up = up_state + up_state_minus
	down = down_state_plus + down_state
	final_state = np.vstack((up, down))

	return final_state

def Operator(up_state, down_state, func, theta):
	'''
	operator include carry, blue and red
	'''
	state = np.vstack((up_state, down_state))
	state = func(state, theta)

	return state[0], state[1]

def one_step(state, theta_1, theta_2):
	'''
	one step of complete quantum walk
	may with the help of auxiliary level
	input state of three level
	output state of three level
	T1:up-->right
	t2:down-->left 
	T2R2T1R1
	'''
	up_state = state[0]
	down_state = state[1]
	aux_state = state[2]
	up_state, down_state = Operator(up_state, down_state, carry, theta_1)
	aux_state, down_state = Operator(aux_state, down_state, carry, np.pi)
	up_state, down_state = Operator(up_state, down_state, red_carry, np.pi)
	aux_state, down_state = Operator(aux_state, down_state, carry, theta_2)
	up_state, down_state = Operator(up_state, down_state, carry, np.pi)
	aux_state, down_state = Operator(aux_state, down_state, blue_carry, np.pi)

	state = np.vstack((up_state, down_state, aux_state))
	return state

def one_step_2(state, theta_1, theta_2):
	'''
	one step of complete quantum walk
	may with the help of auxiliary level
	input state of three level
	output state of three level
	T1:up-->right
	t2:down-->left 
	T1R1T2R2
	'''
	up_state = state[0]
	down_state = state[1]
	aux_state = state[2]
	up_state, down_state = Operator(up_state, down_state, carry, theta_2)
	aux_state, down_state = Operator(aux_state, down_state, red_carry, np.pi)
	up_state, down_state = Operator(up_state, down_state, carry, np.pi)
	aux_state, down_state = Operator(aux_state, down_state, carry, theta_1)
	up_state, down_state = Operator(up_state, down_state, blue_carry, np.pi)
	aux_state, down_state = Operator(aux_state, down_state, carry, np.pi)

	state = np.vstack((up_state, down_state, aux_state))
	return state

def one_step_3(state, theta_1, theta_2):
	'''
	one step of complete quantum walk
	may with the help of auxiliary level
	input state of three level
	output state of three level
	T1:up-->right
	t2:down-->left 
	r1T2R2T1r1   r1=sqrt(R1)
	'''
	up_state = state[0]
	down_state = state[1]
	aux_state = state[2]
	up_state, down_state = Operator(up_state, down_state, carry, theta_1/2)
	aux_state, down_state = Operator(aux_state, down_state, carry, np.pi)
	up_state, down_state = Operator(up_state, down_state, red_carry, np.pi)
	aux_state, down_state = Operator(aux_state, down_state, carry, theta_2)
	up_state, down_state = Operator(up_state, down_state, carry, np.pi)
	aux_state, down_state = Operator(aux_state, down_state, blue_carry, np.pi)
	up_state, down_state = Operator(up_state, down_state, carry, theta_1/2)

	state = np.vstack((up_state, down_state, aux_state))
	return state

def one_step_4(state, theta_1, theta_2):
	'''
	one step of complete quantum walk
	may with the help of auxiliary level
	input state of three level
	output state of three level
	T1:up-->right
	t2:down-->left 
	r2T1R1T2r2  r2=sqrt(R2)
	'''
	up_state = state[0]
	down_state = state[1]
	aux_state = state[2]
	up_state, down_state = Operator(up_state, down_state, carry, theta_2/2)
	aux_state, down_state = Operator(aux_state, down_state, red_carry, np.pi)
	up_state, down_state = Operator(up_state, down_state, carry, np.pi)
	aux_state, down_state = Operator(aux_state, down_state, carry, theta_1)
	up_state, down_state = Operator(up_state, down_state, blue_carry, np.pi)
	aux_state, down_state = Operator(aux_state, down_state, carry, np.pi)
	up_state, down_state = Operator(up_state, down_state, carry, theta_2/2)

	state = np.vstack((up_state, down_state, aux_state))
	return state

def N_step(state, theta_1, theta_2, N):
	'''
	N step of complete quantum walk
	'''
	for i in range(N):
		state = one_step(state, theta_1, theta_2)

	return state



# Parameters: M(phonon number space),n(inial place)
M = 100
# n = 1
# N = 50
# up_state = [0]*M
# down_state = [0]*M
# aux_state = [0]*M
# up_state[0] = 1/np.sqrt(2)
# down_state[1]=-1j/np.sqrt(2)
# np.array(up_state)
# np.array(down_state)
# np.array(aux_state)
# init_state = np.vstack((up_state, down_state, aux_state))

up_state = [0]*M
down_state = [0]*M
aux_state = [0]*M
up_state[0] = 0
down_state[1]=1
np.array(up_state)
np.array(down_state)
np.array(aux_state)
init_state = np.vstack((up_state, down_state, aux_state))


# step of walk
theta_1 = np.linspace(3*np.pi/2,5*np.pi/2,24)
theta_2 = np.pi/2
# state = N_step(init_state, theta_1, theta_2, N=N)

# # plot result
# P = abs(state[0])**2 + abs(state[1])**2 + abs(state[2])**2
# phonon = range(len(P))
# # plt.scatter(phonon, P, s=1000*P)
# print('total pro', sum(P))
# plt.bar(range(len(P)),P)

# plt.title('$\varepsilon$',fontsize=20)
plt.figure(figsize=(20,20))
for i,theta in enumerate(theta_1):
    state = N_step(init_state, theta, theta_2, N=50)
    P = abs(state[0])**2 + abs(state[1])**2 + abs(state[2])**2
    plt.subplot(6,4,i+1)
    plt.xlabel('theta=%f'%theta)
    plt.bar(range(20),P[:20])
    


# # plot thermal picture
# array=np.zeros((N,M))
# for i in range(0,N):
# 	state = N_step(init_state, theta_1, theta_2, N=i)
# 	P = abs(state[0])**2 + abs(state[1])**2 + abs(state[2])**2
# 	array[i]=P
# plt.figure()
# plt.xlim(0,N)
# plt.xlabel('phonon')
# plt.ylabel('step')
# plt.imshow(array)
# plt.colorbar()

# plt.figure()
# theta=np.linspace(0*np.pi,4*np.pi,100)
# P_0=[]
# P_1=[]
# P_2=[]
# P_3=[]
# P_4=[]
# for x in theta:
#     state = N_step(init_state, x, theta_2, N=N)
#     P = abs(state[0])**2 + abs(state[1])**2 + abs(state[2])**2
#     P_0=np.append(P_0,P[0])
#     P_1=np.append(P_1,P[1])
#     P_2=np.append(P_2,P[2])
#     P_3=np.append(P_3,P[3])
#     P_4=np.append(P_4,P[4])
# plt.subplot(2,3,1)
# plt.plot(theta,P_0)
# plt.subplot(2,3,2)
# plt.plot(theta,P_1)
# plt.subplot(2,3,3)
# plt.plot(theta,P_2)
# plt.subplot(2,3,4)
# plt.plot(theta,P_3)
# plt.subplot(2,3,5)
# plt.plot(theta,P_4)
plt.savefig('1_3.jpg')
plt.show()