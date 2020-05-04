###preivew steps 


import numpy as np
import control
from matplotlib import pyplot as plt

def gen_preview_control_parameter(Zc,T,t_preview,Qe,R):
####Zc is the height of CoM
####T sampling time
####t_preview is the preview time 
#### Qe and R is the weight
	g = 9.8
	A_d = np.array([[1,T,T*T/2],[0,1,T],[0,0,1]])
	B_d = np.array([[pow(T,3)/6],[pow(T,2)/2],[T]])
	C_d = np.array([1,0,-Zc/g])
	D_d = np.array([0])
####A,B,C matrix for LQI control

	A_tilde = np.array([[1,1,T,T*T/2-Zc/g],[0,1,T,T*T/2],[0,0,1,T],[0,0,0,1]])
	B_tilde = np.array([[pow(T,3)/6-Zc*T/g],[pow(T,3)/6],[T*T/2],[T]])
	C_tilde = np.array([1,0,0,0])
	Q = np.zeros((4,4))
	Q[0,0] = Qe
	I_tilde = np.array([[1],[0],[0],[0]])
	[P,DC1,DC2] = control.dare(A_tilde,B_tilde,Q,R)
	Gi = 1/(R+np.dot(B_tilde.T,np.dot(P,B_tilde)))*np.dot(B_tilde.T,np.dot(P,I_tilde))
	F_tilde=np.vstack((np.dot(C_d,A_d),A_d))
	Gx = 1/(R+np.dot(B_tilde.T,np.dot(P,B_tilde)))*np.dot(B_tilde.T,np.dot(P,F_tilde))
	K = np.hstack((Gi,Gx))
	Gd = np.zeros((1,1+int(t_preview/T)))
	Gd[0,0]=-Gi
	##Orignial formula in the paper is to complicated, it can be reduced into equation below
	Ac_tilde = A_tilde-np.dot(B_tilde,K)
	X_tilde = - np.dot(Ac_tilde.T,np.dot(P,I_tilde))
	for i in range(1,1+int(t_preview/T)):
		Gd[0,i]=pow(R+np.dot(B_tilde.T,np.dot(P,B_tilde)),-1)*np.dot(B_tilde.T,X_tilde)
		X_tilde = np.dot(Ac_tilde.T,X_tilde)
	return A_d,B_d,C_d,Gi,Gx,Gd



def preview_control(zmpx,zmpy,T,t_preview,t_cal,A_d,B_d,C_d,Gi,Gx,Gd):
	##state space for COM in x and y direction respectively
	x_x = np.array([[0],[0],[0]])
	x_y = np.array([[0],[0],[0]])
	com_x = np.array([])
	com_y = np.array([])
	i = 0;
	for ii in range (int(t_cal/T)):
		y_x = np.dot(C_d,x_x)
		y_y = np.dot(C_d,x_y)
		e_x = zmpx[i]-y_x
		e_y = zmpy[i]-y_y
		preview_x = 0
		preview_y = 0
		j= 1
		for jj in range(ii,ii+int(t_preview/T)+1):
			preview_x = preview_x+Gd[0,j-1]*zmpx[i+j-1]
			preview_y = preview_y+Gd[0,j-1]*zmpy[i+j-1]
			j=j+1
		#input u
		u_x = -Gi[0,0]*e_x-np.dot(Gx,x_x)-preview_x
		u_y = -Gi[0,0]*e_y-np.dot(Gx,x_y)-preview_y
		x_x = np.dot(A_d,x_x)+np.dot(B_d,u_x)
		x_y = np.dot(A_d,x_y)+np.dot(B_d,u_y)
		com_x = np.hstack((com_x,np.array([x_x[0,0]])))
		com_y = np.hstack((com_y,np.array([x_y[0,0]])))
		i = i+1
		##preview horizon

	return com_x, com_y

def gen_zmp(footplace,T,t_step,foot_step):
	n_step = foot_step
	k = 0;
	zmpx = np.array([]);
	zmpy = np.array([]);
	for i in range(0,1+int(n_step*t_step/T)):
		zmpx = np.hstack((zmpx,footplace[k,0]))
		zmpy = np.hstack((zmpy,footplace[k,1]))
		if i%(int(t_step/T))==0:
			if i > 0:
				k = k+1
	return zmpx,zmpy
if __name__ == '__main__':
	Zc = 0.22
	T = 0.01
	Qe = 0.0001
	R = 0.000001
	t_preview = 1
	foot_step = 9
	t_step = 0.7
	t_cal = t_step*foot_step-t_preview-T
	footplace = np.array([[0,0,0],[0.2,0.06,0],[0.4,-0.06,0.0],[0.6,0.09,0],[0.8,-0.03,0.0],[1.3,0.09,0.0],[1.7,-0.03,0.0],[1.9,0.09,0.0],[2.0,-0.03,0.0]])
	zmpx,zmpy = gen_zmp(footplace,T,t_step,foot_step)
	A_d,B_d,C_d,Gi,Gx,Gd = gen_preview_control_parameter(Zc,T,t_preview,Qe,R)
	com_x,com_y = preview_control(zmpx,zmpy,T,t_preview,t_cal,A_d,B_d,C_d,Gi,Gx,Gd)

	x1 = range(0,631)
	x2 = range(0,529)
	plt.subplot(2,2,1)
	plt.plot(x1,zmpx)
	plt.plot(x2,com_x)
	plt.subplot(2,2,2)
	plt.plot(x1,zmpy)
	plt.plot(x2,com_y)
	plt.subplot(2,2,3)
	plt.plot(com_x)
	plt.subplot(2,2,4)
	plt.plot(com_y)
	plt.show()
