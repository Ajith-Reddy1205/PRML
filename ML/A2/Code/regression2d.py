import numpy as np  
import matplotlib.pyplot as plt
import math
import random as rnd
from mpl_toolkits import mplot3d

def calcError(w_ml, t, fun_x, lam):

	error = np.linalg.norm(np.array(t) - np.array(fun_x))
	error = error*error
	error = error/2
	return error

def getval(x,y,w_ml,degree):
	n = len(x)
	z = []
	for i in range(n):
		m = len(x[i])
		temp = []
		for j in range(m):
			tt = []
			qq = []
			qq.append(x[i][j])
			qq.append(y[i][j])
			tt.append(qq)
			phi,count = genPhi(tt,degree)
			# print(phi)
			temp.append(np.dot(phi[0,:],w_ml))
		z.append(temp)

	return np.array(z)


def plot3D(x_train,x_dev,y_train,y_dev,train_fun_x,dev_fun_x,degree,w_ml,st):

	x = []
	y = []
	z = y_test

	for i in x_dev:
		x.append(i[0])
		y.append(i[1])

	z = np.array(z)

	plt.clf()
	fig = plt.figure(figsize = (10, 7))
	ax = plt.axes(projection ="3d")
	 
	ax.scatter3D(x, y, z, color = "green")
	x = np.outer(np.linspace(-1, 1, 50), np.ones(50))
	y = x.copy().T
	ax.plot_surface(x,y,getval(x,y,w_ml,degree),alpha = 0.5)
	plt.title("Plot on test data")
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	ax.set_zlim((min(y_dev)-1.0,max(y_dev)+1.0))
	plt.savefig(st)

def genPhi(x,degree):
	n = len(x)
	phi = []
	count = 0
	for x_i in x:
		poly = []
		for d1 in range(degree+1):
			for d2 in range(degree+1):
				if d1 + d2 <= degree:
					poly.append(pow(x_i[0],d1)*pow(x_i[1],d2))
		phi.append(poly) 
		count = len(poly)

	phi = np.array(phi)
	return phi, count

def model(x_train,x_dev,y_train,y_dev,degree,lam,flag):
	phi,count = genPhi(x_train,degree)
	phi_t = phi.transpose()
	t = (np.array(y_train)).transpose()
	reg_term = lam * np.identity(count, dtype = float)
	phi_psuedoinv = np.dot(np.linalg.inv(np.dot(phi_t,phi) + reg_term), phi_t)
	# phi_psuedoinv = np.linalg.pinv(phi)
	w_ml = np.dot(phi_psuedoinv, t)

	train_fun_x = []
	for i in range(len(x_train)):
		train_fun_x.append(np.dot(phi[i,:], w_ml)) 

	least_sq_error_train = calcError(w_ml,y_train,train_fun_x,lam)

	phi_dev,count = genPhi(x_dev,degree)
	dev_fun_x = []
	for i in range(len(x_dev)):
		dev_fun_x.append(np.dot(phi_dev[i,:], w_ml)) 

	least_sq_error_dev = calcError(w_ml,y_dev,dev_fun_x,lam)

	if flag == 10:
		plot3D(x_train,x_dev,y_train,y_dev,train_fun_x,dev_fun_x,degree,w_ml,"Test 1")

	if flag == 15:
		plot3D(x_train,x_dev,y_train,y_dev,train_fun_x,dev_fun_x,degree,w_ml,"Test 2")
	
	return least_sq_error_dev, least_sq_error_train

def plotvsN(x_train,x_dev,y_train,y_dev):
	for i in {10,20,50,100}:
		rand = []
		for j in range(i):
			rand.append(rnd.randint(0,n-1))
		rand.sort()
		x_train_rand = []
		y_train_rand = []
		for j in rand:
			x_train_rand.append(x_train[j])
			y_train_rand.append(y_train[j])
		model(x_train_rand,x_dev,y_train_rand,y_dev,7,0,0)

def plotvsM(x_train,x_dev,y_train,y_dev):
	for i in range(0,8):
		model(x_train,x_dev,y_train,y_dev,i,0,0)

def testvstrain(x_train,x_dev,y_train,y_dev):
	dev_err = []
	train_err = []
	x = []
	for i in range(0,20):
		de, te = model(x_train,x_dev,y_train,y_dev,i,0)
		dev_err.append(de)
		train_err.append(te)
		x.append(i)

	plt.clf()
	plt.plot(x,train_err,label = "Train set", color = "blue",marker = 'o',markerfacecolor='blue')
	plt.plot(x,dev_err,label = "Develepment Set", color = "red",marker = 'o',markerfacecolor='red')
	plt.xlabel("M")
	plt.ylabel("Least Squared Error")
	plt.savefig("2d_Error vs M_new")

def findOptLam(x_train,x_dev,y_train,y_dev,degree):
	dev_err = []
	train_err = []
	x = []
	mn = 1000
	mn_lam = 0
	for i in np.arange(0,0.1,0.001):
		de, te = model(x_train,x_dev,y_train,y_dev,degree,i,0)
		dev_err.append(de)
		train_err.append(te)
		x.append(i)
		if de + te < mn:
			mn = de + te 
			mn_lam = i

	# print(mn_lam)

	# # print(dev_err, train_err)
	# plt.clf()
	# plt.plot(x,train_err,label = "Train set", color = "blue")
	# plt.title("Training set error vs lambda")
	# # plt.plot(x,dev_err,label = "Develepment Set", color = "red")
	# plt.xlabel("Lambda")
	# plt.ylabel("Least Squared Error")
	# plt.savefig("2d_Error vs Lambda(M = 4)_tr")
	# plt.clf()
	# # plt.plot(x,train_err,label = "Train set", color = "blue")
	# plt.plot(x,dev_err,label = "Develepment Set", color = "red")
	# plt.title("Develepment set error vs lambda")
	# plt.xlabel("Lambda")
	# plt.ylabel("Least Squared Error")
	# plt.savefig("2d_Error vs Lambda(M = 4)_dev")
	return mn_lam



def lamvsplot(x_train,x_dev,y_train,y_dev):
	for i in {0,0.05,0.1,1,10}:
		model(x_train,x_dev,y_train,y_dev,7,i,0)

def read(st):
	s = ""
	x = []
	y = 0.0
	for c in st:
		if c == ' ':
			x.append(float(s))
			s = ""
		elif c == "\n":
			y = float(s)
			s = ""
		else:
			s += c

	return x,y

# Train data file
f = open('2d_team_14_train.txt', 'r+')
D1 = f.read()
# print(D1)
f.close()

x_train = []
y_train = []
s = ""
for p in D1:
	s = s + p
	if p == '\n' :
		x_i, y_i = read(s)
		x_train.append(x_i)
		y_train.append(y_i)
		s = ""

n = len(x_train)


# Dev data file
f = open('2d_team_14_dev.txt', 'r+')
D1 = f.read()
# print(D1)
f.close()

x_dev = []
y_dev = []
s = ""
for p in D1:
	s = s + p
	if p == '\n' :
		x_i, y_i = read(s)
		x_dev.append(x_i)
		y_dev.append(y_i)
		s = ""

x_mix = x_train + x_dev
y_mix = y_train + y_dev

x_train_new = []
y_train_new = []
x_dev_new = []
y_dev_new = []

nums = np.random.choice([0, 1], size=2000, p=[.5, .5])

for i in range(2000):
	if nums[i] == 0 :
		x_train_new.append(x_mix[i])
		y_train_new.append(y_mix[i])
	else:
		x_dev_new.append(x_mix[i])
		y_dev_new.append(y_mix[i])

# Test data file
f = open('2d_team_14_dev.txt', 'r+')
D1 = f.read()
f.close()

x_test = []
y_test = []
s = ""
for p in D1:
	s = s + p
	if p == '\n' :
		x_i, y_i = read(s)
		x_test.append(x_i)
		y_test.append(y_i)
		s = ""


lam1 = findOptLam(x_train,x_dev,y_train,y_dev,4)
lam2 = findOptLam(x_train_new,x_dev_new,y_train_new,y_dev_new,4)


# testing using initial model

test_err, train_err = model(x_train,x_test,y_train,y_test,4,lam1,10)
dev_err, train_err = model(x_train,x_dev,y_train,y_dev,4,lam1,20)

print("MODEL 1(Initial train and dev sets) :\n")
print("degree : " + str(4))
print("lambda : " + str(lam1))
print("Train Error : " + str(train_err))
print("Dev Error : " + str(dev_err))
print("Test Error : " + str(test_err))

print("\n\n\n\n\n")
# testing using randomly picked model

test_err, train_err = model(x_train_new,x_test,y_train_new,y_test,4,lam2,15)
dev_err, train_err = model(x_train_new,x_dev_new,y_train_new,y_dev_new,4,lam2,20)

print("MODEL 2(Mixed train and dev sets) :\n")
print("degree : " + str(4))
print("lambda : " + str(lam2))
print("Train Error : " + str(train_err))
print("Dev Error : " + str(dev_err))
print("Test Error : " + str(test_err))
