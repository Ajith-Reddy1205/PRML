import numpy as np  
import matplotlib.pyplot as plt
import math
import random as rnd

def calcError(w_ml, t, fun_x, lam):

	error = np.linalg.norm(np.array(t) - np.array(fun_x))
	error = error*error
	error = error/2
	return error

def genPhi(x,degree):
	n = len(x)
	phi = []
	for x_i in x:
		poly = []
		cur = 1
		for d in range(degree+1):
			poly.append(cur)
			cur = cur * x_i
		phi.append(poly) 

	phi = np.array(phi)
	return phi

def model(x_train,x_dev,y_train,y_dev,degree,lam,flag):
	phi = genPhi(x_train,degree)
	phi_t = phi.transpose()
	t = (np.array(y_train)).transpose()
	reg_term = lam * np.identity(degree+1, dtype = float)
	phi_psuedoinv = np.dot(np.linalg.inv(np.dot(phi_t,phi) + reg_term), phi_t)
	# phi_psuedoinv = np.linalg.pinv(phi)
	w_ml = np.dot(phi_psuedoinv, t)

	train_fun_x = []
	for i in range(len(x_train)):
		train_fun_x.append(np.dot(phi[i,:], w_ml)) 

	least_sq_error_train = calcError(w_ml,y_train,train_fun_x,lam)

	phi_dev = genPhi(x_dev,degree)
	dev_fun_x = []
	for i in range(len(x_dev)):
		dev_fun_x.append(np.dot(phi_dev[i,:], w_ml)) 

	least_sq_error_dev = calcError(w_ml,y_dev,dev_fun_x,lam)

	if flag == 20 :
		n = len(x_train)
		plt.clf()
		plt.scatter(x_dev, y_dev, facecolors = "none", edgecolors = 'b')
		plt.plot(x_train, train_fun_x, color = "red")
		plt.title("Plot on test set")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.savefig("Test")

	if flag == 10 :
		n = len(x_train)
		plt.clf()
		plt.scatter(x_dev, y_dev, facecolors = "none", edgecolors = 'b')
		plt.plot(x_train, train_fun_x, color = "red")
		plt.title("Plot on development set")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.savefig("Dev")
	

	return least_sq_error_dev,least_sq_error_train

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
	for i in range(0,10):
		print(model(x_train,x_dev,y_train,y_dev,i,0,0))

def testvstrain(x_train,x_dev,y_train,y_dev):
	dev_err = []
	train_err = []
	x = []
	for i in range(0,10):
		de, te = model(x_train,x_dev,y_train,y_dev,i,0,0)
		dev_err.append(de)
		train_err.append(te)
		x.append(i)

	plt.clf()
	plt.plot(x,train_err,label = "Train set", color = "blue",marker = 'o',markerfacecolor='blue')
	plt.plot(x,dev_err,label = "Develepment Set", color = "red",marker = 'o',markerfacecolor='red')
	plt.xlabel("M")
	plt.ylabel("Least Squared Error")
	plt.savefig("Error vs M")

def findOptLam(x_train,x_dev,y_train,y_dev,degree):
	dev_err = []
	train_err = []
	x = []
	mn = 100
	mn_lam = 0
	for i in np.arange(0,1,0.001):
		de, te = model(x_train,x_dev,y_train,y_dev,degree,i,0)
		dev_err.append(de)
		train_err.append(te)
		x.append(i)
		if de + te < mn:
			mn = de + te 
			mn_lam = i

	# print(mn,model(x_train,x_dev,y_train,y_dev,degree,0.006))
	# plt.clf()
	# plt.plot(x,train_err,label = "Train set", color = "blue")
	# plt.plot(x,dev_err,label = "Develepment Set", color = "red")
	# plt.xlabel("Lambda")
	# plt.ylabel("Least Squared Error")
	# plt.savefig("Error vs Lambda(M=7,N=40)")

	return mn_lam


def lamvsplot(x_train,x_dev,y_train,y_dev):
	n = len(x_train)
	rand = []
	for j in range(40):
		rand.append(rnd.randint(0,n-1))
	rand.sort()
	x_train_rand = []
	y_train_rand = []
	for j in rand:
		x_train_rand.append(x_train[j])
		y_train_rand.append(y_train[j])
	for i in {0,0.05,0.1,0.2,0.5,1}:
		model(x_train_rand,x_dev,y_train_rand,y_dev,4,i,0)

	findOptLam(x_train_rand,x_dev,y_train_rand,y_dev)


# Train data file
f = open('1d_team_14_train.txt', 'r+')
D1 = f.read()
# print(D1)
f.close()

x_train = []
y_train = []
s = ""
for p in D1:
	if p == ' ' : 
		x_train.append(float(s))
		s = ""
	elif p == '\n' :
		y_train.append(float(s))
		s = ""
	else :
		s = s + p 

n = len(x_train)

# Dev data file
f = open('1d_team_14_dev.txt', 'r+')
D1 = f.read()
# print(D1)
f.close()

x_dev = []
y_dev = []
s = ""
for p in D1:
	if p == ' ' : 
		x_dev.append(float(s))
		s = ""
	elif p == '\n' :
		y_dev.append(float(s))
		s = ""
	else :
		s = s + p 

mn = 100000
mn_deg = -1
for i in range(0,8):
	de,te = model(x_train,x_dev,y_train,y_dev,i,0,0)
	if de + te <= mn:
		mn = de + te
		mn_deg = i
# print(mn_deg)

lam = findOptLam(x_train,x_dev,y_train,y_dev,mn_deg)

print("Degree : " + str(mn_deg))
print("Lambda : " + str(lam))


# test data file
f = open('1d_team_14_dev.txt', 'r+')
D1 = f.read()
# print(D1)
f.close()

x_test = []
y_test = []
s = ""
for p in D1:
	if p == ' ' : 
		x_test.append(float(s))
		s = ""
	elif p == '\n' :
		y_test.append(float(s))
		s = ""
	else :
		s = s + p 

dev_err, train_err = model(x_train,x_dev,y_train,y_dev,mn_deg,lam,10)
test_err, train_err = model(x_train,x_test,y_train,y_test,mn_deg,lam,20)

print("Train Error : " + str(train_err))
print("Dev Error : " + str(dev_err))
print("Test Error : " + str(test_err))