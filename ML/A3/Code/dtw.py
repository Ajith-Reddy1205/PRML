import math
import random
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from sklearn import metrics

cur_dir = os.getcwd() 

def normalise(lst):
	mn = min(lst)
	mx = max(lst)
	for i in range(len(lst)):
		lst[i] = (lst[i] - mn)/(mx - mn)
	return lst

def getTrainData_letter_norm(l):
	ret = []
	for x in sorted(os.listdir(cur_dir + "/TeluguLetter/" + l + "/train")):
		points_x = []
		points_y = []
		points_norm = []
		data_pointer = open(cur_dir + "/TeluguLetter/" + l + "/train/" + x,'r')
		temp_list = data_pointer.read()
		dim = int(temp_list.split(" ")[0])
		data = [float(i) for i in temp_list.split(" ")[1:-1]]
		for i in range(dim) : 
			points_x.append(data[2*i])
			points_y.append(data[2*i + 1])
		norm_x = normalise(points_x)
		norm_y = normalise(points_y)
		for i in range(dim) : 
			points_norm.append([norm_x[i], norm_y[i]])
		ret.append(points_norm)
	return ret

def getDevData_letter_norm(l):
	ret = []
	for x in sorted(os.listdir(cur_dir + "/TeluguLetter/" + l + "/dev")):
		points_x = []
		points_y = []
		points_norm = []
		data_pointer = open(cur_dir + "/TeluguLetter/" + l + "/dev/" + x,'r')
		temp_list = data_pointer.read()
		dim = int(temp_list.split(" ")[0])
		data = [float(i) for i in temp_list.split(" ")[1:-1]]
		for i in range(dim) : 
			points_x.append(data[2*i])
			points_y.append(data[2*i + 1])
		norm_x = normalise(points_x)
		norm_y = normalise(points_y)
		for i in range(dim) : 
			points_norm.append([norm_x[i], norm_y[i]])
		ret.append(points_norm)
	return ret


def getTrainData_audio(l):
	ret = []
	for x in os.listdir(cur_dir + "/Audio/" + l + "/train"):
		if x.endswith(".mfcc"):
			data_pointer = open(cur_dir + "/Audio/" + l + "/train/" + x,'r')
			data = data_pointer.read()
			row = data.split("\n")
			dim = row[0].split(" ")
			dim = [int(i) for i in dim]
			dt = []
			for i in range(1,dim[1]+1):
				temp = row[i][1:].split(" ")
				temp = [float(j) for j in temp]
				dt.append(temp)
			ret.append(dt)
	return ret

def getDevData_audio(l):
	ret = []
	for x in os.listdir(cur_dir + "/Audio/" + l + "/dev"):
		
		if x.endswith(".mfcc"):
			data_pointer = open(cur_dir + "/Audio/" + l + "/dev/" + x,'r')
			data = data_pointer.read()
			row = data.split("\n")
			dim = row[0].split(" ")
			dim = [int(i) for i in dim]
			dt = []
			for i in range(1,dim[1]+1):
				temp = row[i][1:].split(" ")
				temp = [float(j) for j in temp]
				dt.append(temp)
			ret.append(dt)
	return ret

def getDist(x,y):

	n = len(x)
	dist = 0.0
	for i in range(n):
		dist += (x[i] - y[i])**2
	return dist

def dtw(x,y):
	n = len(x)
	m = len(y)

	dp = np.zeros((n+1,m+1))

	for i in range(n+1):
		for j in range(m+1):
			dp[i][j] = 1e16
	# print(dp)
	dp[0][0] = 0

	for i in range(1,n+1):
		for j in range(1,m+1):
			cost = getDist(x[i-1], y[j-1])**0.5
			dp[i][j] = cost + min(dp[i-1][j], min(dp[i][j-1],dp[i-1][j-1]))

	return dp[n][m]

def testData_letter(train_data,dev_data,k,ax):
	right = 0
	wrong = 0
	score = []
	y_test = []
	score = []

	confusion = np.zeros((5,5))
	for i in range(5):
		# print(i)
		for j in range(len(dev_data[i])):
			cur = 1e16
			cl = -1
			xx = []
			for ii in range(5):
				temp = []
				for jj in range(len(train_data[ii])):
					ret = dtw(train_data[ii][jj], dev_data[i][j])
					temp.append(ret)
				temp.sort()
				ret = sum(temp[:k])/k
				if ret < cur:
					cur = ret
					cl = ii
				xx.append(-ret)
			if cl != i:
				wrong += 1
				# print(j)
			else:
				right += 1
			y_test.append(i)
			score.append(xx)
			confusion[i][cl] += 1

	# roc_plots(y_test, score, str(k), 5,ax)
	print("Right Predicrions  = " + str(right), "Wrong Predictions = " + str(wrong), "k = " + str(k))
	# plotconfusion(confusion, "dtw_confusion_letter.png")

def testData_audio(train_data,dev_data,k):
	right = 0
	wrong = 0
	score = []
	y_test = []
	score = []

	confusion = np.zeros((5,5))
	for i in range(5):
		# print(i)
		for j in range(len(dev_data[i])):
			cur = 1e16
			cl = -1
			xx = []
			for ii in range(5):
				temp = []
				for jj in range(len(train_data[ii])):
					ret = dtw(train_data[ii][jj], dev_data[i][j])
					temp.append(ret)
				temp.sort()
				ret = sum(temp[:k])/k
				if ret < cur:
					cur = ret
					cl = ii
				xx.append(ret)
			if cl != i:
				wrong += 1
				# print(j)
			else:
				right += 1
			y_test.append(i)
			score.append(xx)
			confusion[i][cl] += 1

	# roc_plots(y_test, score, "k = " + str(k), 5)
	print("Right Predicrions  = " + str(right), "Wrong Predictions = " + str(wrong), "k = " + str(k))
	# plotconfusion(confusion, "dtw_confusion_audio.png")

def roc_plots(y,allclasses,case,class_count):
	th=[]
	for i in allclasses:
		for j in i:
			th.append(j)
	th.sort()
	tpr=[]
	fpr=[]
	fnr = []
	rates=[]
	for threshhold in th:
		(tp,fp,fn,tn)=(0.0,0.0,0.0,0.0)
		for i in range(len(y)):
			for j in range(class_count):
				if(allclasses[i][j]>=threshhold):#predict positive
					if(y[i]==j):
						tp+=1
					else:
						fp+=1
						
				else:
					if(y[i]==j):
						fn+=1
					else:
						tn+=1
		rates.append([tp/(tp+fn),fp/(fp+tn)])
		fnr.append(fn/(tp+fn))
	tpr=[i[0] for i in rates]
	fpr=[i[1] for i in rates]
	plt.xlabel("False Positive Rate(FPR)")
	plt.ylabel("True Positive Rate(TPR)")
	plt.plot(fpr,tpr,label = case)
	return (fpr,fnr)

def plotconfusion(confusion,st):
    df_cm = pd.DataFrame(confusion, index = [i for i in "12345"], columns = [i for i in "12345"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(st)

def plotROC(scores ,st):
	print(scores)
	fig,axis = plt.subplots(1,figsize=(10,8))
	temp_list = []
	for score in scores: 
		temp_list.append(score[0])
	scores = sorted(scores)
	temp_list = sorted(temp_list)
	TPR = []  # TP/(TP+FN)
	FNR = []  # FN/(FN+TP)
	FPR = []  # FP/(FP+TN)

	for threshold in temp_list : 
	    tp = 0
	    tn = 0
	    fp = 0
	    fn = 0 
	    for data in scores : 
	        if(data[0] >= threshold): 
	            if(data[1] == data[2]): 
	                tp += 1
	            else : 
	                fp += 1
	        else : 
	            if(data[1] == data[2]): 
	                fn += 1
	            else :
	                tn += 1
	    TPR.append(float(tp/(tp+fn)))
	    FNR.append(float(fn/(fn+tp)))
	    FPR.append(float(fp/(fp+tn)))

	axis.plot(FPR,TPR)
	axis.set_title(f"ROC Curve")
	axis.set_xlabel("False Positive Rate(FPR)")
	axis.set_ylabel("True Positive Rate(TPR)")
	fig.savefig(st)

def plotDET(scores,st): 
    fig,axis = plt.subplots(1,figsize=(10,8))
    temp_list = []
    for score in scores : 
        temp_list.append(score[0])
    scores = sorted(scores)
    temp_list = sorted(temp_list)
    TPR = []  # TP/(TP+FN)
    FNR = []  # FN/(FN+TP)
    FPR = []  # FP/(FP+TN)

    for threshold in temp_list : 
        tp = 0
        tn = 0
        fp = 0
        fn = 0 
        for data in scores : 
            if(data[0] >= threshold) : 
                if(data[1] == data[2]) : 
                    tp += 1
                else : 
                    fp += 1
            else : 
                if(data[1] == data[2]) : 
                    fn += 1
                else :
                    tn += 1
        TPR.append(float(tp/(tp+fn)))
        FNR.append(float(fn/(fn+tp)))
        FPR.append(float(fp/(fp+tn)))

    blah1 = norm.ppf(FPR)
    blah2 = norm.ppf(FNR)
    axis.plot(blah1,blah2)
    axis.set_title(f"DET Curve")
    axis.set_xlabel("False Alarm Rate")
    axis.set_ylabel("Missed Detection Rate")

    values = axis.get_yticks()
    axis.set_yticklabels(["{:.0%}".format(y) for y in normalise(values)])
    values = axis.get_xticks()
    axis.set_xticklabels(["{:.0%}".format(x) for x in normalise(values)])

    fig.savefig(st)

if __name__ == "__main__" :
	
	print('DIGIT-AUDIO')
	lst = ['1','3','4','8','o']
	train_data = []
	for l in lst:
		ret = getTrainData_audio(l)
		train_data.append(ret)

	dev_data = []
	for l in lst:
		ret = getDevData_audio(l)
		dev_data.append(ret)

	for k in [1,3,5,10,12]:
		testData_audio(train_data, dev_data,k)

	ax = plt.axes()
	ax.set_xlabel("False Positive Rate(FPR)")
	ax.set_ylabel("True Positive Rate(TPR)")

	print('TELUGU LETTER-HANDWRITTEN')
	lst = ['a', 'ai', 'bA', 'dA', 'tA']
	train_data = []
	for l in lst:
		ret = getTrainData_letter_norm(l)
		train_data.append(ret)

	dev_data = []
	for l in lst:
		ret = getDevData_letter_norm(l)
		dev_data.append(ret)

	for k in [1,3,5,10,12]:
		testData_letter(train_data, dev_data, k)