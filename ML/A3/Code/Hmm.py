import os
import math
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import norm
from turtle import screensize
from cgitb import handler

cur_dir = os.getcwd()

def normalise(lst):
	mn = min(lst)
	mx = max(lst)
	for i in range(len(lst)):
		lst[i] = (lst[i] - mn)/(mx - mn)
	return lst

def getTrainData_audio(l):
	ret = []
	for x in sorted(os.listdir(cur_dir + "/Audio/" + l + "/train")):
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
	for x in sorted(os.listdir(cur_dir + "/Audio/" + l + "/dev")):
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

def getTrainData_letter(l):
	ret = []
	for x in sorted(os.listdir(cur_dir + "/TeluguLetter/" + l + "/train")):
		data_pointer = open(cur_dir + "/TeluguLetter/" + l + "/train/" + x,'r')
		temp_list = data_pointer.read()
		dim = int(temp_list.split(" ")[0])
		data = [float(i) for i in temp_list.split(" ")[1:-1]]
		points = []
		for i in range(dim) : 
			points.append(data[2*i : 2*i+2])
		ret.append(points)
	return ret

def getDevData_letter(l):
	ret = []
	for x in sorted(os.listdir(cur_dir + "/TeluguLetter/" + l + "/dev")):
		data_pointer = open(cur_dir + "/TeluguLetter/" + l + "/dev/" + x,'r')
		temp_list = data_pointer.read()
		dim = int(temp_list.split(" ")[0])
		data = [float(i) for i in temp_list.split(" ")[1:-1]]
		points = []
		for i in range(dim) : 
			points.append(data[2*i : 2*i+2])
		ret.append(points)
	return ret


def KMeansTrain(K, X) :
    num_samples = X.shape[0]
    num_features = X.shape[1]

    centroids = np.array([]).reshape(num_features, 0)
    for k in range(K):
        centroids = np.c_[centroids, X[random.randint(0, num_samples-1)]]

    for i in range(10):
        distances = np.array([]).reshape(num_samples, 0)
        for k in range(K):
            distances = np.c_[distances, np.sum((X - centroids[:,k])**2, axis=1)]

        cluster_numbers = np.argmin(distances, axis=1)
        
        clusters = [np.array([]).reshape(0, num_features)] * K

        for n in range(num_samples):
            clusters[cluster_numbers[n]] = np.r_["0,2", clusters[cluster_numbers[n]], X[n]]

        for k in range(K):
            centroids[:,k] = np.nanmean(clusters[k], axis=0)
    
    return centroids

def KMeansTest(X, centroids) : 
    K = centroids.shape[1]
    distances = np.array([]).reshape(1, 0)
    for k in range(K) : 
        distances = np.c_[distances, np.sum((X - centroids[:,k])**2)]
    
    cluster_number = np.argmin(distances)
    return cluster_number 

def HMMProbs(seq,aii,aij,biik,bijk):
	n = len(seq)
	num_states = len(aii)

	probs = np.zeros((num_states, n))
	probs[0][0] = aii[0] * biik[0][seq[0]]
	probs[1][0] = aij[0] * bijk[0][seq[0]]

    # Only transistions form 0th state to itslef, since there are no previous states
	for i in range(1,n) : 
		probs[0][i] = probs[0][i-1] * aii[0] * biik[0][seq[i]]

    # Transition either from the same state to itself or from the previous state
	for i in range(1,n) : 
		for j in range(1,num_states) : 
			probs[j][i] += probs[j-1][i-1] * aij[j-1] * bijk[j-1][seq[i]]
			probs[j][i] += probs[j][i-1] * aii[j] * biik[j][seq[i]] 

	sum = 0
	for i in range(num_states) :
		sum += probs[i][-1]
	return sum 

def plotROC(score_list): 
    fig,axis = plt.subplots(1,figsize=(10,8))
    k = 9
    for scores in score_list:
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

        axis.plot(FPR,TPR,label=f"k = {k}")
        k += 3
    axis.legend()
    axis.set_title(f"ROC Curve")
    axis.set_xlabel("False Positive Rate(FPR)")
    axis.set_ylabel("True Positive Rate(TPR)")
    fig.savefig("ROC_curve.png")

def plotDET(score_list): 
    fig,axis = plt.subplots(1,figsize=(10,8))
    k = 9
    for scores in score_list :
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

        x = norm.ppf(FPR)
        y = norm.ppf(FNR)
        axis.plot(x,y,label=f"k = {k}")
        k += 3
    axis.legend()
    axis.set_title(f"DET Curve")
    axis.set_xlabel("False Alarm Rate")
    axis.set_ylabel("Missed Detection Rate")

    values = axis.get_yticks()
    axis.set_yticklabels(["{:.0%}".format(y) for y in normalise(values)])
    values = axis.get_xticks()
    axis.set_xticklabels(["{:.0%}".format(x) for x in normalise(values)])

    fig.savefig("DET_curve.png")

def plotconfusion(confusion):
    df_cm = pd.DataFrame(confusion, index = [i for i in "12345"], columns = [i for i in "12345"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("img_confusion.png")

def Digits():
    score_list = []
    # Testing for 3 different values of K(number of symbols produced)
    for K in [9, 12, 15] :
        num_states = 5
        input_data = np.array([]).reshape(0,38)
        for class_name in ["1", "3", "4", "8", "o"] :
            temp_input = getTrainData_audio(class_name)
            for sample in temp_input : 
                for frame in sample : 
                    input_data = np.r_["0,2", input_data, np.array(frame)]
        # cetroids of the K clusters
        centroids = KMeansTrain(K, input_data)

        # Getting the symbols produced by each feature vector in every sample of every class
        input_labels = []
        for class_name in ["1", "3", "4", "8", "o"] : 
            temp_input = getDevData_audio(class_name)
            temp_list1 = []
            for sample in temp_input : 
                temp_list2 = []
                for frame in sample : 
                    temp_list2.append(KMeansTest(np.array(frame), centroids))
                temp_list1.append(temp_list2)
            input_labels.append(temp_list1)

        models = []
        for class_list in input_labels : 
            # Training HMM models with the sequences of symbols extraced above
            file = open("labelTrainDigit.hmm.seq","w")
            for sample in class_list : 
                for label in sample : 
                    file.write(str(label) + " ")
                file.write("\n")
            tempStr = "./train_hmm labelTrainDigit.hmm.seq 1234 "+ str(num_states) + " " + str(K) + " 0.0001"
            file.close()

            # Getting the probability values of various transitions (state to state, and production of symbol from state)
            os.system(tempStr)
            file = open("labelTrainDigit.hmm.seq.hmm","r")
            file.readline()
            file.readline()
            aii = []
            aij = []
            biik = []
            bijk = []

            for i in range(num_states) : 
                temp_list1 = [element.split("\n")[0] for element in file.readline().split("\t")]        
                temp_list2 = [element.split("\n")[0] for element in file.readline().split("\t")]

                aii.append(float(temp_list1[0]))
                aij.append(float(temp_list2[0]))

                temp_list3 = []
                temp_list4 = []
                for j in range(1,K+1) : 
                    temp_list3.append(float(temp_list1[j]))
                    temp_list4.append(float(temp_list2[j]))
                biik.append(temp_list3)
                bijk.append(temp_list4)
                
                models.append([aii, aij, biik, bijk])
                file.readline()
        
        # Checking accuracy of the model and plotting the respective graphs
        total = 0
        true = 0
        curr_class = 0
        scores = []
        conf_matrix = []
        for class_name in ["1", "3", "4", "8", "o"] :
            temp_list = [0] * 5
            samples_data = getDevData_audio(class_name)
            for sample_data in samples_data : 
                label_seq = []
                for frame in sample_data : 
                    label_seq.append(KMeansTest(np.array(frame), centroids))
                probs = []
                for model in models : 
                    probs.append(HMMProbs(label_seq, model[0], model[1], model[2], model[3]))
                temp_ind = 0
                for score in probs : 
                    scores.append((score, curr_class, int(temp_ind/num_states)))
                    temp_ind += 1
                ind = np.argmax(probs)
                temp_list[int(ind/num_states)] += 1
                if(curr_class == ind/num_states) : 
                    true += 1
                total += 1
            curr_class += 1
            conf_matrix.append(temp_list)
        print(f"Accuracy is: {100 * true / total}%")
        score_list.append(scores)
    plotROC(score_list)
    plotDET(score_list)
    plotconfusion(conf_matrix)

def Handwriting(): 
    score_list = []
    # Testing for 3 different values of K(number of symbols produced)
    for K in [9, 12, 15] :
        num_states = 4
        input_data = np.array([]).reshape(0,2)
        for class_name in ["a", "ai", "bA", "dA", "tA"] :
            temp_input = getTrainData_letter(class_name)
            for sample in temp_input : 
                for frame in sample : 
                    input_data = np.r_["0,2", input_data, np.array(frame)]
        centroids = KMeansTrain(K, input_data)

        # Getting the symbols produced by each feature vector in every sample of every class
        input_labels = []
        for class_name in ["a", "ai", "bA", "dA", "tA"] : 
            temp_input = getDevData_letter(class_name)
            temp_list1 = []
            for file in temp_input : 
                temp_list2 = []
                for frame in file : 
                    temp_list2.append(KMeansTest(np.array(frame), centroids))
                temp_list1.append(temp_list2)
            input_labels.append(temp_list1)

        models = []
        for class_list in input_labels : 
            # Training the HMM model with the sequences of symbols extraced above
            file = open("labelTrainLetter.hmm.seq","w")
            for sample in class_list : 
                for label in sample : 
                    file.write(str(label) + " ")
                file.write("\n")
            tempStr = "./train_hmm labelTrainLetter.hmm.seq 7500 "+ str(num_states) + " " + str(K) + " 0.0001"
            file.close()
            
            # Getting the probability values of various transitions (state to state, and production of symbol from state)
            os.system(tempStr)
            file = open("labelTrainLetter.hmm.seq.hmm","r")
            file.readline()
            file.readline()
            aii = []
            aij = []
            biik = []
            bijk = []

            for i in range(num_states) : 
                temp_list1 = [element.split("\n")[0] for element in file.readline().split("\t")]        
                temp_list2 = [element.split("\n")[0] for element in file.readline().split("\t")]

                aii.append(float(temp_list1[0]))
                aij.append(float(temp_list2[0]))

                temp_list3 = []
                temp_list4 = []
                for j in range(1,K+1) : 
                    temp_list3.append(float(temp_list1[j]))
                    temp_list4.append(float(temp_list2[j]))
                biik.append(temp_list3)
                bijk.append(temp_list4)
                
                models.append([aii, aij, biik, bijk])
                file.readline()

        # Checking accuracy of the model and plotting the respective graphs
        total = 0
        true = 0
        curr_class = 0
        scores = []
        conf_matrix = []
        for class_name in ["a", "ai", "bA", "dA", "tA"] :
            temp_list = [0] * 5
            samples_data = getDevData_letter(class_name)
            for sample_data in samples_data : 
                label_seq = []
                for frame in sample_data : 
                    label_seq.append(KMeansTest(np.array(frame), centroids))
                probs = []
                for model in models : 
                    probs.append(HMMProbs(label_seq, model[0], model[1], model[2], model[3]))
                temp_ind = 0
                for score in probs : 
                    scores.append((score, curr_class, int(temp_ind/num_states)))
                    temp_ind += 1
                ind = np.argmax(probs)
                temp_list[int(ind/num_states)] += 1
                if(curr_class == ind/num_states) : 
                    true += 1
                total += 1
            curr_class += 1
            conf_matrix.append(temp_list)
        print(f"Accuracy is: {100 * true / total}%")
        score_list.append(scores)
    plotROC(score_list)
    plotDET(score_list)
    plotconfusion(conf_matrix)

if __name__ == "__main__" : 
    Digits()
    Handwriting()