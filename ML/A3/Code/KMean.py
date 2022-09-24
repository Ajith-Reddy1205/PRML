import math
import random
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os
from scipy.stats import norm


def getSynTrainData() :
    data = np.genfromtxt(r"synthetic/train.txt", delimiter=",")
    return data

def getSynDevData() :
    data = np.genfromtxt(r"synthetic/dev.txt", delimiter=",")
    return data

def getImgTrainData() :
    train_all = []
    for class_type in ["coast","forest","highway","mountain","opencountry"] : 
        X = [np.array([]).reshape(0,23)] * 36
        string = f"features/{class_type}" + r"/train"
        for image in os.listdir(string) : 
            file = open(string+f"/{image}","r")
            ind = 0
            for line in file: 
                temp_list = line.split(" ")
                temp_array = []
                for temp in temp_list:
                    temp_array.append(temp.split("\n")[0])
                temp_array = np.array(temp_array, dtype="float")
                X[ind] = np.r_["0,2", X[ind], (temp_array)]
                ind += 1
        train_all.append(X)  
    return train_all

def getImgTestData() :
    test_all = []
    for class_type in ["coast","forest","highway","mountain","opencountry"] : 
        X = []
        string = f"features/{class_type}" + r"/dev"
        for image in os.listdir(string) : 
            Y = []
            file = open(string+f"/{image}","r")
            for line in file: 
                temp_list = line.split(" ")
                temp_array = []
                for temp in temp_list:
                    temp_array.append(temp.split("\n")[0])
                temp_array = np.array(temp_array, dtype="float")
                Y.append(temp_array)
            X.append(Y)
        test_all.append(X)  
    return test_all

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
        
        clusters = {}
        for k in range(K):
            clusters[k] = np.array([]).reshape(0, num_features)

        for n in range(num_samples):
            clusters[cluster_numbers[n]] = np.r_["0,2", clusters[cluster_numbers[n]], X[n]]

        for k in range(K):
            centroids[:,k] = np.mean(clusters[k], axis=0)
    
    return centroids


def KMeansTest(X, centroids) : 
    num_samples = X.shape[0]
    K = centroids.shape[1]
    distances = np.array([]).reshape(num_samples, 0)
    for k in range(K) : 
        distances = np.c_[distances, np.sum((X - centroids[:,k])**2, axis=1)]
    
    cluster_numbers = np.argmin(distances, axis=1)
    return cluster_numbers 

def getcov(X,gamma,mean,k):
    
    cov = np.zeros([X.shape[1],X.shape[1]], dtype = float)
    n_k = 0
    for i in range(X.shape[0]):
        temp = np.array(X[i] - mean).reshape(X.shape[1],1)
        temp = np.dot(temp, temp.transpose())
        temp = np.multiply(temp, gamma[i][k])
        cov = np.add(cov,temp)
        # print(cov)
        n_k += gamma[i][k]

    # for i in range(cov.shape[0]):
    #     for j in range(cov.shape[1]):
    #         if i != j:
    #             cov[i][j] = 0
    return np.multiply(cov,1.0/n_k)

def distribution(mean,cov,x):
    num_features = mean.size
    
    denom = ((2*math.pi)**(num_features/2)) * (np.linalg.det(cov))**.5
    temp = x - mean
    temp_t = temp.transpose()
    power = - 0.5 * np.dot(np.dot(temp_t,np.linalg.inv(cov)), temp)
    # print(power)
    num = math.exp(power)
    return num/denom

def log_distribution(mean,cov,x):
    num_features = mean.size
    
    denom = ((2*math.pi)**(num_features/2)) * (np.linalg.det(cov))**.5
    temp = x - mean
    temp_t = temp.transpose()
    power = - 0.5 * np.dot(np.dot(temp_t,np.linalg.inv(cov)), temp)
    # print(power)
    # num = math.exp(power)
    return -np.log(denom) + power

def GMM(K, X):
    num_samples = X.shape[0]
    num_features = X.shape[1]

    centroids = KMeansTrain(K,X)
    cl = KMeansTest(X, centroids)
    gamma = np.zeros(shape=(num_samples, K))

    for i in range(num_samples):
        gamma[i][cl[i]] = 1
    
    pi = np.sum(gamma,axis = 0)
    # print(pi)
    cov = []
    mean = []
    for q in range(4):
        # print(q)
        # Mean and variance
        cov = []
        mean = []
        new_gamma = np.zeros(shape=(num_samples, K))
        X_k = np.empty(shape=X.shape)
        for k in range(K):
            n_k = 0 
            for i in range(X.shape[0]):
                    # print(i)
                    X_k[i] = (np.multiply(X[i],gamma[i][k]))
                    n_k += gamma[i][k]
            me = np.multiply(np.sum(X_k,axis = 0), 1.0/n_k)
            mean.append(me)
            cov.append(getcov(X,gamma,me,k))

        mean = np.array(mean)
        cov = np.array(cov)
        # E-step
        for n in range(num_samples):
            # print(n)
            total = 0
            for k in range(K):
                # print(mean[k],cov[k],X[n])
                # print(1)
                total += pi[k] * distribution(mean[k],cov[k],X[n])
            for k in range(K):
                new_gamma[n][k] = pi[k] * distribution(mean[k],cov[k],X[n]) / total

        gamma = new_gamma
        pi = np.sum(gamma,axis = 0)
        # print(pi)

    return gamma,mean,cov

def GMMTest(X,syn_test1,syn_test2,syn_gamma1,syn_gamma2,mean1,cov1,mean2,cov2,K):
    count = 0
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    scores = []
    for x in syn_test1:
        mx1 = 0.0
        for k in range(K):
            mx1 = max(mx1,distribution(mean1[k],cov1[k],x))
        mx2 = 0.0
        for k in range(K):
            mx2 = max(mx2,distribution(mean2[k],cov2[k],x))
        # print(mx1,mx2)
        pr = mx1/(mx1 + mx2)
        if mx1 > mx2 :
            count += 1
            x1.append(x[0])
            y1.append(x[1])
            scores.append((pr,1,1))
        else:
            x2.append(x[0])
            y2.append(x[1])
            scores.append((1-pr,1,2))

    for x in syn_test2:
        mx1 = 0.0
        for k in range(K):
            mx1 = max(mx1,distribution(mean1[k],cov1[k],x))
        mx2 = 0.0
        for k in range(K):
            mx2 = max(mx2,distribution(mean2[k],cov2[k],x))
        pr = mx1/(mx1 + mx2)
        if mx1 <  mx2 :
            count += 1
            x2.append(x[0])
            y2.append(x[1])
            scores.append((1-pr,2,2))
        else:
            x1.append(x[0])
            y1.append(x[1])
            scores.append((pr,2,1))

    # plt.scatter(x1, y1, color = "blue")
    # plt.scatter(x2, y2, color = "green")
    print("GMM : Correctly classified = "  + str(count))
    # plt.savefig("scatter.png")
    # print(count)
    # plotROC(scores)

def GMMTest_img(img_gmms, img_data_test,K):
    right = 0
    wrong = 0
    scores = []
    y_test = []
    confusion = np.zeros((5,5))
    for p in range(5):
        ri = right
        wr = wrong
        # print(p)
        probab = []
        for q in range(5):
            prr = []
            for j in range(len(img_data_test[p])):
                pr = 1.0
                for l in range(36):
                    mx = -1e40
                    for k in range(K):
                        mx = max(mx, log_distribution(img_gmms[q][0][k],img_gmms[q][1][k],img_data_test[p][j][l]))
                    pr += mx
                prr.append(pr)
            probab.append(prr)
        # print(probab)
        for j in range(len(img_data_test[p])):
            mx = 0
            cl = -1
            pre = 0
            tt = []
            for q in range(5):
                if probab[q][j] > mx:
                    mx = probab[q][j]
                    cl = q
                tt.append(probab[q][j])
                pre += probab[q][j]
            if cl == p:
                right+=1
            else:
                wrong+=1
            y_test.append(p)
            scores.append(tt)
            confusion[p][cl] += 1
        print("Right Predicrions  = " + str(right - ri), "Wrong Predictions = " + str(wrong - wr), "Class = " + str(p+1))

    tpr,fpr,fnr = roc_plots(y_test,scores,5,K)
    print("Overall : Right Predicrions  = " + str(right), "Wrong Predictions = " + str(wrong))
    return tpr,fpr,fnr,confusion

def normalise(arr):
    mn = 1e50
    mx = -1e50
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                for l in range(len(arr[i][j][k])):
                    mn = min(mn, arr[i][j][k][l])
                    mx = max(mx, arr[i][j][k][l])

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                for l in range(len(arr[i][j][k])):
                    x = arr[i][j][k][l]
                    x = (x - mn)/(mx - mn)
                    arr[i][j][k][l] = x
    return arr

def nrm(arr):
    x = np.array(arr[0])
    mn = np.amin(x)
    mx = np.amax(x)
    x = [(i-mn)/(mx-mn) for i in x]
    print(x)
    return np.array(x)

def plotconfusion(confusion):
    df_cm = pd.DataFrame(confusion, index = [i for i in "12345"], columns = [i for i in "12345"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("img_confusion.png")

def roc_plots(y,allclasses,class_count,kk):
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
    return (tpr,fpr,fnr)

def plotROC(scores): 
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
    fig.savefig("ROC_curves.png")

def plotDET(scores): 
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

    fig.savefig("DET_curve.png")


def plotdecisionbdry_syn(syn_gamma1,syn_gamma2,mean1,cov1,mean2,cov2,K):
    X=np.linspace(-15,12,500)
    Y=np.linspace(-15,12,500)
    boundary=[]
    for i in X:
        for j in Y:
            point=np.array([i,j])
            sumofsquares1=np.zeros((20,1))
            sumofsquares2=np.zeros((20,1))
            distpointc1=point-GMMmeanclass1
            distpointc2=point-GMMmeanclass2
            for k in range(ndim):
                sumofsquares1+=np.c_[distpointc1[:,k]*distpointc1[:,k]]
                sumofsquares2+=np.c_[distpointc2[:,k]*distpointc2[:,k]]
            eucdistc1=np.sqrt(sumofsquares1)
            eucdistc2=np.sqrt(sumofsquares2)
            if(abs(np.amin(eucdistc1) - np.amin(eucdistc2))<0.05):
                boundary.append(list(point))
    xd, yd = np.array(boundary).T
    plt.scatter(xd,yd,color="black")

# a=np.array([1,2])
# b=np.array([[3,4],[5,6]])

if __name__ == "__main__" :
    print("SYNTHETIC DATA")
    for K in [2,3,5,10,15,20]:
        print("K = " + str(K))
        syn_input = getSynTrainData()
        syn_data = np.array(syn_input[:,:2], dtype="float")
        syn_data1 = np.array(syn_data[:1250,:])
        syn_data2 = np.array(syn_data[1250:,:])
        syn_class = np.array(syn_input[:,2], dtype="int")
        syn_centroids1 = KMeansTrain(K, syn_data1)
        syn_centroids2 = KMeansTrain(K, syn_data2)
        syn_centroids = np.c_[syn_centroids1, syn_centroids2]

        syn_test_input = getSynDevData()
        syn_test1 = np.array(syn_test_input[:500,:2], dtype="float")
        syn_test2 = np.array(syn_test_input[500:,:2], dtype="float")

        true = 0
        cluster_numbers1 = KMeansTest(syn_test1, syn_centroids)
        cluster_numbers2 = KMeansTest(syn_test2, syn_centroids)

        for i in range(500) : 
            if(cluster_numbers1[i] < K) :
                true += 1
            if(cluster_numbers2[i] >= K) :
                true += 1

        print("Kmeans : Correctly classified = "  + str(true))

        syn_gamma1,mean1,cov1 = GMM(K, syn_data1)
        syn_gamma2,mean2,cov2 = GMM(K, syn_data2)
        GMMTest(syn_data,syn_test1,syn_test2,syn_gamma1,syn_gamma2,mean1,cov1,mean2,cov2,K)

    # ax = plt.axes()
    # plt.xlabel("False Positive Rate(FPR)")
    # plt.ylabel("True Positive Rate(TPR)")


    # fig, ax = plt.subplots()
    # plt.xlabel("False Alarm Rate")
    # plt.ylabel("Missed Detection Rate")

    print("IMAGE DATA")

    for K in [2,5,10,15]:
        print("K = " + str(K))
        img_data_train = getImgTrainData()
        # img_data_train = normalise(img_data_train)
        # print(len(img_data_train), len(img_data_train[0]), len(img_data_train[0][0]))

        img_gmms = [None]*5
        for p in range(5):
            temp_data = []
            for i in range(36):
                temp_data.append(img_data_train[p][i])
            tt_data = []
            for i in range(36):
                for j in range(len(temp_data[i])):
                    tt_data.append(temp_data[i][j])
            temp_data = tt_data
            # print(np.array(temp_data).shape)
            img_gamma,mean,cov = GMM(K, np.array(temp_data))
            img_gmms[p] = [mean, cov, img_gamma]
            # print(np.array(mean).shape, np.array(cov).shape)


        img_data_test = getImgTestData()
        # img_data_test = normalise(img_data_test)
        tpr,fpr,fnr,confusion = GMMTest_img(img_gmms, img_data_test, K)
        # plt.plot(fpr,tpr,label = 'K = ' + str(K))
        # blah1 = norm.ppf(fpr)
        # blah2 = norm.ppf(fnr)
        # axes = plt.plot(blah1,blah2, label = 'K = ' + str(K))
        # values = plt.yticks()
        # plt.yticklabels(["{:.0%}".format(y) for y in normalise(values)])
        # values = plt.xticks()
        # plt.xticklabels(["{:.0%}".format(x) for x in normalise(values)])

    # values = plt.yticks()
    # # print(values)
    # ax.set_yticklabels(["{:.0%}".format(y) for y in nrm(values)])
    # values = plt.xticks()
    # ax.set_xticklabels(["{:.0%}".format(x) for x in nrm(values)])
    # plt.legend(loc='best')
    # plt.savefig("roc_img_final.png")