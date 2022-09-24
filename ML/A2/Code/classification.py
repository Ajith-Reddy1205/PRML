import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.colors as colors   
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from math import log
from math import sqrt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix

# Returns the mean vector of all 2d points 
def getMeans(X,Y) : 
    mu_set = []
    for i in range(3) : 
        mu_set.append(np.array([np.mean(X[i]), np.mean(Y[i])]))
    return mu_set


#-----------------------------------Training---------------------------------------

# Case-1 : Bayers with same covariance matrix for all classes
def train1(X,Y,X_total,Y_total) : 
    mu_set = getMeans(X,Y)
    sigma = np.zeros(shape=(2,2))
    sum = 0
    for i in range(3) : 
        n = X[i].shape[0]
        sigma += (n-1) * np.cov(np.array([X[i],Y[i]]))
        sum += n
    sigma_set = [sigma/(sum-3)] * 3

    plot_set1(X_total,Y_total,mu_set,sigma_set,1)
    return sigma_set

# Case-2 : Bayers with different covariance matrix for all classes
def train2(X,Y,X_total,Y_total) :
    mu_set = getMeans(X,Y)
    sigma_set = []
    for i in range(3) : 
        n = X[i].shape[0]
        sigma = np.cov(np.array([X[i],Y[i]]))
        sigma_set.append(sigma)

    plot_set1(X_total,Y_total,mu_set,sigma_set,2)
    return sigma_set

# Case-3 : Naive Bayers with C = sigma^2 I 
def train3(X,Y,X_total,Y_total) : 
    mu_set = getMeans(X,Y)
    n = 0 
    sigma = 0.0
    for i in range(3) : 
        m = X[i].shape[0]
        n += m*2
        for j in range(m) : 
            sigma += (X[i][j] - mu_set[i][0])**2 + (Y[i][j] - mu_set[i][1])**2
    temp_array = np.array([[sigma/(n-6),0],[0,sigma/(n-6)]])
    sigma_set = [temp_array] * 3

    plot_set1(X_total,Y_total,mu_set,sigma_set,3)
    return sigma_set

# Case-4 : Naive Bayers with C1 = C2
def train4(X,Y,X_total,Y_total) : 
    mu_set = getMeans(X,Y)
    n = 0
    sigma1 = 0.0
    sigma2 = 0.0
    for i in range(3) :
        m = X[i].shape[0]
        n += m
        for j in range(m) : 
            sigma1 += (X[i][j] - mu_set[i][0])**2
            sigma2 += (Y[i][j] - mu_set[i][1])**2
    temp_array = np.array([[sigma1/(n-3),0],[0,sigma2/(n-3)]])
    sigma_set = [temp_array] * 3

    plot_set1(X_total,Y_total,mu_set,sigma_set,4)
    return sigma_set

# Case-5 : Naive Bayers with C1 != C2
def train5(X,Y,X_total,Y_total) : 
    mu_set = getMeans(X,Y)
    sigma_set = []
    for i in range(3) : 
        m = X[i].shape[0]
        sigma1 = 0.0
        sigma2 = 0.0
        for j in range(m) : 
            sigma1 += (X[i][j] - mu_set[i][0])**2
            sigma2 += (Y[i][j] - mu_set[i][1])**2
        temp_array = np.array([[sigma1/(m-1),0],[0,sigma2/(m-1)]])
        sigma_set.append(temp_array)
    
    plot_set1(X_total,Y_total,mu_set,sigma_set,5)
    return sigma_set

#-----------------------------------------------------------------------------

#-----------------------------Plotting Part-1---------------------------------

def plot_set1(X_total, Y_total, mu_set, sigma_set,case) :
    x_min = min(X_total)
    x_max = max(X_total)
    y_min = min(Y_total)
    y_max = max(Y_total)

    temp1 = np.arange(x_min,x_max,0.1)
    temp2 = np.arange(y_min,y_max,0.1)

    # for real data, steps are taken as 5 instead of 0.1 due to the large range of values
    # temp1 = np.arange(x_min,x_max,5)
    # temp2 = np.arange(y_min,y_max,5)

    x,y = np.meshgrid(temp1,temp2)
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = x
    pos[:,:,1] = y

    Z = []
    for i in range(3) : 
        sigma_det = np.linalg.det(sigma_set[i])
        sigma_inv = np.linalg.inv(sigma_set[i])
        N = np.sqrt((2*np.pi)**2 * sigma_det)
        fac = np.einsum("...k,kl,...l->...", pos-mu_set[i], sigma_inv, pos-mu_set[i])
        Z.append(np.exp(-fac/2)/N)

    pdf = np.maximum(np.maximum(Z[0],Z[1]),Z[2])
    
    plotPDF_Helper(x,y,pdf,case)
    plotContour_Helper(x,y,pdf,x_min,x_max,y_min,y_max,mu_set,sigma_set,case)

# PDF Plots 
def plotPDF_Helper(x,y,pdf,case) : 
    fig = plt.figure(figsize=(14,9))
    ax = plt.axes(projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.plot_surface(x, y, pdf, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    ax.view_init(20, 30)

    plt.title(f"Probability Density Functions Case-{case}")
    plt.savefig(f"PDF-{case}.png")

# Constant Density and Eigen Vectors Curves 
def plotContour_Helper(x,y,pdf,x_min,x_max,y_min,y_max,mu_set,sigma_set,case) : 
    fig = plt.figure(figsize=(8,8))
    
    # Constant Density Plots 
    ax = fig.gca()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    cfset = ax.contourf(x, y, pdf, cmap="coolwarm")
    ax.imshow(np.rot90(pdf), cmap="coolwarm", extent=[x_min, x_max, y_min, y_max])
    cset = ax.contour(x, y, pdf, colors="k")
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Eigen Vectors 
    for i in range(3) : 
        mean = mu_set[i]
        sigma = sigma_set[i]
        eigen_vectors = np.linalg.eig(sigma)[1]
        
        # Choosing 1000 as the length of the eigen vectors to display 
        plt.arrow(mean[0], mean[1], eigen_vectors[:,0][0]*1000, eigen_vectors[:,0][1]*1000)
        plt.arrow(mean[0], mean[1], eigen_vectors[:,1][0]*1000, eigen_vectors[:,1][1]*1000)
    plt.title(f"Constant Density Curves and Eigen Vectors Case-{case}")
    plt.savefig(f"Contour-{case}.png")

#------------------------------------------------------------------------------

#-------------------------------Predictions------------------------------------

# Case-1 : Bayers with same covariance matrix 
def classifier1(point,mu_set,sigma_set) :
    sigma = sigma_set[0] 
    values = []
    scores = []
    for i in range(3) : 
        v1 = 2 * point.T @ np.linalg.inv(sigma) @ mu_set[i] 
        v2 = mu_set[i].T @ np.linalg.inv(sigma) @ mu_set[i]
        # temp = np.linalg.det(v1-v2)
        values.append((v1-v2,i+1))
        scores.append(v1-v2)
    return (max(values)[1],scores)

# Case-2 : Bayers with different covariance matrix
def classifier2(point,mu_set,sigma_set) : 
    values = []
    scores = []
    for i in range(3) : 
        v1 = log(np.linalg.det(sigma_set[i]))
        v2 = (point - mu_set[i]).T @ np.linalg.inv(sigma_set[i]) @ (point - mu_set[i])
        values.append((-(v1+v2),i+1))
        scores.append(-v1-v2)
    return (max(values)[1],scores)

# Case-3 : Naive Bayers with C = sigma^2 I
def classifier3(point,mu_set,sigma_set) :
    values = []
    scores = []
    for i in range(3) : 
        v1 = (point[0] - mu_set[i][0])**2
        v2 = (point[1] - mu_set[i][1])**2
        values.append((-(v1+v2),i+1))
        scores.append(-v1-v2)
    return (max(values)[1],scores)

# Case-4 : Naive Bayers with C1 = C2 
def classifier4(point,mu_set,sigma_set) :
    values = []
    scores = []
    sigma1 = sigma_set[0][0][0]
    sigma2 = sigma_set[0][1][1]
    for i in range(3) : 
        v1 = log(sqrt(sigma1)) + ((point[0] - mu_set[i][0])**2)/(2 * sigma1**2)
        v2 = log(sqrt(sigma2)) + ((point[1] - mu_set[i][1])**2)/(2 * sigma2**2)
        values.append((-(v1+v2),i+1))
        scores.append(-v1-v2)
    return (max(values)[1],scores)

# Case-5 : Naive Bayers with C1 != C2 
def classifier5(point,mu_set,sigma_set) :
    values = []
    scores = []
    for i in range(3) : 
        sigma1 = sigma_set[i][0][0]
        sigma2 = sigma_set[i][1][1]
        v1 = log(sqrt(sigma1)) + ((point[0] - mu_set[i][0])**2)/(2 * sigma1)
        v2 = log(sqrt(sigma2)) + ((point[1] - mu_set[i][1])**2)/(2 * sigma2)
        values.append((-(v1+v2),i+1))
        scores.append(-v1-v2)
    return (max(values)[1],scores)

# Predict the prediction set with all 5 models, given the parameters of each model(mu_set and sigma_sets, since the mean is the same for all models)
def classify(mu_set,sigma_sets,points,class_test_set) :
    class_pred_sets = [[],[],[],[],[]]
    class_scores = [[],[],[],[],[]]
    x_max = max(points[:,0])
    x_min = min(points[:,0])
    y_max = max(points[:,1])
    y_min = min(points[:,1])

    temp1 = np.arange(x_min,x_max,0.1)
    temp2 = np.arange(y_min,y_max,0.1)
    
    # for real data, gaps are taken as 5 instead of 0.1 due to the large range of values
    # temp1 = np.arange(x_min,x_max,5)
    # temp2 = np.arange(y_min,y_max,5)
    
    x,y = np.meshgrid(temp1,temp2)

    for point in points : 
        class_pred1,class_scores1 = classifier1(point,mu_set,sigma_sets[0])
        class_pred_sets[0].append(class_pred1)
        class_scores[0].append(class_scores1)

        class_pred2,class_scores2 = classifier2(point,mu_set,sigma_sets[1])
        class_pred_sets[1].append(class_pred2)
        class_scores[1].append(class_scores2)

        class_pred3,class_scores3 = classifier3(point,mu_set,sigma_sets[2])
        class_pred_sets[2].append(class_pred3)
        class_scores[2].append(class_scores3)

        class_pred4,class_scores4 = classifier4(point,mu_set,sigma_sets[3])
        class_pred_sets[3].append(class_pred4)
        class_scores[3].append(class_scores4)

        class_pred5,class_scores5 = classifier5(point,mu_set,sigma_sets[4])
        class_pred_sets[4].append(class_pred5)
        class_scores[4].append(class_scores5)

    plot_set2(class_scores,class_test_set,class_pred_sets,x,y,mu_set,sigma_sets)

#-------------------------------------------------------------------------------

#-------------------------------Plotting Part-2---------------------------------

def plot_set2(class_scores,class_test_set,class_pred_sets,x,y,mu_set,sigma_sets) : 
    for i in range(5) : 
        plotDecision_helper(x,y,mu_set,sigma_sets[i],i)
    
    for i in range(5) : 
        plotConfusionMatrix(class_test_set,class_pred_sets[i],i)

    plotROC(class_scores,class_test_set)

# Decision Boundary 
def plotDecision_helper(x,y,mu_set,sigma_set,case) :
    x_temp = x.ravel()
    y_temp = y.ravel()
    z_temp = [] 
        
    if(case == 0) :
        for i in range(x_temp.shape[0]) : 
            point = np.array([x_temp[i],y_temp[i]])
            z_temp.append(classifier1(point,mu_set,sigma_set)[0])
    if(case == 1) :
        for i in range(x_temp.shape[0]) : 
            point = np.array([x_temp[i],y_temp[i]])
            z_temp.append(classifier2(point,mu_set,sigma_set)[0])
    if(case == 2) :
        for i in range(x_temp.shape[0]) :
            point = np.array([x_temp[i],y_temp[i]])
            z_temp.append(classifier3(point,mu_set,sigma_set)[0])
    if(case == 3) :
        for i in range(x_temp.shape[0]) :
            point = np.array([x_temp[i],y_temp[i]])
            z_temp.append(classifier4(point,mu_set,sigma_set)[0])
    if(case == 4) :
        for i in range(x_temp.shape[0]) :
            point = np.array([x_temp[i],y_temp[i]])
            z_temp.append(classifier5(point,mu_set,sigma_set)[0])
    z_temp = np.array(z_temp)
    z = z_temp.reshape(x.shape)

    fig = plt.figure(figsize=(8,8))
    plt.contourf(x, y, z, cmap=colors.ListedColormap(["red","green","blue"]))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Decision Surface Case-{case+1}")
    plt.savefig(f"Decision-{case+1}.png")      
    plt.clf()  

# Confusion Matrix
def plotConfusionMatrix(class_test_set,class_pred_set,case) : 
    class_test_set = np.array(class_test_set)
    class_pred_set = np.array(class_pred_set)
    fig,ax = plt.subplots(1)
    columns = ["class %s" %(i) for i in range(1,4)]
    confusionMatrix = confusion_matrix(class_test_set,class_pred_set)
    df_cm = DataFrame(confusionMatrix, index=columns, columns=columns)
    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)
    ax.set_xlabel("Target Class")
    ax.set_ylabel("Output Class")
    plt.title(f"Confusion Matrix Case-{case+1}")
    fig.savefig(f"Confusion-{case+1}.png")

# ROC and DET curves
def plotROC(total_scores,class_test_set) : 
    fig,axarr = plt.subplots(1,2,figsize=(18,8))
    for k in range(5) : 
        class_scores = []
        temp = []
        scores = total_scores[k]
        for i in range(len(scores)) : 
            for j in range(3) : 
                class_scores.append((scores[i][j],class_test_set[i],j+1))
                temp.append(scores[i][j])
        class_scores = sorted(class_scores)
        TPR = []  # TP/(TP+FN)
        FNR = []  # FN/(FN+TP)
        FPR = []  # FP/(FP+TN)
        for threshold in sorted(temp) :  
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for data in class_scores : 
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

        axarr[0].plot(FPR,TPR,label=f"case-{k+1}")
        axarr[1].plot(FPR,FNR,label=f"case-{k+1}")

    axarr[0].legend()
    axarr[0].set_title(f"ROC Curve")
    axarr[0].set_xlabel("False Positive Rate(FPR)")
    axarr[0].set_ylabel("True Positive Rate(TPR)")
    axarr[1].legend()
    axarr[1].set_title(f"DET Curve")
    axarr[1].set_xlabel("False Alarm Rate")
    axarr[1].set_ylabel("Missed Detection Rate")

    values = axarr[1].get_yticks()
    axarr[1].set_yticklabels(["{:.0%}".format(y) for y in values])
    values = axarr[1].get_xticks()
    axarr[1].set_xticklabels(["{:.0%}".format(x) for x in values])

    fig.savefig("ROC_DET_curves.png")

#----------------------------------------------------------------------------

def bayerClassification(train,dev) :
    file1 = open(train,"r")
    X_temp = [[],[],[]]
    Y_temp = [[],[],[]]
    X_total = []
    Y_total = []
    for string in file1 : 
        temp = string.strip().split(",")
        ind = int(temp[2])-1
        X_total.append(float(temp[0]))
        Y_total.append(float(temp[1]))
        X_temp[ind].append(float(temp[0]))
        Y_temp[ind].append(float(temp[1])) 
    file1.close()

    X = []
    Y = []
    X_total = np.array(X_total)
    Y_total = np.array(Y_total)
    for i in range(3) : 
        X.append(np.array(X_temp[i]))
        Y.append(np.array(Y_temp[i]))

    sigma_sets = []
    sigma_sets.append(train1(X,Y,X_total,Y_total))
    sigma_sets.append(train2(X,Y,X_total,Y_total))
    sigma_sets.append(train3(X,Y,X_total,Y_total))
    sigma_sets.append(train4(X,Y,X_total,Y_total))
    sigma_sets.append(train5(X,Y,X_total,Y_total))
    
    mu_set = getMeans(X,Y)
    
    class_test_set = []
    points = []
    file2 = open(dev,"r")
    for string in file2 : 
        temp = string.strip().split(",")
        point = np.array([float(temp[0]),float(temp[1])])
        points.append(point)
        class_test = int(temp[2])
        class_test_set.append(class_test)
    file2.close()
    points = np.array(points)
    classify(mu_set,sigma_sets,points,class_test_set)

#--------------------------------------------------------------------------

if __name__ == "__main__" : 
    train = input()
    dev = input()
    bayerClassification(train,dev)
