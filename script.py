#Group 99
#Members: Samuel Kriever, Tom Ludwig, Matt O'Connor

#CSE 474
#Programming Assignment 2

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

def ldaLearn(X,y):
    print("LDA Learn...\n")
    
    d = X.shape[1]
    k = len( np.unique(y) )
    
    means = []
    for c in range(k): # do for each class
        row = []
        for f in range(d): # do for each feature
            row.append( np.mean(X[y == c+1][f]) )
        means.append(row)
    
    covmat = np.cov(X, rowvar=0)
    
    means = np.array(means)
    covmat = np.array(covmat)
    
    return means,covmat


def qdaLearn(X,y):
    print("QDA Learn...\n")
    
    N = X.shape[0]
    d = X.shape[1]
    k = len( np.unique(y) )
    
    means = []
    for c in range(k): # do for each class
        row = []
        for f in range(d): # do for each feature
            row.append( np.mean(X[y == c+1][f]) )
        means.append(row)
    
    covmats = []
    for c in range(k): # do for each class
        cX = []
        for x in range(N): # do for each object
            if y[x] == c+1:
                cX.append( X[x] ) # only chose objs from that class
        covmats.append( np.cov(cX, rowvar=0) )
    
    means = np.array(means)
    covmats = np.array(covmats)
    
    return means,covmats


def ldaTest(means,covmat,Xtest,ytest):
    print("LDA Test...\n")
    
    N = Xtest.shape[0]
    d = Xtest.shape[1]
    k = len( np.unique(y) )
    
    denr = (2.0 * np.pi) ** (d / 2.0)
    Cdet = sqrt( np.linalg.det(covmat) )
    Cinv = np.linalg.inv(covmat)
    
    assigned = []
    for x in range(N): # do for each object
        row = []
        for c in range(k): # do for each class
            norm = Xtest[x] - means[c]
            tran = np.transpose(norm)
            
            expo = -0.5 * ( np.dot( tran, np.dot(Cinv, norm) ) )
            
            pdf = ( 1.0 / (denr * Cdet) ) * np.exp( expo )
            row.append(pdf)
        
        assigned.append( [np.argmax(row) + 1] ) # assign to class with highest fx
        
    assigned = np.array(assigned).astype(float)
    
    acc = np.mean(assigned == ytest)
    
    """diff = []
    for x in range( assigned.shape[0] ):
        d = assigned[x] - ytest[x]
        diff.append(d)
    diff = np.array(diff)
    print( "Prediction difference per object:\n%s" % (diff,) )"""
    
    return acc


def qdaTest(means,covmats,Xtest,ytest):
    print("QDA Test...\n")
    
    N = Xtest.shape[0]
    d = Xtest.shape[1]
    k = len( np.unique(y) )
    
    denr = (2.0 * np.pi) ** (d / 2.0)
    
    assigned = []
    for x in range(N): # do for each object
        row = []
        for c in range(k): # do for each class
            Cdet = sqrt( np.linalg.det(covmats[c]) )
            Cinv = np.linalg.inv(covmats[c])
            
            norm = Xtest[x] - means[c]
            tran = np.transpose(norm)
            
            expo = -0.5 * ( np.dot( tran, np.dot(Cinv, norm) ) )
            
            pdf = ( 1.0 / (denr * Cdet) ) * np.exp( expo )
            row.append(pdf)
        
        assigned.append( [np.argmax(row) + 1] ) # assign to class with highest fx
    
    assigned = np.array(assigned).astype(float)
    
    acc = np.mean(assigned == ytest)
    
    """diff = []
    for x in range( assigned.shape[0] ):
        d = assigned[x] - ytest[x]
        diff.append(d)
    diff = np.array(diff)
    print( "Prediction difference per object:\n%s" % (diff,) )"""
    
    return acc


def learnOLERegression(X,y):
    # w = (XT X)^-1 (XT y)
    
    # XT
    XT = X.transpose()
    # (XT X)
    XTX = XT.dot(X)
    # (XT X)^-1
    XTXinv = np.linalg.inv(XTX)
    # (XT y)
    XTy = XT.dot(y)
    # (XT X)^-1 (XT y)
    w = XTXinv.dot(XTy)
    
    return w


def learnRidgeRegression(X,y,lambd):
    # w = (XT X + lam I)^-1 (XT y)
    
    d = X.shape[1]
    # 65x65 identity matrix
    idMat = np.matrix(np.identity(d), copy=False)
    # XT
    XT = X.transpose()
    # (XT X)
    XTX = XT.dot(X)
    # (XT y)
    XTy = XT.dot(y)
    # (lam I)
    idMatLam = idMat * lambd
    # (XT X + lam I)
    XTXLam = np.add(XTX, idMatLam)
    # (XT X + lam I)^-1
    XTXLamInv = np.linalg.inv(XTXLam)
    # (XT X + lam I)^-1 (XT y)
    w = XTXLamInv.dot(XTy)   
                                                    
    return w


def testOLERegression(w,Xtest,ytest):
    # J(w) = (1/N) SQRT( SUM[1,N]: (yi - wT xi)^2 )
    # J(w) = (1/N) SQRT( SUM: (y - wT X)^2 )
    
    N = float(Xtest.shape[0])
    # w already aligned... transpose not needed!
    # (wT X)
    yPredicted = Xtest.dot(w)
    # (y - wT X)
    error = np.asarray(ytest - yPredicted)
    # (y - wT X)^2
    errorSquared = error ** 2
    # SUM: (y - wT X)^2
    errorSquaredSum = np.sum(errorSquared)
    # (1/N) SQRT( SUM: (y - wT X)^2 )
    rmse = sqrt(errorSquaredSum) / N
    
    return rmse


def regressionObjVal(w, X, y, lambd):
    # E(w) = (1/N) (y - X w)T (y - X w) + (lam/2) (wT w)
    # E_grad(w) = w - lam XT (X w - y)
    
    N = float(X.shape[0])
    # w is in the opposite orientation to start, so w and wT are reversed
    # make compatible with matrices
    w = np.matrix(w)
    w = w.transpose()
    # wT
    wT = w.transpose()
    # (X w)
    yPredicted = X.dot(w)
    # (y - X w)
    errorCalc = np.asarray(yPredicted - y)
    # (y - X w)T
    errorCalcT = errorCalc.transpose()
    # (lam/2) (wT w)
    regTerm = (lambd / 2.0) * ( wT.dot(w) )
    # (1/N) (y - X w)T (y - X w) + (lam/2) (wT w)
    error = (1.0 / N) * ( errorCalcT.dot(errorCalc) ) + regTerm
    
    # lam XT
    XTlam = X.transpose() * lambd
    # (y - X w)
    derivErrorCalc = np.asarray(y - yPredicted)
    # w - lam XT (y - X w)
    error_grad = w - XTlam.dot(derivErrorCalc)
    
    error_grad = np.squeeze(np.asarray(error_grad))
    
    return error, error_grad


def mapNonLinear(x,p):
    Xd = np.asarray(x) ** p
    
    Xd = np.matrix(Xd).transpose()
    
    return Xd


# Main script

# REPLACE THESE PATHS FOR ONES THAT WORK FOR YOU!
sample_pickle_path = "C:\\Users\\bond\\Downloads\\proj2\\sample.pickle"
diabetes_pickle_path = "C:\\Users\\bond\\Downloads\\proj2\\diabetes.pickle"

# load the sample data                                                                 
#X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')            
X,y,Xtest,ytest = pickle.load(open(sample_pickle_path,'rb'))

# Problem 1
# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))


#X,y,Xtest,ytest = pickle.load(open(diabetes_pickle_path,'rb'),encoding = 'latin1')   
X,y,Xtest,ytest = pickle.load(open(diabetes_pickle_path,'rb'))

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)


# Problem 2
w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))


# Problem 3
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,X_i,y)
    i = i + 1
plt.plot(lambdas,rmses3)


# Problem 4
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,X_i,y)
    i = i + 1
plt.plot(lambdas,rmses4)
plt.legend(('3','4'))
plt.show()


# Problem 5
k = 101
lambdas = np.linspace(0, 0.004, num=k)
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()
