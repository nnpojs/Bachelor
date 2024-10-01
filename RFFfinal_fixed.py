import math
import numpy as np
import scipy.io
import time


def RFF_Gauss1(N,d,P):
    x=np.random.rand(N,d)
    x_weights=np.random.randn(N,1)
    variance = 1 / (4 * np.pi**2)
    standard_deviation = np.sqrt(variance)
    covariance_matrix=np.eye(d)
    mean = np.zeros(d)
    sigma=1/ (2*np.pi)
    xi=np.random.multivariate_normal(mean, covariance_matrix, size=P)
    high=2*np.pi
    b=np.random.uniform(low=0.0, high=high, size=P)
    start_part1 = time.time()
    #calculate part B
    B1=0
    B=[]
    for i in range(P):
        cos_b=[]
        for n in range(N):
            cos_n=x_weights[n]*np.cos(np.dot(x[n], xi[i]) + b[i])
            cos_b.append(cos_n)
        B1=np.sum(cos_b)
        B.append(B1)

    #calculate sum s_m
    cos_a=[]
    cos_A=[]
    for m in range(N):
        cos_a=[]
        for i in range(P):
            cos_i=np.cos(np.dot(xi[i],x[m])+b[i])*B[i]
            #print(np.dot(xi[i],x[m]))
            cos_a.append((2/P)*cos_i)
        Sum1=np.sum(cos_a)
        cos_A.append(Sum1)
    end_part1 = time.time()
    time_part1 = end_part1 - start_part1
    RFF=[float(x) for x in cos_A]  

    start_part2 = time.time()
    #Naive
    SumN=[]
    for m in range(N):
        Sum_m=[]
        for i in range(N):
            diff=x[m]-x[i]
            norm=np.linalg.norm(diff)
            ker=np.exp(-(norm**2)/2)
            Sum_m.append(x_weights[i]*ker)
        SumN.append(np.sum(Sum_m))
    end_part2 = time.time()
    time_part2 = end_part2 - start_part2
    Naive = [float(x) for x in SumN]
    #print(x_weights)

    #print(Naive)
    #print(RFF)

    error=[]
    for i in range(N):

        error.append(np.abs(Naive[i]-RFF[i]))
            

    rel_error1=error/(np.abs(np.sum(x_weights))*N)
    rel_error=np.mean(rel_error1)
    #print(error)
    #rel_error = np.sum(error / (np.sum(np.abs(x_weights)) * N))
    #rel_error = np.mean(error)
    #print(Naive)
    #print(RFF)
    return rel_error,time_part1,time_part2


def test(N,d,P,M):
    errors1=np.zeros(M)
    runtime1=np.zeros(M)
    runtime2=np.zeros(M)
    for i in range(M):
        errors1[i]+=RFF_Gauss1(N,d,P)[0]
        runtime1[i]+=RFF_Gauss1(N,d,P)[1]
        runtime2[i]+=RFF_Gauss1(N,d,P)[2]

    return np.mean(errors1),np.mean(runtime1),np.mean(runtime2)


'''
print(test(1000,500,50,10))
print(test(1000,500,100,10))
print(test(1000,500,150,10))
print(test(1000,500,200,10))
print(test(1000,500,250,10))
print(test(1000,500,300,10))
print(test(1000,500,350,10))
print(test(1000,500,400,10))
print(test(1000,500,450,10))
print(test(1000,500,500,10))'''

