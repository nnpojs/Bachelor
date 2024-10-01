import scipy.optimize as opt
import scipy
import time
from scipy.integrate import quad
import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
import invertcdf
from invertcdf import sample
from invertcdf import sample_sphere
import SlicedRFF
from SlicedRFF import RFF_Gauss1
from SlicedRFF import test


def compare(N,d,P,Q):
    #point, weight generation
    x=np.random.rand(N,d)
    x_weights=np.random.randn(N,1)

    startnaive=time.time()
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
    endnaive=time.time()
    timenaive=endnaive-startnaive
    Naive = [float(x) for x in SumN]

    #sample P directions
    Ps=sample_sphere(d,P)

    #sample P*Q r_pq
    r_pq=sample(d,P*Q)
    start=time.time()
    #calculate B part of f with RFF
    P_list=[[] for _ in range(P)]
    high=2*np.pi
    b=np.random.uniform(low=0.0, high=high, size=Q)
    
    Brlist=[]
    Sm=[]
    list2=[]
    for p in range(P):
        for q in range(Q):
            Br=0
            for n in range(N):
                Br+=x_weights[n]*np.cos(r_pq[p*Q+q]*np.dot(x[n]*2*np.pi,Ps[p])+b[q])
            Brlist.append(Br)
    #print(Brlist)
    for m in range(N):
        Sump=0
        for p in range(P):
            for q in range(Q):
                Sump+=(2/Q)*np.cos(r_pq[p*Q+q]*np.dot(x[m]*2*np.pi,Ps[p])+b[q])*Brlist[p*Q+q]
            #print(Sump)
        Sm.append((1/P)*Sump)
    end=time.time()
    time1=end-start
    Sm_float = [sm.item() for sm in Sm]
    error=np.zeros(N)
    for i in range(N):
        error[i]+=np.abs((Naive[i]-Sm_float[i]))
    rel_error = np.sum(error) / (N*np.sum(np.abs(x_weights)))
    #rel_error = np.mean(error)
    #print(Naive)
    #print(Sm_float)
    return rel_error,time1,timenaive


def test1(N,d,P,Q,M):
    errors1=np.zeros(M)
    runtime1=np.zeros(M)
    runtime2=np.zeros(M)
    for i in range(M):
        errors1[i]+=compare(N,d,P,Q)[0]
        runtime1[i]+=compare(N,d,P,Q)[1]
        runtime2[i]+=compare(N,d,P,Q)[2]
    print(N,d,P,Q)

    return np.mean(errors1),np.mean(runtime1),np.mean(runtime2)


        
         
