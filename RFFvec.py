import numpy as np
import time
#Vectorized RFF for Gauss
def RFF_Gauss1(N, d, P):
    x = np.random.rand(N, d)
    x_weights = np.random.randn(N, 1)
    variance = 1 / (4 * np.pi**2)
    covariance_matrix = np.eye(d)
    mean = np.zeros(d)
    

    xi = np.random.multivariate_normal(mean, covariance_matrix, size=P)
    b = np.random.uniform(low=0.0, high=2 * np.pi, size=P)


    start_part1 = time.time()
    

    cos_b_matrix = np.dot(x, xi.T) + b  
    cos_b = x_weights * np.cos(cos_b_matrix)
    B = np.sum(cos_b, axis=0)  
    
    
    cos_A_matrix = np.dot(xi, x.T) + b[:, np.newaxis]  
    cos_A = (2 / P) * np.dot(B, np.cos(cos_A_matrix))  
    

    end_part1 = time.time()
    time_part1 = end_part1 - start_part1


    start_part2 = time.time()

    x_squared_sum = np.sum(x**2, axis=1).reshape(-1, 1)
    pairwise_diffs = x_squared_sum + x_squared_sum.T - 2 * np.dot(x, x.T)
    norms = np.sqrt(np.maximum(pairwise_diffs, 0))
    kernel_matrix = np.exp(-(norms**2) / 2)


    naive_sum = np.dot(kernel_matrix, x_weights)
    Naive = naive_sum.flatten()  
    
   
    end_part2 = time.time()
    time_part2 = end_part2 - start_part2
    
   

    error = np.sum(np.abs(Naive - cos_A))  
    rel_error = error / (np.sum(np.abs(x_weights)) * N)
    


    return rel_error, time_part1, time_part2,N,d,P

def test(N,d,P,M):
    errors1=np.zeros(M)
    runtime1=np.zeros(M)
    runtime2=np.zeros(M)
    for i in range(M):
        errors1[i]+=RFF_Gauss1(N,d,P)[0]
        runtime1[i]+=RFF_Gauss1(N,d,P)[1]
        runtime2[i]+=RFF_Gauss1(N,d,P)[2]
    print(N,d,P)

    return np.mean(errors1),np.mean(runtime1),np.mean(runtime2)

for i in range(20):
    print(test((i+1)*150,500,300,10))

#print(test(1000,500,5000,10))

