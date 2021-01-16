
import numpy as np
from matplotlib import pyplot as plt
import random
import math
import random
from tqdm import tqdm

""" Generating Billing Data. """
y_d = [0, 0]
x = []

a1=[2,3,4]
a2=[1,2,3]
a1=np.array(a1)
a2=np.array(a2)
x.append(a1)
x.append(a2)


for i in range(2, 100):
    a = []
    a.append(0.8 - 0.5 * math.exp(-((y_d[i-1])**2)))
    a.append((0.3 + 0.9 * math.exp(-(y_d[i-1]**2)))*(y_d[i-2]))
    a.append(0.1 * math.pi * math.sin(math.pi*(y_d[i-1])))
    a = np.array(a)
    x.append(a)
    y_d.append((a[0] + a[1]+a[2] + random.random()))



def data_vectorizer(data_to_vector, n):
    final = [a.reshape((n, 1)) for a in data_to_vector]
    return final


def datayi_karistir_test_ve_egitimi_ayir(x, y):
    c = list(zip(x, y))
    random.shuffle(c)
    (x_shuffled, yd_shuffled) = zip(*c)
    x_egitim = x_shuffled[0:80]
    x_test = x_shuffled[80:100]
    yd_egitim = yd_shuffled[0:80]
    yd_test = yd_shuffled[80:100]
    return x_egitim, x_test, yd_egitim, yd_test


x_egitim, x_test, yd_egitim, yd_test = datayi_karistir_test_ve_egitimi_ayir(
    x, y_d)
print('y_d')
print(np.shape(y_d))
print('x')
print(np.shape(x))

x_egitim = data_vectorizer(x_egitim, 3)

""" 
yd_test = data_vectorizer(yd_test, 1)
yd_egitim = data_vectorizer(yd_egitim, 1) 
"""

x_test = data_vectorizer(x_test, 3)

def v_to_x(v):
    
    return v

def derivative_v_to_x(v):
    return v

weights_u = np.random.randn(7, 3) 
weights_x = np.random.randn(7, 7) 
weights_y = np.random.randn(1, 7) 
epoch = 1200
learning_rate = 0.5
epoch_iterator = tqdm(range(epoch))

for e in range(1200):
    for i, (u, yd) in enumerate(zip(x_egitim, yd_egitim)):

        #forward
        if e==0 and i==0:
            x_k = np.random.randn(7,1)
        
        V_k = np.dot(weights_x, x_k) + np.dot(weights_u, u)
        x_k = v_to_x(V_k)
        y_k = np.dot(weights_y , x_k)

        #hata
        e = yd - y_k
        E = 0.5 * e.T * e
        """         
        print('yd')
        print((yd))
        print('y_k')
        print(np.shape(y_k))        
        
       
        print('weights_y.T')
        print(np.shape(weights_y.T))
        print('********')
        print('e')
        print(np.shape(e))
        print('********')   
        print('np.dot(weights_y.T, e)')
        print(np.shape(np.dot(weights_y.T, e)))
        print('********')  
        print('derivative_v_to_x(V_k)')
        print(np.shape(derivative_v_to_x(V_k)))
        print('********')   
        print('x_k.T')
        print(np.shape(x_k.T))
        print('********')  
        """     
        
        #agırlık güncellenmesi
        weights_x = weights_x + learning_rate * np.dot(((np.dot(weights_y.T, e))*derivative_v_to_x(V_k)), x_k.T)
        weights_u = weights_u + learning_rate * np.dot(((np.dot(weights_y.T, e))*derivative_v_to_x(V_k)), u.T)
        weights_y = weights_y + learning_rate * e * x_k.T
        print(e)


        