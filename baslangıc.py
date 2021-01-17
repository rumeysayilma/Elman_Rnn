
import numpy as np
from matplotlib import pyplot as plt
import random
import math
import random
from tqdm import tqdm

""" Generating Billing Data. """
y_d = [0, 0]
x = [1,2]


for i in range(2, 100):
    random_noise = random.random()
    a1 = (0.8 - 0.5 * math.exp(-((y_d[i-1])**2)))
    a2 = ((0.3 + 0.9 * math.exp(-(y_d[i-1]**2)))*(y_d[i-2]))
    a3 = (0.1 * math.pi * math.sin(math.pi*(y_d[i-1])))
    x.append(random_noise)
    y_d.append((a1 + a2 + a3 + random_noise))



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

#x_egitim = data_vectorizer(x_egitim, 1)


#yd_test = data_vectorizer(yd_test, 1)
#yd_egitim = data_vectorizer(yd_egitim, 1) 


#x_test = data_vectorizer(x_test, 1)

def v_to_x(x):
    #t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))   
    #t = np.tanh(x) 
    t = 1/(1 + np.exp(-x))
    return t

def derivative_v_to_x(x):
    dt = v_to_x(x)
    return dt*(1-dt)

weights_u = np.random.randn(5, 1) 
weights_x = np.random.randn(5, 5) 
weights_y = np.random.randn(1, 5) 
epoch = 50
learning_rate = 0.5
epoch_iterator = tqdm(range(epoch))
ydler = []
yklar = []
hatalar = []
for e in range(50):
    E=0
    for i, (u, yd) in enumerate(zip(x_egitim, yd_egitim)):

        #forward
        if e==0 and i==0:
            x_k = np.random.randn(5,1)
        
        V_k = np.dot(weights_x, x_k) + np.dot(weights_u, u)
        x_k = v_to_x(V_k)
        y_k = np.dot(weights_y , x_k)
        ydler.append(yd)
        yklar.append(y_k)
        #hata
        e = yd - y_k
        E += 0.5 * e**2
        
        


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
        weights_u = weights_u + learning_rate * np.dot(((np.dot(weights_y.T, e))*derivative_v_to_x(V_k)), u)
        weights_y = weights_y + learning_rate * e * x_k.T
    hatalar.append(E /len(x_egitim) )
"""
np.reshape(yklar,(np.shape(ydler)))
plt.plot(ydler, np.reshape(yklar,(np.shape(ydler))), color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12)
plt.show()
"""

plt.plot( range(len(hatalar)), np.reshape(hatalar,(len(hatalar))))        
plt.show()
