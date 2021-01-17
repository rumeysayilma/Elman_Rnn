
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


#giriş datası eğitim ve test kümesi olarak önce karıştırılarak ayrıştırılır
def datayi_karistir_test_ve_egitimi_ayir(x, y):
    c = list(zip(x, y))
    random.shuffle(c)
    (x_shuffled, yd_shuffled) = zip(*c)
    x_egitim = x_shuffled[0:80]
    x_test = x_shuffled[80:100]
    yd_egitim = yd_shuffled[0:80]
    yd_test = yd_shuffled[80:100]
    return x_egitim, x_test, yd_egitim, yd_test

#oluşturulan fonksiyon çalıştırılır.
x_egitim, x_test, yd_egitim, yd_test = datayi_karistir_test_ve_egitimi_ayir(
    x, y_d)
print('y_d')
print(np.shape(y_d))
print('x')
print(np.shape(x))

#fonksiyon olarak sigmoid kullanılır
def v_to_x(x):
    t = 1/(1 + np.exp(-x))
    return t

#gradyan hesabı için sigmoidin türev fonksiyonu oluşturulur.
def derivative_v_to_x(x):
    dt = v_to_x(x)
    return dt*(1-dt)

#başlangıç ağırlıkları atanır
weights_u = np.random.randn(5, 1) 
weights_x = np.random.randn(5, 5) 
weights_y = np.random.randn(1, 5) 

#iterasyon sayısı belirlenir
epoch = 50

#öğrenme hızı belirlenir
learning_rate = 0.5

epoch_iterator = tqdm(range(epoch))

#durum portresi için yd ve y_k lar bir listede tutulacak. bunun için boş liste açılır
ydler = []
yklar = []

#karesel ortalama hata hesabı için e(hata) ler bir listede tutulacak. bunun için boş liste açılır
hatalar = []

for e in range(50):
    E=0
    for i, (u, yd) in enumerate(zip(x_egitim, yd_egitim)):

        #forward
        #gizli katmanın ilk elemanı rastgele atanır
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
        
        #agırlık güncellenmesi
        weights_x = weights_x + learning_rate * np.dot(((np.dot(weights_y.T, e))*derivative_v_to_x(V_k)), x_k.T)
        weights_u = weights_u + learning_rate * np.dot(((np.dot(weights_y.T, e))*derivative_v_to_x(V_k)), u)
        weights_y = weights_y + learning_rate * e * x_k.T
    hatalar.append(E /len(x_egitim) )


ydler_test = []
yklar_test = []
test_karesel_hata = []
#test 
for i, (u, yd) in enumerate(zip(x_test, yd_test)):

    V_k = np.dot(weights_x, x_k) + np.dot(weights_u, u)
    x_k = v_to_x(V_k)
    y_k = np.dot(weights_y , x_k)

    ydler_test.append(yd)
    yklar_test.append(y_k)

    #hata
    e = yd - y_k
    E = 0.5 * e**2
    test_karesel_hata.append(E)

ort_test_hatası = np.sum(test_karesel_hata)/ len(x_test)

plt.plot( range(len(test_karesel_hata)), np.reshape(test_karesel_hata,(len(test_karesel_hata))))  
plt.title('Test Kümesi Ortalama Karesel Hata = ' + str(ort_test_hatası)) 
plt.xlabel('Test Datası')
plt.ylabel('Ortalama Karesel Hata')

plt.show()

p = range(0, 20, 1)
l = np.reshape(yklar_test,(np.shape(ydler_test)))
plt.plot(p, l , label ='yk')


c = range(0, 20, 1)
q = np.reshape(ydler_test,(np.shape(ydler_test)))
plt.plot(c, q, label ='y_d')
plt.legend( loc ="lower right") 
plt.show()

"""
plt.plot( range(len(hatalar)), np.reshape(hatalar,(len(hatalar))))        
plt.show()
"""

"""
m = range(0, 4000, 1)
y = np.reshape(yklar,(np.shape(ydler)))
plt.plot(m, y )


z = range(0, 4000, 1)
t = np.reshape(ydler,(np.shape(ydler)))
plt.plot(z, t)

plt.show()
"""
