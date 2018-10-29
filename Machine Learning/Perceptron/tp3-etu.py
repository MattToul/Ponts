import  numpy as np
import matplotlib.pyplot as plt
from tools import *

def Classifier(object):
    def __init__(self):
        pass
    def predict(self,data):
        pass
    def fit(self,data,labels):
        pass
    def score(self,data,labels):
        return (self.predict(data)==labels).mean()


def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    """ l : liste des chiffres a extraire"""
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   zip(*[get_usps(i,datax,datay) for i in l])
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    idx = np.random.permutation(range(len(tmpy)))
    return tmpx[idx,:],tmpy[idx]

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")


### Donnees artificielles
plt.ion()
xgentrain,ygentrain = gen_arti(data_type=0,sigma=0.5,nbex=1000,epsilon=0.1)
xgentest,ygentest = gen_arti(data_type=0,sigma=0.5,nbex=1000,epsilon=0.1)
plt.figure()
plot_data(xgentrain,ygentrain)

### Donnees reelles
plt.figure()
xuspstrain,yuspstrain = load_usps("USPS_train.txt")
xuspstest,yuspstest = load_usps("USPS_test.txt")
x06train,y06train = get_usps([0,6],xuspstrain,yuspstrain)
x06test,y06test = get_usps([0,6],xuspstest,yuspstest)
show_usps(x06train[0])




#### Pour la visualisation des couts
#### En 2D, f etant une fonction de R^2 -> R
grid,xx,yy = make_grid(xmin=-1,xmax=3,ymin=-1,ymax=3,step=50)
plt.figure()
plt.contourf(xx,yy,f(grid).reshape(xx.shape),256)
