import numpy as np
import matplotlib.pyplot as plt
from random import *

def read_file():
    datas = []
    classes = []
    with open("data.txt") as f:
        for line in f:
            datas.append([int(i) for i in line.split(',')])
            classes.append(datas[-1][-1])
            del(datas[-1][-1])
    return datas,classes


def main():
    classes = []
    datas = []
    datas2 = []
    datas3 = []
    means = []
    datas,classes = read_file()
    means = np.mean(datas,axis=0,dtype=np.float64)
    for i in range(len(datas)):
        datas2.append(datas[i]-means)

    datas3 = np.cov(datas2,rowvar=False)
    eigen = np.linalg.eig(datas3)
    eigen1 = np.transpose(eigen[1])[0]
    eigen2 = np.transpose(eigen[1])[1]
    ureduce = []
    ureduce.append(eigen1)
    ureduce.append(eigen2)
    z = []
    for x in datas:
        z.append(np.matmul(ureduce,x))
    axis_x = []
    axis_y = []
    for x in z:
        axis_x.append(-1 * x[0]) #because np.linalg.eig() has some sign problems on the values
        axis_y.append(x[1])
    for i in range(200):
        rando = randint(0,3822)
        plt.annotate(classes[rando], xy=(axis_x[rando],axis_y[rando]),fontsize=9)
    plt.plot(axis_x,axis_y,'ro',markersize=2)
    plt.show()

if __name__== "__main__":
    main()
