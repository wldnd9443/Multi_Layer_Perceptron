import numpy as np
import matplotlib.pyplot as plt

## Data Generation
NOISE = 0.02
mat_covs = np.array([[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]])*NOISE

mus =  np.array([[1,1],[0,0],[1,0],[0,1]])
Ns = np.array([400,400,400,400])
clss = [1,1,0,0]

X = np.zeros((0,mus.shape[1]))
Y = np.zeros(0)


for mu, mat_cov, N, cls in zip(mus, mat_covs, Ns, clss):
    X_ = np.random.multivariate_normal(mu, mat_cov, N)
    Y_ = np.ones(N)*cls
    X = np.vstack((X,X_))
    Y = np.hstack((Y,Y_))
    
cls_unique = np.unique(Y)

def plot_data(X,Y):
    legends = []
    for cls in cls_unique:
        idx = Y==cls
        plt.plot(X[idx,0],X[idx,1],'.')
        legends.append(cls.astype('int'))

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(legends)
    plt.grid(True)
    plt.xlim([-1,2])
    plt.ylim([-1,2])
    plt.show()
    
plot_data(X,Y)


## Training Step
# Fully Connected Layer 2:3:3:2
# Activation Function: Sigmoid
# Random Initialize
Ni = 2 # input layer
No = 2 # output layer
Nhs = [3,3]  # hidden layer

Ws = [] # weights
Bs = [] # biases
N_prev = Ni
for Nh in Nhs:
    Ws.append(np.random.random((N_prev,Nh)))
    Bs.append(np.random.random((1,Nh)))
    N_prev = Nh
Ws.append(np.random.random((N_prev,No)))


## Define Function y-hat, sigmoid, 
def sigm(Vin):
    Vout = 1/(1+np.exp(-Vin))
    return Vout

def diff_sigm(Vin):
    Vout = (-np.exp(-Vin))/(1+np.exp(-Vin))**-2
    return Vout

def predict(X,Ws,Bs):
    rst = X
    for W,B in zip(Ws,Bs):
        rst = rst@W+B
    return rst[:,0] < rst[:,1]
Bs.append(np.random.random((1,No)))

x1 = np.linspace(-1, 2, 100).reshape(-1)
x2 = np.linspace(-1, 2, 100).reshape(-1)
Xv1, Xv2 = np.meshgrid(x1, x2)

Xg = np.hstack((Xv1.reshape(-1)[:,None],Xv2.reshape(-1)[:,None]))
Yg_pred = predict(Xg,Ws,Bs)

z = Yg_pred.reshape(Xv1.shape)
plt.contourf(Xv1,Xv2,z)
plot_data(X,Y)
# plt.colorbar()
# plt.clim(0,1)
plt.show()

N_iter = 1
lr_alpha = 0.001

for it in range(N_iter):
    dif = (Y- predict(X,Ws,Bs))
    J = np.sqrt((dif**2).mean())
    
    
    curs = []
    diff_yhWs = [] # Derivative of y hat respect to W
    diff_yhBs = [] # Derivative of y hat respect to B
    
    for i in range(len(Ws)):
        cur = X
        diff_yhW = np.eye(cur.shape[0])
        for j in range(i+1):
            cur = cur @ Ws[j] + Bs[j]
            cur = sigm(cur)
            diff_yhW = diff_yhW @ cur
            print(j,cur.shape)
            diff_JW = -2*Y*
        Ws[i] = 

        diff_JW = -2*Y*
        Ws[i] = Ws[i] - alpha * diff_yhWs[i]
        Bs[i] = Bs[i] - alpha * diff_yhBs[i]

    for W,B in zip(Ws,Bs):
        diff_yhWs.append(cur)
        cur = cur@W+B
        curs.append(cur)

        diff_yhBs.append(diff_sigm(cur))
        cur = sigm(cur)

    for W,B,diff_JW,diff_JB in zip(Ws,Bs,diff_JW,diff_JB):
        W = W - alpha * diff_JW
        B = B - alpha * diff_JW

    for i in range(len(Ws)):
        diff_JW = -2*Y*
        Ws[i] = Ws[i] - alpha * diff_yhWs[i]
        Bs[i] = Bs[i] - alpha * diff_yhBs[i]
    print(J)
