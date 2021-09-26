# Multi Layer Perceptron
> Multi Layer Perceptron 구현해보기

Multi Layer Perceptron을 tensorflow 없이 직접 구현해보는 코드입니다.

## 소개
이 Repository에서는 Linear Regression이나 Logsitic Regression으로는 대표할 수 없는 Data Set을 해석할 수 있는 Multi Layer Perceptron을 직접 구현하고 해석하는 코드를 다룰 것입니다. 데이터들이 항상 선형적인 경향성을 가질 수는 없기 때문입니다. XOR형태의 데이터를 다룰 때 그러합니다.
  
위와 같은 형식의 데이터는 선형의 그래프도, Sigmoid 함수도 대표할 수 있는 그래프가 없습니다. 
  
![MLP_dataplt](https://user-images.githubusercontent.com/44831709/134707491-6f7679ae-d0ce-4d74-b6bd-7e8045c15f21.png)

아래와 같은 선으로 데이터를 분류해야 합니다. 어떻게 이런 선을 정의할 수 있을까요?
  
![MLP_datasplit_bad](https://user-images.githubusercontent.com/44831709/134708173-7820cd5e-a6ed-45d1-b9a5-b3e3f1492286.png)

이에 대해 우리는 Multi Layer Perceptron으로 해결책을 제시할 수 있습니다. 
Multi Layer Perceptron은 기존과 다르게 직선이 아닌 복잡한 형태의 그래프로 가르기 때문에 다수의 Weight와 Bias가 사용되어야 합니다.

![MLP_structure](https://user-images.githubusercontent.com/44831709/134811088-c3aeeb28-75ff-4ad0-804e-d82a4e377208.png)

위와 같은 형태의 Multi Layer Perceptron을 직접 구현하는 과정을 소개하려 합니다.

## 구현 과정
### Data Generation

```python
import numpy as np
import matplotlib.pyplot as plt

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
```

![MLP_dataplt](https://user-images.githubusercontent.com/44831709/134707491-6f7679ae-d0ce-4d74-b6bd-7e8045c15f21.png)

한가지 선으로 데이터를 분류하기 어렵도록 배치합니다. (XOR과 유사합니다.)

모델은 Fully Connected Layer로 2:3:3:2의 구조를 갖는걸로 설정합니다. Activation Function은 Sigmoid로 설정합니다. 초기값은 임의로 설정합니다.
```
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
Bs.append(np.random.random((1,No)))
```
