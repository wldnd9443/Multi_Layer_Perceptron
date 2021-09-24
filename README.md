# Multi Layer Perceptron
> Multi Layer Perceptron 구현해보기

Multi Layer Perceptron을 tensorflow 없이 직접 구현해보는 코드입니다.

## 소개
  이 Repository에서는 Linear Regression이나 Logsitic Regression으로는 대표할 수 없는 Data Set을 해석할 수 있는 Multi Layer Perceptron을 직접 구현하고 해석하는 코드를 다룰 것입니다. 데이터들이 항상 선형적인 경향성을 가질 수는 없기 때문입니다. XOR형태의 데이터를 다룰 때 그러합니다.
  
  위와 같은 형식의 데이터는 선형의 그래프도, Sigmoid 함수도 대표할 수 있는 그래프가 없습니다. 
  
![MLP_dataplt](https://user-images.githubusercontent.com/44831709/134707491-6f7679ae-d0ce-4d74-b6bd-7e8045c15f21.png)

  그렇다고 아래와 같은 형태로 분류를 하기에는 기준도 형식도 모호합니다.
  
![MLP_datasplit_bad](https://user-images.githubusercontent.com/44831709/134708173-7820cd5e-a6ed-45d1-b9a5-b3e3f1492286.png)

  이에 대해 우리는 Multi Layer Perceptron으로 해결책을 제시할 수 있습니다. 
Perceptron은 사람의 신경세포를 본떠 만든 인공 신경망을 이야기 합니다. Linear나 Logistic Regression에서 Weight와 Bias 각각 하나씩 가져와서 이야기 했다면 Perceptron은 여러개의 Weight와 Bias에 따라 그 출력값을 결정합니다. 
