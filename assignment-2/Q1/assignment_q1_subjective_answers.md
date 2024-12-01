# Q1 Gradient Descent

<div style="display:flex;">
  <img src="Dataset1.png" alt="Dataset 1" width="45%" />
  <img src="Dataset2.png" alt="Dataset 2" width="45%" />
</div>

## Full batch Gradient descent on Dataset 1
<div style="display:flex;">
  <img src="FBGD1.png" alt="FBGD1" width="45%" />
  <img src="FBGD1_1.png" alt="FBGD1_1" width="45%" />
</div>

## Full batch Gradient descent on Dataset 2
<div style="display:flex;">
  <img src="FBGD2.png" alt="FBGD2" width="45%" />
  <img src="FBGD2_1.png" alt="FBGD2_1" width="45%" />
</div>

## Stochastic Gradient descent on Dataset 1
<div style="display:flex;">
  <img src="SGD1.png" alt="SGD1" width="45%" />
  <img src="SGD1_1.png" alt="SGD1_1" width="45%" />
</div>

## Stochastic Gradient descent on Dataset 2
<div style="display:flex;">
  <img src="SGD2.png" alt="SGD2" width="45%" />
  <img src="SGD2_1.png" alt="SGD2_1" width="45%" />
</div>

## Full batch Gradient descent with momentum on Dataset 1
<div style="display:flex;">
  <img src="FBGDM1.png" alt="FBGDM1" width="45%" />
  <img src="FBGDM1_1.png" alt="FBGDM1_1" width="45%" />
</div>

## Full batch Gradient descent with momentum on Dataset 2
<div style="display:flex;">
  <img src="FBGDM2.png" alt="FBGDM2" width="45%" />
  <img src="FBGDM2_1.png" alt="FBGDM2_1" width="45%" />
</div>

## Stochastic Gradient descent with momentum on Dataset 1
<div style="display:flex;">
  <img src="SGDM1.png" alt="SGDM1" width="45%" />
  <img src="SGDM1_1.png" alt="SGDM1_1" width="45%" />
</div>

## Stochastic Gradient descent with momentum on Dataset 2
<div style="display:flex;">
  <img src="SGDM2.png" alt="SGDM2" width="45%" />
  <img src="SGDM2_1.png" alt="SGDM2_1" width="45%" />
</div>


## Result
![Result](Result.png)


#### * Which dataset and optimizer takes a larger number of epochs to converge, and why?
Full batch gradient descent on dataset 2 takes a larger number of epochs to converge because dataset 2 is more scattered from true function than dataset 1. A more scattered dataset may have a wider range of patterns and variations that the model needs to learn, which could require more iterations through the data to capture most precise theta values. Also stochastic gradient descent often converges faster than full batch gradient descent, especially in the early stages of training. The more frequent updates allow the model to adapt quickly to the training data. So if number of features and samples are very large then for faster computational time stochastic gradient descent is used.

