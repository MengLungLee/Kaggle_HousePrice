# House Prices: Advanced Regression Techniques(Kaggle Competition)
For this project, I run overall processes from Pre-process data, Exploring data analysis, Feature Selection, Building Model, and Validation Model(hidden layer and activation function).

I applied Neural Network Regression model(NN) to handle this project. Although I did not get a good score on Kaggle Competition, I learn the whole process and understand each section what has been done, including updating the data and scripts on Github.

## Dataset
Goal: Predict Houses' SalePrice.

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

## Toolkit
Pandas  Numpy Scikit-learn  Pytorch

### Exploratory Data Analysis 
At first, I checked all features and then considered the Sale Price of house that would be related to the houses' age.
Thus, I created a new feature, House_ages, through 'year sold' deducting 'year built' and group it.

<img src="https://github.com/MengLungLee/Kaggle_HousePrice/blob/master/EDA_screenshot/house_ages.png" width="300" height="200">


As we can see, the region is also associated with Sale Price and then dummies by pandas.

<img src="https://github.com/MengLungLee/Kaggle_HousePrice/blob/master/EDA_screenshot/region.png" width="300" height="200">

Finally, I selected the top 15 features of high coefficient with SalePrice through heatmap.
According to occam's razor rule, I decided to drop some features that similar to each other, making the model much simple.

<img src="https://github.com/MengLungLee/Kaggle_HousePrice/blob/master/EDA_screenshot/top15_heatmap.png" width="350" height="250">

### Model building and Model Performance

At first, I spilt the data into training and validation sets with a validation size of 30% (on validation.py)

I set up some parameters below and followed the competition for the loss function that used the RMSE.

batch_size = 120, epoch = 1000, optimizer = Adam, learning rate = 0.001, 4 hidden layers with Relu.

I tried many times by setting a neural network.

<img src="https://github.com/MengLungLee/Kaggle_HousePrice/blob/master/ModelBuilding_screenshot/Validation%20loss.png" width="250" height="200">

RMSE of Validation set: 0.1526

After that, I applied the same parameters to train whole training sets and then predicted the test sets

<img src="https://github.com/MengLungLee/Kaggle_HousePrice/blob/master/ModelBuilding_screenshot/output.png" width="300" height="200">

RMSE of whole training sets: 0.0975

### Concluded

Finally, I got 3387/4701 on the leaderboard with a score of 0.15322, which means it is not a good outcome. (09/28/2020)

I listed some problems that might be resulted in this bad grade.

1. the number of datasets is not enough for NN learning good.
2. I should not drop the features because NN could drop up useless features while learning datasets.
3. It might be trapped into local minima and saddle points.
4. NN is hard to set up hyperparameters and easy to overfitting and I confronted it as well.
