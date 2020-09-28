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

### Model building

