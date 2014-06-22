Practical Machine Learnign Course Project
========================================================

First we load the data and set the outcome class as a factor. 


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pmltrain <- read.csv("pml-training.csv")
pmltest <- read.csv("pml-testing.csv")
pmltrain$classe <- as.factor(pmltrain$classe)
```

Now we do some basic preprocessing.  Although there are 160 columns in the data sets, not all columns are appropriate for use as predictors.  The first seven columns appear to contain meta-data; any correlations that may happen to exist between these columns and the outcome class are unlikely to be meaningful.  Additionally, there are many columns that contain missing values in the test set, and it makes no sense to predict based on features not present in the test data. We identify these columns and remove them from our model.


```r
metacols <- seq(1,7)
# max will return NA for any column with missing values
badcols <- which(is.na(apply(pmltest,2,max)))
redtrain <- pmltrain[,-c(metacols,badcols)]
dim(redtrain)
```

```
## [1] 19622    53
```

We see that we now have only 53 columns, including the outcome class.

Next, we separate the training data into a training and validation sets.


```r
intrain <- createDataPartition(y=redtrain$classe,p=0.7,list=FALSE)
training <- redtrain[intrain,]
val <- redtrain[-intrain,]
```

Now we train a random forest model.  We then assess it against the training data.


```r
rfmod1 <- train(classe~.,data=training,method="rf", 
                trControl=trainControl("oob",allowParallel=TRUE))
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
trainpred <- predict(rfmod1,training)
ptab1 <- table(trainpred,training$classe)
confusionMatrix(ptab1)
```

```
## Confusion Matrix and Statistics
## 
##          
## trainpred    A    B    C    D    E
##         A 3906    0    0    0    0
##         B    0 2658    0    0    0
##         C    0    0 2396    0    0
##         D    0    0    0 2252    0
##         E    0    0    0    0 2525
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

To estimate the out-of-sample error, we predict classes for the validation set and compare against the known classes.


```r
valpred <- predict(rfmod1,val)
ptab2 <- table(valpred,val$classe)
confusionMatrix(ptab2)
```

```
## Confusion Matrix and Statistics
## 
##        
## valpred    A    B    C    D    E
##       A 1672    5    0    0    0
##       B    1 1129    9    0    0
##       C    1    4 1012    7    1
##       D    0    1    5  956    2
##       E    0    0    0    1 1079
## 
## Overall Statistics
##                                         
##                Accuracy : 0.994         
##                  95% CI : (0.991, 0.996)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.992         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.991    0.986    0.992    0.997
## Specificity             0.999    0.998    0.997    0.998    1.000
## Pos Pred Value          0.997    0.991    0.987    0.992    0.999
## Neg Pred Value          1.000    0.998    0.997    0.998    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.172    0.162    0.183
## Detection Prevalence    0.285    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.995    0.992    0.995    0.999
```

We thus estimate less than 1% error on new data.  Finally, we generate predictions for the test data.


```r
testpred <- predict(rfmod1,pmltest)
```

Upon submission of these results to the grading system, we obtain 100% accuracy.

