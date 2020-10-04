---
title: "Project Assignment"
output:
  html_document:
    keep_md: yes
---






## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Introduction 

The objective of our report is to find a good model to predict the outcome "classe" of the dataset that we have. 
The outcome variable is a factor with 5 levels. Participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different fashions:

- exactly according to the specification (Class A)
- throwing the elbows to the front (Class B)
- lifting the dumbbell only halfway (Class C)
- lowering the dumbbell only halfway (Class D)
- throwing the hips to the front (Class E)

## Libraries and Data processing 

This are the packages used in this report, all necessary to use the models tested here.


```r
library("caret")
library("randomForest")
library("rpart")
library("gbm")
```

We suppose here that you are already in your work repository, and since we already downloaded the data, we didn't actually run that code. After that we read 
the csv files and we take out the NA values, then we take out the first seven columns since this aren't variables that can be used to predict the model. We put the variable "classe" as factor since the data doesn't treat it as a factor.


```r
#train_url    <- 'http://d396qusza40orc.cloudfront.net/predmachlearn/pml-train.csv'
#test_url  <- 'http://d396qusza40orc.cloudfront.net/predmachlearn/pml-test.csv'
#download.file(train_url,"train.csv")
#download.file(test_url,"test.csv")

train <- read.csv("train.csv", na.strings=c("NA","#DIV/0!", ""))
test <- read.csv("test.csv" , na.strings=c("NA", "#DIV/0!", ""))
train<-train[,colSums(is.na(train)) == 0]
test <-test[,colSums(is.na(test)) == 0]
train$classe <- as.factor(train$classe)

train   <-train[,-c(1:7)]
test <-test[,-c(1:7)]
```

After making the changes on the data we will separate the training data into two separate data.frames to test our models (cross-validation).



```r
set.seed(278910)

crossvalidation <- createDataPartition(y=train$classe, p=0.80, list=FALSE)
cv_train <- train[crossvalidation, ] 
cv_test <- train[-crossvalidation, ]
```

## Prediction models 

The first model that we fit into the data is a simple decision tree model, after training the model we test it in the cross-validation section for it, and we can see the results in the confusion matrix ploted below. With an accuracy below 90% we decide to test another prediction model for the data.


```r
fit_dt <- rpart(classe ~ ., data=cv_train, method="class")

predict_dt <- predict(fit_dt, cv_test, type = "class")

confusionMatrix(predict_dt, cv_test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1005  118   42   66   13
##          B   41  483   55   72   60
##          C   25   62  517   58   60
##          D   37   54   57  359   47
##          E    8   42   13   88  541
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7405          
##                  95% CI : (0.7265, 0.7542)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6703          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9005   0.6364   0.7558  0.55832   0.7503
## Specificity            0.9149   0.9279   0.9367  0.94055   0.9528
## Pos Pred Value         0.8079   0.6793   0.7161  0.64801   0.7818
## Neg Pred Value         0.9586   0.9141   0.9478  0.91570   0.9443
## Prevalence             0.2845   0.1935   0.1744  0.16391   0.1838
## Detection Rate         0.2562   0.1231   0.1318  0.09151   0.1379
## Detection Prevalence   0.3171   0.1812   0.1840  0.14122   0.1764
## Balanced Accuracy      0.9077   0.7822   0.8463  0.74943   0.8516
```

The second model tested is the random forest model, and as we can see in the confusion matrix, the results are much better than the last model. Still, we will try one more model for classification and see if we can get better results.


```r
fit_rf <- randomForest(classe ~ ., data=cv_train)

predict_rf <- predict(fit_rf, cv_test, type = "class")

confusionMatrix(predict_rf, cv_test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1114    5    0    0    0
##          B    2  751    2    0    0
##          C    0    3  682    5    0
##          D    0    0    0  638    3
##          E    0    0    0    0  718
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9921, 0.9969)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9895   0.9971   0.9922   0.9958
## Specificity            0.9982   0.9987   0.9975   0.9991   1.0000
## Pos Pred Value         0.9955   0.9947   0.9884   0.9953   1.0000
## Neg Pred Value         0.9993   0.9975   0.9994   0.9985   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1914   0.1738   0.1626   0.1830
## Detection Prevalence   0.2852   0.1925   0.1759   0.1634   0.1830
## Balanced Accuracy      0.9982   0.9941   0.9973   0.9957   0.9979
```


The third and last model that we test is a gradient boost model. This model was better than the decision tree, but the random forest model is better.


```r
fit_gbm<- train(classe~., data=cv_train, method="gbm", verbose= FALSE)

predict_gbm<- predict(fit_gbm, cv_test)

confusionMatrix(predict_gbm, cv_test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1100   22    0    1    2
##          B   11  720   18    2   11
##          C    4   16  660   23   10
##          D    1    1    5  611   10
##          E    0    0    1    6  688
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9633         
##                  95% CI : (0.9569, 0.969)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9536         
##                                          
##  Mcnemar's Test P-Value : 1.104e-05      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9857   0.9486   0.9649   0.9502   0.9542
## Specificity            0.9911   0.9867   0.9836   0.9948   0.9978
## Pos Pred Value         0.9778   0.9449   0.9257   0.9729   0.9899
## Neg Pred Value         0.9943   0.9877   0.9925   0.9903   0.9898
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2804   0.1835   0.1682   0.1557   0.1754
## Detection Prevalence   0.2868   0.1942   0.1817   0.1601   0.1772
## Balanced Accuracy      0.9884   0.9677   0.9743   0.9725   0.9760
```


## Prediction

Our prediction is made with the random forest model, that had the best results in our tests.


```r
predict(fit_rf, test, type = "class")
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
