Activity Monitoring
================
Rajat Sanadhya

# Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>
(see the section on the Weight Lifting Exercise Dataset).

# Data

The **training data** for this project are available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>
<br/>The **test data** are available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>
<br/>The data for this project come from this source:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.
If you use the document you create for this class for any purpose please
cite them as they have been very generous in allowing their data to be
used for this kind of assignment.

# Goal

The goal of your project is to predict the manner in which they did the
exercise. This is the “classe” variable in the training set. You may use
any of the other variables to predict with. You should create a report
describing how you built your model, how you used cross validation, what
you think the expected out of sample error is, and why you made the
choices you did. You will also use your prediction model to predict 20
different test cases.

# Reproducibility

This markdown contains all the required necessary steps to make sure
this code is reproducible. The code doesn’t require much external
resources and relies mostly on pre-installed R packages. But you will
need to install the packages listed below. These packages are only for
the machine learning parts of the script.

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(rpart)
library(rpart.plot)
```

# Getting Data

Before strarting this, just place this file wherever you seems fit, this
script will download and save the data on its own, **Just make sure you
have a folde named “Data” in the directory you are running this scrip.**
To know your current working directory can be found by calling the
function `getwd()` in your R console.You can navigate to the desired
directory, where the `Data` folder exists, using `setwd(dirPath)`
function in R console.Here, `dirPath` is the path to the directory.

``` r
if (!file.exists(file.path(getwd(),"Data/training.csv"))) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = file.path(getwd(),"Data/training.csv"))
}
if (!file.exists(file.path(getwd(),"Data/testing.csv"))) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = file.path(getwd(),"Data/testing.csv"))
}
```

# Loading Data

Reading the downloaded data

``` r
train <- read.csv("Data/training.csv")
test <- read.csv("Data/testing.csv")
cat("\nTrain Data Dimensions : ", dim(train))
```

    ## 
    ## Train Data Dimensions :  19622 160

``` r
cat("\nTest Data Dimensions : ", dim(test))
```

    ## 
    ## Test Data Dimensions :  20 160

The **training data** contains `19622` observations and `160` variables.
The **test data** contains `20` observations and `160` variables.

# Data Cleaning

  - Firstly we will get rid of all the observations that have `NA`
    values

<!-- end list -->

``` r
cols <- (colSums(is.na(train)) == 0)
train <- train[, cols]
test <- test[, cols]
rm(cols)
```

  - Now we will remove variables that do not have accelerometer
    measurememnts.

<!-- end list -->

``` r
train <- train[, !grepl("^X|timestamp|user_name", names(train))]
test <- test[, !grepl("^X|timestamp|user_name", names(test))]
```

  - Now we will remove all those variables that have almost do not have
    a variance. This indicates that they are almost constant throughout
    the dataset.

<!-- end list -->

``` r
cols_nzv <-nearZeroVar(train, saveMetrics = TRUE)$nzv
train <- train[, !cols_nzv]
test <- test[, !cols_nzv]
rm(cols_nzv)
```

<br/>Now that we have cleaned our data, let’s check the dimensions of
our train and test datasets.

``` r
cat("Train Data Dimensions : ", dim(train))
```

    ## Train Data Dimensions :  19622 54

``` r
cat("\nTest Data Dimensions : ", dim(test))
```

    ## 
    ## Test Data Dimensions :  20 54

# Splitting Data

Now we will break the `train` data into **training set** and
**validation set**. Here I set the seed to `1407`, so you also get the
same random values as me.

``` r
set.seed(1407) # For reproducibile purpose
index_train <- createDataPartition(train$classe, p = 0.80, list = FALSE)
validation <- train[-index_train, ]
train <- train[index_train, ]
rm(index_train)
```

# Data Modelling

Now we will apply various models over our data to see which works best
for us.

  - ## Decision Trees
    
    We generate a predictive model using the ***Decision Tree***
    algorithm. we will also preview the decision tree created.

<!-- end list -->

``` r
modelTree <- rpart(classe ~ ., data = train, method = "class")
prp(modelTree)
```

![](Project_files/figure-gfm/Creating%20Decision%20Tree-1.png)<!-- -->
Now lets see how this performs on our validation data.

``` r
confusionMatrix(factor(validation$classe), predict(modelTree, validation, type = "class"))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   A   B   C   D   E
    ##          A 965  75   1  65  10
    ##          B  50 591  37  33  48
    ##          C   2  47 569  57   9
    ##          D  12  52  23 513  43
    ##          E   4  36   3  51 627
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.8323          
    ##                  95% CI : (0.8202, 0.8438)
    ##     No Information Rate : 0.2633          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.7885          
    ##                                           
    ##  Mcnemar's Test P-Value : 5.139e-11       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9342   0.7378   0.8989   0.7135   0.8507
    ## Specificity            0.9478   0.9462   0.9650   0.9594   0.9705
    ## Pos Pred Value         0.8647   0.7787   0.8319   0.7978   0.8696
    ## Neg Pred Value         0.9758   0.9336   0.9802   0.9372   0.9656
    ## Prevalence             0.2633   0.2042   0.1614   0.1833   0.1879
    ## Detection Rate         0.2460   0.1507   0.1450   0.1308   0.1598
    ## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      0.9410   0.8420   0.9320   0.8365   0.9106

The Decision Tree model gives us an **Accuracy** of about `83.23%`.

  - ## Random Forest
    
    We generate a predictive model using the ***Random Forest***
    algorithm. Here we will use **4-fold cross validation** on the
    training data.

<!-- end list -->

``` r
model_rf <- train(classe ~ ., data = train, method = "rf", trControl = trainControl(method = "cv", 4), ntree = 200)
```

Now lets see how this performs on our validation data.

``` r
confusionMatrix(factor(validation$classe), predict(model_rf, validation))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1116    0    0    0    0
    ##          B    1  758    0    0    0
    ##          C    0    2  682    0    0
    ##          D    0    0    2  641    0
    ##          E    0    0    0    2  719
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9982          
    ##                  95% CI : (0.9963, 0.9993)
    ##     No Information Rate : 0.2847          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9977          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9991   0.9974   0.9971   0.9969   1.0000
    ## Specificity            1.0000   0.9997   0.9994   0.9994   0.9994
    ## Pos Pred Value         1.0000   0.9987   0.9971   0.9969   0.9972
    ## Neg Pred Value         0.9996   0.9994   0.9994   0.9994   1.0000
    ## Prevalence             0.2847   0.1937   0.1744   0.1639   0.1833
    ## Detection Rate         0.2845   0.1932   0.1738   0.1634   0.1833
    ## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      0.9996   0.9985   0.9982   0.9981   0.9997

The Random Forest model gives us an **Accuracy** of about `99.77%`.

<br/>Hence, we ca conclude that **Random Forest Model** gives us the
better **Accuracy** among the two models.

# Predicting Test Data

Now we will apply the **Random Forest Model** to the `test` dataset.

``` r
predict(model_rf, test[, -length(names(test))])
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

# End-Note

In this document we create a model using the given data, I know this may
not be the best model ever to be created for this problem, this is just
small effort from my side. This problem can be further explored and
various other models with different parameters can also be tried to
improve the predictions.
