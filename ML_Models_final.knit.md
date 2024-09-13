---
title: "Applying ML Models"
output:
  html_document:
    df_print: paged
---

___________________________________________________________________________________   ___________________________________________________________________________________   


```r
# Load necessary libraries
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 4.3.3
```

```
## randomForest 4.7-1.1
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(class)
```

```
## Warning: package 'class' was built under R version 4.3.3
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 4.3.3
```

```
## Loading required package: ggplot2
```

```
## 
## Attaching package: 'ggplot2'
```

```
## The following object is masked from 'package:randomForest':
## 
##     margin
```

```
## Loading required package: lattice
```

```r
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 4.3.3
```

```
## Loading required package: rpart
```

```r
library(corrplot)
```

```
## Warning: package 'corrplot' was built under R version 4.3.3
```

```
## corrplot 0.92 loaded
```

```r
library(nnet)
library(knitr)

# Read the data
df <- read.csv("/Users/Rajnandini/Desktop/ds cp/sycsd grp 5/zomatoIndiaCleaned.csv", encoding = "latin1")

# Shuffle the data
set.seed(123) # Setting seed for reproducibility
df <- df[sample(nrow(df)),]
```



```r
## Subset the dataframe to include only the relevant columns
cor_matrix <- df[, c("Average.Cost.for.two", "Has.Table.booking", "Has.Online.delivery", "Is.delivering.now", "Votes", "Rating.text")]

# Convert factors to numeric
factor_cols <- sapply(cor_matrix, is.factor)
cor_matrix[factor_cols] <- lapply(cor_matrix[factor_cols], as.numeric)

# Handle any remaining non-numeric columns
cor_matrix <- cor_matrix[sapply(cor_matrix, is.numeric)]

# Calculate correlation matrix
cor_matrix <- cor(cor_matrix, use = "pairwise.complete.obs")

# Print the correlation matrix
print(cor_matrix)
```

```
##                      Average.Cost.for.two Has.Table.booking Has.Online.delivery
## Average.Cost.for.two          1.000000000        0.59595940         -0.02516928
## Has.Table.booking             0.595959403        1.00000000          0.03886871
## Has.Online.delivery          -0.025169276        0.03886871          1.00000000
## Is.delivering.now            -0.001363984       -0.02568486          0.09571299
## Votes                         0.250567740        0.16418255          0.05481102
##                      Is.delivering.now        Votes
## Average.Cost.for.two      -0.001363984  0.250567740
## Has.Table.booking         -0.025684857  0.164182552
## Has.Online.delivery        0.095712991  0.054811022
## Is.delivering.now          1.000000000 -0.004225401
## Votes                     -0.004225401  1.000000000
```
___________________________________________________________________________________   


```r
# Plot correlation heatmap
corrplot(cor_matrix, method = "circle")
```

<img src="ML_Models_final_files/figure-html/unnamed-chunk-3-1.png" width="672" />
___________________________________________________________________________________   



```r
# Split the data into training and test sets
set.seed(123) # Setting seed for reproducibility
train_index <- createDataPartition(df$Rating.text, p = 0.6, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]
train_data$Rating.text <- as.factor(train_data$Rating.text)
```

   
   
___________________________________________________________________________________   
___________________________________________________________________________________   
   
### Model 1: AdaBoost with Decision Trees



```r
set.seed(123) # Setting seed for reproducibility
ada_model <- randomForest(Rating.text ~ ., data = train_data, ntree = 60)
ada_pred <- predict(ada_model, test_data)
#accuracy
ada_accuracy <- sum(ada_pred == test_data$Rating.text) / length(test_data$Rating.text)
cat("Accuracy of Adaboost with Decision Tree is: ", ada_accuracy)
```

```
## Accuracy of Adaboost with Decision Tree is:  0.9992317
```


___________________________________________________________________________________   
   
#### AB: confusion matrix   


```r
# Create confusion matrix
conf_matrix_AB <- table(ada_pred, test_data$Rating.text)

# Print confusion matrix
kable(conf_matrix_AB, caption = "AB Confusion Matrix", format = "html")
```

<table>
<caption>AB Confusion Matrix</caption>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Average </th>
   <th style="text-align:right;"> Excellent </th>
   <th style="text-align:right;"> Good </th>
   <th style="text-align:right;"> Poor </th>
   <th style="text-align:right;"> Very Good </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Average </td>
   <td style="text-align:right;"> 1471 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Excellent </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 44 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Good </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 738 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Poor </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 72 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Very Good </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 276 </td>
  </tr>
</tbody>
</table>

___________________________________________________________________________________   

#### AB: model performance


```r
# Function to calculate precision
calculate_precision <- function(conf_matrix_AB) {
  true_positive <- diag(conf_matrix_AB)
  false_positive <- colSums(conf_matrix_AB) - true_positive
  precision <- true_positive / (true_positive + false_positive)
  return(precision)
}

# Function to calculate recall
calculate_recall <- function(conf_matrix_AB) {
  true_positive <- diag(conf_matrix_AB)
  false_negative <- rowSums(conf_matrix_AB) - true_positive
  recall <- true_positive / (true_positive + false_negative)
  return(recall)
}

# Function to calculate F1 score
calculate_f1_score <- function(precision, recall) {
  f1_score <- 2 * precision * recall / (precision + recall)
  return(f1_score)
}

# Calculate precision and recall
precision <- calculate_precision(conf_matrix_AB)
recall <- calculate_recall(conf_matrix_AB)

# Calculate F1 score for each class
f1_score <- calculate_f1_score(precision, recall)

# Output F1 scores
cat("F1 Score for each class:\n")
```

```
## F1 Score for each class:
```

```r
cat("Rating Class\tF1 Score\n")
```

```
## Rating Class	F1 Score
```

```r
for (i in 1:length(precision)) {
  cat(names(precision)[i], "\t\t", f1_score[i], "\n")
}
```

```
## Average 		 1 
## Excellent 		 0.9777778 
## Good 		 1 
## Poor 		 1 
## Very Good 		 0.9963899
```

```r
# Average F1 score across all classes
avg_f1_score <- mean(f1_score, na.rm = TRUE)
cat("\nAverage F1 Score of Adaboost: ", avg_f1_score, "\n")
```

```
## 
## Average F1 Score of Adaboost:  0.9948335
```


___________________________________________________________________________________   
___________________________________________________________________________________   


### Model 2: Random Forest Classification


```r
set.seed(123) # Setting seed for reproducibility
rf_model <- randomForest(Rating.text ~ ., data = train_data, ntree = 200)
rf_pred <- predict(rf_model, test_data)
rf_accuracy <- sum(rf_pred == test_data$Rating.text) / length(test_data$Rating.text)

cat("\nAccuracy of Random Forest is: ", rf_accuracy)
```

```
## 
## Accuracy of Random Forest is:  0.9992317
```

___________________________________________________________________________________   

#### RF: confusion matrix


```r
#confusion matrix of random forest
conf_matrix <- table(predicted = rf_pred, actual = test_data$Rating.text)
kable(conf_matrix, caption = "RF Confusion Matrix", format = "html")
```

<table>
<caption>RF Confusion Matrix</caption>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Average </th>
   <th style="text-align:right;"> Excellent </th>
   <th style="text-align:right;"> Good </th>
   <th style="text-align:right;"> Poor </th>
   <th style="text-align:right;"> Very Good </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Average </td>
   <td style="text-align:right;"> 1471 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Excellent </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 44 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Good </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 738 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Poor </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 72 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Very Good </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 276 </td>
  </tr>
</tbody>
</table>

___________________________________________________________________________________   

#### RF: model performance


```r
# Function to calculate precision
calculate_precision <- function(conf_matrix) {
  true_positive <- diag(conf_matrix)
  false_positive <- colSums(conf_matrix) - true_positive
  precision <- true_positive / (true_positive + false_positive)
  return(precision)
}

# Function to calculate recall
calculate_recall <- function(conf_matrix) {
  true_positive <- diag(conf_matrix)
  false_negative <- rowSums(conf_matrix) - true_positive
  recall <- true_positive / (true_positive + false_negative)
  return(recall)
}

# Function to calculate F1 score
calculate_f1_score <- function(precision, recall) {
  f1_score <- 2 * precision * recall / (precision + recall)
  return(f1_score)
}

# Calculate precision and recall
precision <- calculate_precision(conf_matrix)
recall <- calculate_recall(conf_matrix)

# Calculate F1 score for each class
f1_score <- calculate_f1_score(precision, recall)

# Output F1 scores
cat("F1 Score for each class:\n")
```

```
## F1 Score for each class:
```

```r
cat("Rating Class\tF1 Score\n")
```

```
## Rating Class	F1 Score
```

```r
for (i in 1:length(precision)) {
  cat(names(precision)[i], "\t\t", f1_score[i], "\n")
}
```

```
## Average 		 1 
## Excellent 		 0.9777778 
## Good 		 1 
## Poor 		 1 
## Very Good 		 0.9963899
```

```r
# Average F1 score across all classes
avg_f1_score <- mean(f1_score, na.rm = TRUE)
cat("\nAverage F1 Score of RF:", avg_f1_score, "\n")
```

```
## 
## Average F1 Score of RF: 0.9948335
```

___________________________________________________________________________________   
___________________________________________________________________________________   


### Model 3: KNN


```r
features <- df[, c("Average.Cost.for.two", "Has.Table.booking", "Has.Online.delivery", "Is.delivering.now", "Votes")]
target <- df$Rating.text

# Convert factors to numeric if needed
factor_cols <- sapply(features, is.factor)
features[factor_cols] <- lapply(features[factor_cols], as.numeric)

# Scale the features
scaled_features <- scale(features)

# Split the data into training and test sets
set.seed(123) # Setting seed for reproducibility
train_index <- createDataPartition(target, p = 0.6, list = FALSE)
train_features <- scaled_features[train_index, ]
test_features <- scaled_features[-train_index, ]
train_target <- target[train_index]
test_target <- target[-train_index]

# Model Training and Prediction
k <- 10  # Specify the value of k (try different values)
knn_model <- knn(train = train_features, test = test_features, cl = train_target, k = k)

# Evaluate the performance of the model
accuracy <- sum(knn_model == test_target) / length(test_target)
cat("\nAccuracy of KNN model:", accuracy)
```

```
## 
## Accuracy of KNN model: 0.6653861
```

___________________________________________________________________________________   

#### KNN: confusion matrix


```r
# Create confusion matrix
conf_matrix1 <- table(predicted = knn_model, actual = test_target)
kable(conf_matrix1, caption = "KNN Confusion Matrix", format = "html")
```

<table>
<caption>KNN Confusion Matrix</caption>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Average </th>
   <th style="text-align:right;"> Excellent </th>
   <th style="text-align:right;"> Good </th>
   <th style="text-align:right;"> Poor </th>
   <th style="text-align:right;"> Very Good </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Average </td>
   <td style="text-align:right;"> 1262 </td>
   <td style="text-align:right;"> 3 </td>
   <td style="text-align:right;"> 271 </td>
   <td style="text-align:right;"> 60 </td>
   <td style="text-align:right;"> 50 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Excellent </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 3 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Good </td>
   <td style="text-align:right;"> 198 </td>
   <td style="text-align:right;"> 28 </td>
   <td style="text-align:right;"> 399 </td>
   <td style="text-align:right;"> 10 </td>
   <td style="text-align:right;"> 154 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Poor </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Very Good </td>
   <td style="text-align:right;"> 11 </td>
   <td style="text-align:right;"> 14 </td>
   <td style="text-align:right;"> 65 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 70 </td>
  </tr>
</tbody>
</table>

___________________________________________________________________________________   

#### KNN: model performance


```r
# Function to calculate precision
calculate_precision <- function(confusion_matrix, label) {
  true_positive <- confusion_matrix[label, label]
  false_positive <- sum(confusion_matrix[label, ]) - true_positive
  precision <- true_positive / (true_positive + false_positive)
  return(precision)
}

# Function to calculate recall
calculate_recall <- function(confusion_matrix, label) {
  true_positive <- confusion_matrix[label, label]
  false_negative <- sum(confusion_matrix[, label]) - true_positive
  recall <- true_positive / (true_positive + false_negative)
  return(recall)
}

# Function to calculate F1 score
calculate_f1_score <- function(confusion_matrix, label) {
  precision <- calculate_precision(confusion_matrix, label)
  recall <- calculate_recall(confusion_matrix, label)
  f1_score <- 2 * precision * recall / (precision + recall)
  return(f1_score)
}

# Create confusion matrix
conf_matrix <- table(predicted = knn_model, actual = test_target)

# Calculate F1 score for each class
classes <- rownames(conf_matrix)
f1_scores <- sapply(classes, function(class) calculate_f1_score(conf_matrix, class))

# Output F1 scores
cat("F1 Scores:\n")
```

```
## F1 Scores:
```

```r
for (i in 1:length(classes)) {
  cat(classes[i], ": ", f1_scores[i], "\n")
}
```

```
## Average :  0.809753 
## Excellent :  0.03846154 
## Good :  0.5225933 
## Poor :  NaN 
## Very Good :  0.3196347
```

```r
avg_f1_score <- mean(f1_scores, na.rm = TRUE)
cat("\nAverage F1 Score of KNN:", avg_f1_score, "\n")
```

```
## 
## Average F1 Score of KNN: 0.4226106
```



