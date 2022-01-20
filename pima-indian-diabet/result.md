#Result:

```
head of data : 
    pregnant  glucose  bp  skin  insulin   bmi  pedigree  age  label
0         6      148  72    35        0  33.6     0.627   50      1
1         1       85  66    29        0  26.6     0.351   31      0
2         8      183  64     0        0  23.3     0.672   32      1
3         1       89  66    23       94  28.1     0.167   21      0
4         0      137  40    35      168  43.1     2.288   33      1
Confusion Matrix: 
 [[117  29]
 [ 41  44]]
Classification Report: 
               precision    recall  f1-score   support

           0       0.74      0.80      0.77       146
           1       0.60      0.52      0.56        85

    accuracy                           0.70       231
   macro avg       0.67      0.66      0.66       231
weighted avg       0.69      0.70      0.69       231

Accuracy: 0.696969696969697


```