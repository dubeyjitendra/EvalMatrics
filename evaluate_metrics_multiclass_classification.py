"""

Most of the metrics that we discussed until now can be converted to a multi-class
version. The idea is quite simple. Let’s take precision and recall. We can calculate
precision and recall for each class in a multi-class classification problem.

There are three different ways to calculate this which might get confusing from time
to time. Let’s assume we are interested in precision first. We know that precision
depends on true positives and false positives.
    - Macro averaged precision: calculate precision for all classes individually
    and then average them
    - Micro averaged precision: calculate class wise true positive and false
    positive and then use that to calculate overall precision
    - Weighted precision: same as macro but in this case, it is weighted average
    depending on the number of items in each class

"""

def true_positive(y_true, y_pred):
 """
 Function to calculate True Positives
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: number of true positives
 """
 # initialize
 tp = 0
 for yt, yp in zip(y_true, y_pred):
     if yt == 1 and yp == 1:
        tp += 1
 return tp

def true_negative(y_true, y_pred):
 """
 Function to calculate True Negatives
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: number of true negatives
 """
 # initialize
 tn = 0
 for yt, yp in zip(y_true, y_pred):
     if yt == 0 and yp == 0:
        tn += 1
 return tn

def false_positive(y_true, y_pred):
 """
 Function to calculate False Positives
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: number of false positives
 """
 # initialize
 fp = 0
 for yt, yp in zip(y_true, y_pred):
     if yt == 0 and yp == 1:
         fp += 1

 return fp

def false_negative(y_true, y_pred):
 """
 Function to calculate False Negatives
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: number of false negatives
 """
 # initialize
 fn = 0
 for yt, yp in zip(y_true, y_pred):
     if yt == 1 and yp == 0:
         fn += 1

 return fn


import numpy as np
def macro_precision(y_true, y_pred):
 """
 Function to calculate macro averaged precision
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: macro precision score
 """

 # find the number of classes by taking
 # length of unique values in true list
 num_classes = len(np.unique(y_true))

 # initialize precision to 0
 precision = 0
 # loop over all classes
 for class_ in range(num_classes):
     # all classes except current are considered negative
     temp_true = [1 if p == class_ else 0 for p in y_true]
     temp_pred = [1 if p == class_ else 0 for p in y_pred]

     # calculate true positive for current class
     tp = true_positive(temp_true, temp_pred)

     # calculate false positive for current class
     fp = false_positive(temp_true, temp_pred)

     # calculate precision for current class
     temp_precision = tp / (tp + fp)

     # keep adding precision for all classes
     precision += temp_precision

     # calculate and return average precision over all classes
 precision /= num_classes
 return precision


def micro_precision(y_true, y_pred):
 """
 Function to calculate micro averaged precision
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: micro precision score
 """

 # find the number of classes by taking
 # length of unique values in true list
 num_classes = len(np.unique(y_true))

 # initialize tp and fp to 0
 tp = 0
 fp = 0
 # loop over all classes
 for class_ in range(num_classes):
     # all classes except current are considered negative
     temp_true = [1 if p == class_ else 0 for p in y_true]
     temp_pred = [1 if p == class_ else 0 for p in y_pred]

     # calculate true positive for current class
     # and update overall tp
     tp += true_positive(temp_true, temp_pred)

     # calculate false positive for current class
     # and update overall tp
     fp += false_positive(temp_true, temp_pred)

     # calculate and return overall precision
 precision = tp / (tp + fp)
 return precision


from collections import Counter
import numpy as np
def weighted_precision(y_true, y_pred):
 """
 Function to calculate weighted averaged precision
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: weighted precision score
 """

 # find the number of classes by taking
 # length of unique values in true list
 num_classes = len(np.unique(y_true))

 # create class:sample count dictionary
 # it looks something like this:
 # {0: 20, 1:15, 2:21}
 class_counts = Counter(y_true)

 # initialize precision to 0
 precision = 0

 # loop over all classes
 for class_ in range(num_classes):
     # all classes except current are considered negative
     temp_true = [1 if p == class_ else 0 for p in y_true]
     temp_pred = [1 if p == class_ else 0 for p in y_pred]

     # calculate tp and fp for class
     tp = true_positive(temp_true, temp_pred)
     fp = false_positive(temp_true, temp_pred)

     # calculate precision of class
     temp_precision = tp / (tp + fp)

     # multiply precision with count of samples in class
     weighted_precision = class_counts[class_] * temp_precision

     # add to overall precision
     precision += weighted_precision

 # calculate overall precision by dividing by
 # total number of samples
 overall_precision = precision / len(y_true)
 return overall_precision


from sklearn import metrics
y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

print("macro_precision",macro_precision(y_true, y_pred))
print("metrics.precision_score",metrics.precision_score(y_true, y_pred, average="macro"))
print("micro_precision", micro_precision(y_true, y_pred))
print("metrics.precision_score",metrics.precision_score(y_true, y_pred, average="micro"))
print("weighted_precision", weighted_precision(y_true, y_pred))
print("metrics.precision_score",metrics.precision_score(y_true, y_pred, average="weighted"))



from collections import Counter
import numpy as np

def precision(y_true, y_pred):
 """
 Function to calculate precision
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: precision score
 """
 tp = true_positive(y_true, y_pred)
 fp = false_positive(y_true, y_pred)
 precision = tp / (tp + fp)
 return precision



def recall(y_true, y_pred):
 """
 Function to calculate recall
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: recall score
 """
 tp = true_positive(y_true, y_pred)
 fn = false_negative(y_true, y_pred)
 recall = tp / (tp + fn)
 return recall


def weighted_f1(y_true, y_pred):
 """
 Function to calculate weighted f1 score
 :param y_true: list of true values
 :param y_proba: list of predicted values
 :return: weighted f1 score
 """

 # find the number of classes by taking
 # length of unique values in true list
 num_classes = len(np.unique(y_true))

 # create class:sample count dictionary
 # it looks something like this:
 # {0: 20, 1:15, 2:21}
 class_counts = Counter(y_true)

 # initialize f1 to 0
 f1 = 0
 # loop over all classes
 for class_ in range(num_classes):
     # all classes except current are considered negative
     temp_true = [1 if p == class_ else 0 for p in y_true]
     temp_pred = [1 if p == class_ else 0 for p in y_pred]

     # calculate precision and recall for class
     p = precision(temp_true, temp_pred)
     r = recall(temp_true, temp_pred)

     # calculate f1 of class
     if p + r != 0:
         temp_f1 = 2 * p * r / (p + r)
     else:
         temp_f1 = 0

     # multiply f1 with count of samples in class
     weighted_f1 = class_counts[class_] * temp_f1

     # add to f1 precision
     f1 += weighted_f1
     # calculate overall F1 by dividing by
     # total number of samples
 overall_f1 = f1 / len(y_true)
 return overall_f1

y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]
print("weighted_f1",weighted_f1(y_true, y_pred))

from sklearn import metrics
print("metrics.f1_score",metrics.f1_score(y_true, y_pred, average="weighted"))



############################################################

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
# some targets
y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
#some predictions
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]
# get confusion matrix from sklearn
cm = metrics.confusion_matrix(y_true, y_pred)
# plot using matplotlib and seaborn
plt.figure(figsize=(10, 10))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0,
as_cmap=True)
sns.set(font_scale=2.5)
sns.heatmap(cm, annot=True, cmap=cmap, cbar=False)
plt.ylabel('Actual Labels', fontsize=20)
plt.xlabel('Predicted Labels', fontsize=20)
# plt.show()

#######################


###########################  multi-label ###############################################
"""In multi-label classification, each sample can have one or more
classes associated with it. One simple example of this type of problem would be a
task in which you are asked to predict different objects in a given image.

The metrics for this type of classification problem are a bit different. Some suitable
and most common metrics are:

- Precision at k (P@k)
- Average precision at k (AP@k)
- Mean average precision at k (MAP@k)
- Log loss

Let’s start with precision at k or P@k. One must not confuse this precision with
the precision discussed earlier. If you have a list of original classes for a given
sample and list of predicted classes for the same, precision is defined as the number
of hits in the predicted list considering only top-k predictions, divided by k.

"""
def pk(y_true, y_pred, k):
 """
 This function calculates precision at k
 for a single sample
 :param y_true: list of values, actual classes
 :param y_pred: list of values, predicted classes
 :param k: the value for k
 :return: precision at a given value k
 """
 # if k is 0, return 0. we should never have this
 # as k is always >= 1
 if k == 0:
    return 0
 # we are interested only in top-k predictions
 y_pred = y_pred[:k]
 # convert predictions to set
 pred_set = set(y_pred)
 # convert actual values to set
 true_set = set(y_true)
 # find common values
 common_values = pred_set.intersection(true_set)
 # return length of common values over k
 return len(common_values) / len(y_pred[:k])

def apk(y_true, y_pred, k):
 """
 This function calculates average precision at k
 for a single sample
 :param y_true: list of values, actual classes
 :param y_pred: list of values, predicted classes
 :return: average precision at a given value k
 """
 # initialize p@k list of values
 pk_values = []
 # loop over all k. from 1 to k + 1
 for i in range(1, k + 1):
     # calculate p@i and append to list
     pk_values.append(pk(y_true, y_pred, i))
 # if we have no values in the list, return 0
 if len(pk_values) == 0:
    return 0
 # else, we return the sum of list over length of list
 return sum(pk_values) / len(pk_values)

y_true = [
 [1, 2, 3],
 [0, 2],
 [1],
 [2, 3],
 [1, 0],
 []
 ]

y_pred = [
[0, 1, 2],
[1],
[0, 2, 3],
[2, 3, 4, 0],
[0, 1, 2],
[0]
]

print("Multi label classification=============================================================")
for i in range(len(y_true)):
    for j in range(1, 4):
        print(f"""y_true={y_true[i]},_pred={y_pred[i]},AP@{j}={apk(y_true[i], y_pred[i], k=j)}""")
        print("==================================================================================")


print("mean average precision===================================")


def mapk(y_true, y_pred, k):
 """
 This function calculates mean avg precision at k
 for a single sample
 :param y_true: list of values, actual classes
 :param y_pred: list of values, predicted classes
 :return: mean avg precision at a given value k
 Now, we can calculate MAP@k for k=1, 2, 3 and 4 for the same list of lists.
 """
 # initialize empty list for apk values
 apk_values = []
 # loop over all samples
 for i in range(len(y_true)):
     # store apk values for every sample
     apk_values.append(apk(y_true[i], y_pred[i], k=k))
     # return mean of apk values list
 return sum(apk_values) / len(apk_values)

print("k=1",mapk(y_true, y_pred, k=1))
mapk("k=2",y_true, y_pred, k=2)
print("k=3",mapk(y_true, y_pred, k=3))
mapk("k=4",y_true, y_pred, k=4)

"""P@k, AP@k and MAP@k all range from 0 to 1 with 1 being the best."""


##########################
import numpy as np

def apk(actual, predicted, k=10):
 """
 Computes the average precision at k.
 This function computes the AP at k between two lists of
 items.
 Parameters
 ----------
 actual : list
 A list of elements to be predicted (order doesn't matter)
 predicted : list
 A list of predicted elements (order does matter)
 k : int, optional
 The maximum number of predicted elements
 Returns
 -------
 score : double
 The average precision at k over the input lists
 """
 if len(predicted)>k:
    predicted = predicted[:k]
 score = 0.0
 num_hits = 0.0
 for i, p in enumerate(predicted):
     if p in actual and p not in predicted[:i]:
         num_hits += 1.0
     score += num_hits / (i + 1.0)

 if not actual:
     return 0.0

 return score / min(len(actual), k)

"""
The most common metric in regression is error. Error is simple and very easy to
understand.
                Error = True Value – Predicted Value

Absolute error is just absolute of the above.

Absolute Error = Abs ( True Value – Predicted Value )

Then we have mean absolute error (MAE). It’s just mean of all absolute errors.
"""
import numpy as np
def mean_absolute_error(y_true, y_pred):

    """
    This function calculates mae
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean absolute error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
     # calculate absolute error
     # and add to error
     error += np.abs(yt - yp)
     # return mean error
    return error / len(y_true)

"""

Similarly, we have squared error and mean squared error (MSE).

                Squared Error = ( True Value – Predicted Value )2

And mean squared error (MSE) can be implemented as follows.
"""

def mean_squared_error(y_true, y_pred):
    """
    This function calculates mse
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean squared error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate squared error
        # and add to error
        error += (yt - yp) ** 2
    # return mean error
    return error / len(y_true)



"""

MSE and RMSE (root mean squared error) are the most popular metrics used in
evaluating regression models.

                    RMSE = SQRT ( MSE )
    
Another type of error in same class is squared logarithmic error. Some people
call it SLE, and when we take mean of this error across all samples, it is known as
MSLE (mean squared logarithmic error) and implemented as follows.

"""

import numpy as np
def mean_squared_log_error(y_true, y_pred):
    """
    This function calculates msle
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean squared logarithmic error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate squared log error
        # and add to error
        error += (np.log(1 + yt) - np.log(1 + yp)) ** 2
    # return mean error
    return error / len(y_true)


"""
Root mean squared logarithmic error is just a square root of this. It is also known
as RMSLE.
Then we have the percentage error:


        Percentage Error = ( ( True Value – Predicted Value ) / True Value ) * 100


Same can be converted to mean percentage error for all samples.
"""

def mean_percentage_error(y_true, y_pred):
    """
    This function calculates mpe
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean percentage error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate percentage error
        # and add to error
        error += (yt - yp) / yt
    # return mean percentage error
    return error / len(y_true)

#========================================

"""
And an absolute version of the same (and more common version) is known as mean
absolute percentage error or MAPE.

"""
import numpy as np
def mean_abs_percentage_error(y_true, y_pred):
    """
    This function calculates MAPE
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean absolute percentage error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate percentage error
        # and add to error
        error += np.abs(yt - yp) / yt
    # return mean percentage error
    return error / len(y_true)


######################################
"""
The best thing about regression is that there are only a few most popular metrics
that can be applied to almost every regression problem. And it is much easier to
understand when we compare it to classification metrics.

Let’s talk about another regression metric known as R2 (R-squared), also known
as the coefficient of determination.

In simple words, R-squared says how good your model fits the data. R-squared
closer to 1.0 says that the model fits the data quite well, whereas closer 0 means
that model isn’t that good. R-squared can also be negative when the model just
makes absurd predictions.

The formula for R-squared is shown in figure 10, but as always a python
implementation makes things more clear.


<link> https://i.ibb.co/jbsc6s7/r2.png

Figure 10: Formula for R-squared

"""

import numpy as np
def r2(y_true, y_pred):
    """
    This function calculates r-squared score
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: r2 score
    """

    # calculate the mean value of true values
    mean_true_value = np.mean(y_true)

    # initialize numerator with 0
    numerator = 0
    # initialize denominator with 0
    denominator = 0

    # loop over all true and predicted values
    for yt, yp in zip(y_true, y_pred):
        # update numerator
        numerator += (yt - yp) ** 2
        # update denominator
        denominator += (yt - mean_true_value) ** 2
    # calculate the ratio
    ratio = numerator / denominator
    # return 1 - ratio
    return 1-ratio




#############################################


""""

There are many more evaluation metrics, and this list is never-ending. I can write a
book which is only about different evaluation metrics. Maybe I will. For now, these
evaluations metrics will fit almost every problem you want to attempt. Please note
that I have implemented these metrics in the most straightforward manner, and that
means they are not efficient enough. You can make most of them in a very efficient
way by properly using numpy. For example, take a look at the implementation of
mean absolute error without any loops.

"""

import numpy as np
def mae_np(y_true, y_pred):
 return np.mean(np.abs(y_true - y_pred))



"""
Then, there are some advanced metrics.

One of them which is quite widely used is quadratic weighted kappa, also known
as QWK. It is also known as Cohen’s kappa. QWK measures the “agreement”
between two “ratings”.
 
 
 The ratings can be any real numbers in 0 to N. And
predictions are also in the same range. An agreement can be defined as how close
these ratings are to each other. So, it’s suitable for a classification problem with N
different categories/classes. If the agreement is high, the score is closer towards 1.0.
In the case of low agreement, the score is close to 0. Cohen’s kappa has a good
implementation in scikit-learn, and detailed discussion of this metric is beyond the
scope of this book.
"""

from sklearn import metrics
y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
y_pred = [2, 1, 3, 1, 2, 3, 3, 1, 2]
print(metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic"))
print(metrics.accuracy_score(y_true, y_pred))


######################## Matthew’s Correlation Coefficient (MCC)

"""
MCC ranges
from -1 to 1. 1 is perfect prediction, -1 is imperfect prediction, and 0 is random
prediction. The formula for MCC is quite simple.

We see that MCC takes into consideration TP, FP, TN and FN and thus can be used
for problems where classes are skewed. You can quickly implement it in python by
using what we have already implemented.
"""

def mcc(y_true, y_pred):
    """
    This function calculates Matthew's Correlation Coefficient
    for binary classification.
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: mcc score
    """
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    numerator = (tp * tn) - (fp * fn)
    denominator = (
            (tp + fp) *
            (fn + tn) *
            (fp + tn) *
            (tp + fn)
    )
    denominator = denominator ** 0.5
    return numerator / denominator



###  VERY IMPORTANT MACHINE LEARNING concept read below
"""

These are the metrics that can help you get started and will apply to almost every
machine learning problem.

One thing to keep in mind is that to evaluate un-supervised methods, for example,
some kind of clustering, it’s better to create or manually label the test set and keep
it separate from everything that is going on in your modelling part. When you are
done with clustering, you can evaluate the performance on the test set simply by
using any of the supervised learning metrics.

Once we understand what metric to use for a given problem, we can start looking
more deeply into our models for improvements.

"""