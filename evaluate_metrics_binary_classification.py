"""
If we talk about classification problems, the most common metrics used are:
- Accuracy
- Precision (P)
- Recall (R)
- F1 score (F1)
- Area under the ROC (Receiver Operating Characteristic) curve or simply
AUC (AUC)
- Log loss
- Precision at k (P@k)
- Average precision at k (AP@k)
- Mean average precision at k (MAP@k)

When it comes to regression, the most commonly used evaluation metrics are:
- Mean absolute error (MAE)
- Mean squared error (MSE)
- Root mean squared error (RMSE)
- Root mean squared logarithmic error (RMSLE)
- Mean percentage error (MPE)
- Mean absolute percentage error (MAPE)
- R2

"""


def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score
    """
    # initialize a simple counter for correct predictions
    correct_counter = 0
    # loop over all elements of y_true
    # and y_pred "together"
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            # if prediction is equal to truth, increase the counter
            correct_counter += 1
    # return accuracy
    # which is correct predictions over the number of samples
    return correct_counter / len(y_true)


from sklearn import metrics
l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]
print(metrics.accuracy_score(l1, l2))

###

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


l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]

print("true_positive==",true_positive(l1, l2))
print("false_positive==",false_positive(l1, l2))
print("false_negative==",false_negative(l1, l2))
print("true_negative==",true_negative(l1, l2))

def accuracy_v2(y_true, y_pred):
 """
 Function to calculate accuracy using tp/tn/fp/fn
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: accuracy score
 """
 tp = true_positive(y_true, y_pred)
 fp = false_positive(y_true, y_pred)
 fn = false_negative(y_true, y_pred)
 tn = true_negative(y_true, y_pred)
 accuracy_score = (tp + tn) / (tp + tn + fp + fn)
 return accuracy_score

l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]

print(accuracy(l1, l2))
print(accuracy_v2(l1, l2))
print(metrics.accuracy_score(l1, l2))


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

l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]

print("precision",precision(l1, l2))


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

l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]

print("recall",recall(l1, l2))

def f1(y_true, y_pred):
 """
 Function to calculate f1 score
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: f1 score
 """
 p = precision(y_true, y_pred)
 r = recall(y_true, y_pred)
 score = 2 * p * r / (p + r)
 return score

y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
y_pred = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0,1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

print("F1",f1(y_true, y_pred))
from sklearn import metrics

print("scikit f1 score",metrics.f1_score(y_true, y_pred))


def tpr(y_true, y_pred): #TPR or recall is also known as sensitivity.
 """
 TPR or recall is also known as sensitivity.
 Function to calculate tpr
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: tpr/recall
 """
 return recall(y_true, y_pred)

def fpr(y_true, y_pred): # And 1 - FPR is known as specificity or True Negative Rate or TNR.
 """
 And 1 - FPR is known as specificity or True Negative Rate or TNR.
 Function to calculate fpr
 :param y_true: list of true values
 :param y_pred: list of predicted values
 :return: fpr
 """
 fp = false_positive(y_true, y_pred)
 tn = true_negative(y_true, y_pred)
 return fp / (tn + fp)

tpr_list = []
fpr_list = []
# actual targets
y_true = [0, 0, 0, 0, 1, 0, 1,
 0, 0, 1, 0, 1, 0, 0, 1]
# predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
 0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
 0.85, 0.15, 0.99]
# handmade thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]
# loop over all thresholds
for thresh in thresholds:
 # calculate predictions for a given threshold
 temp_pred = [1 if x >= thresh else 0 for x in y_pred]
 # calculate tpr
 temp_tpr = tpr(y_true, temp_pred)
 # calculate fpr
 temp_fpr = fpr(y_true, temp_pred)
 # append tpr and fpr to lists
 tpr_list.append(temp_tpr)
 fpr_list.append(temp_fpr)


print("tpr===",tpr_list)
print("fpr====",fpr_list)


from sklearn import metrics
y_true = [0, 0, 0, 0, 1, 0, 1,0, 0]
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,0.85, 0.15, 0.99]
print("roc_auc_score ===",metrics.roc_auc_score(y_true, y_pred))

"""
AUC values range from 0 to 1.
- AUC = 1 implies you have a perfect model. Most of the time, it means that
you made some mistake with validation and should revisit data processing
and validation pipeline of yours. If you didn’t make any mistakes, then
congratulations, you have the best model one can have for the dataset you
built it on.
- AUC = 0 implies that your model is very bad (or very good!). Try inverting
the probabilities for the predictions, for example, if your probability for the
positive class is p, try substituting it with 1-p. This kind of AUC may also
mean that there is some problem with your validation or data processing.
- AUC = 0.5 implies that your predictions are random. So, for any binary
classification problem, if I predict all targets as 0.5, I will get an AUC of
0.5.
AUC values between 0 and 0.5 imply that your model is worse than random. Most
of the time, it’s because you inverted the classes. If you try to invert your
predictions, your AUC might become more than 0.5. AUC values closer to 1 are
considered good.
But what does AUC say about our model?
Suppose you get an AUC of 0.85 when you build a model to detect pneumothorax
from chest x-ray images. This means that if you select a random image from your
dataset with pneumothorax (positive sample) and another random image without
pneumothorax (negative sample), then the pneumothorax image will rank higher
than a non-pneumothorax image with a probability of 0.85.
"""

############################################ log loss

"""                 
log loss. In case of a binary classification problem, we define log loss as:

Log Loss = - 1.0 * ( target * log(prediction) + (1 - target) * log(1 - prediction) )
"""

import numpy as np
def log_loss(y_true, y_proba):
    """
     Function to calculate log loss
     :param y_true: list of true values
     :param y_proba: list of probabilities for 1
     :return: overall log loss
     """
    # define an epsilon value
    # this can also be an input
    # this value is used to clip probabilities
    epsilon = 1e-15
    # initialize empty list to store
    # individual losses
    loss = []
    # loop over all true and predicted probability values
    for yt, yp in zip(y_true, y_proba):
        # adjust probability
        # 0 gets converted to 1e-15
        # 1 gets converted to 1-1e-15
        # Why? Think about it!
        yp = np.clip(yp, epsilon, 1 - epsilon)
        # calculate loss for one sample
        temp_loss = - 1.0 * (
                yt * np.log(yp)
                + (1 - yt) * np.log(1 - yp)
        )
        # add to loss list
        loss.append(temp_loss)
    # return mean loss over all samples
    return np.mean(loss)

y_true = [0, 0, 0, 0, 1, 0, 1,0, 0, 1]
y_proba = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,0.85, 0.15, 0.99]
print("log_loss====",log_loss(y_true, y_proba))

from sklearn import metrics
print("scikit learn======",metrics.log_loss(y_true, y_proba))