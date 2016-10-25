from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
from sklearn import metrics
import numpy as np

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)

    print "Accuracy on training set : "
    print clf.score(X_train, y_train)
    print "Accuracy on testing set : "
    print clf.score(X_test, y_test)

    y_pred = clf.predict(X_test)

    print "Classification Report : "
    print metrics.classification_report(y_test, y_pred)
    print "Confusion Matrix : "
    print metrics.confusion_matrix(y_test, y_pred)

def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print scores
    print ("Mean score : {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))