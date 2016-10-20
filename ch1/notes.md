### Save plots to a pdf file
~~~~{.python}
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('output.pdf')
plt.savefig(pp, format='pdf')
pp.savefig()
pp.close()
~~~~

### Cross Validation
~~~~{.python}
clf = Pipeline([('scaler', preprocessing.StandardScaler()), ('linear_model', SGDClassifier())])
# KFold( **# of data**, k, ...)
cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)
scores = cross_val_score(clf, X,y, cv=cv)
print scores
def mean_score(scores):
    return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))
print mean_score(scores)
~~~~

### Draw multiple graphs
~~~~{.python}
for i in [0,1,2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class'+str(i)+' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    pylab.sca(axes[i])
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.prism)
    ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
    plt.plot(xs, ys, hold=True)
~~~~

### model evaluation
~~~~{.python}
y_train_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)
print metrics.accuracy_score(y_train, y_train_pred)
print metrics.accuracy_score(y_test, y_pred)
print metrics.classification_report(y_test, y_pred, target_names=iris.target_names)
print metrics.confusion_matrix(y_test, y_pred)
~~~~
