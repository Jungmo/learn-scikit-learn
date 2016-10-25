from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import function as fnc

faces = fetch_olivetti_faces()


print faces.keys()

print faces.images.shape
print faces.data.shape
print faces.target.shape

# Check data is normalized

print np.max(faces.data)
print np.max(faces.data)
print np.mean(faces.data)

svc1 = SVC(kernel='linear') #linear, polynomial, rbf, sigmoid
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=0)

fnc.evaluate_cross_validation(svc1, X_train, y_train, 5)

fnc.train_and_evaluate(svc1, X_train, X_test, y_train, y_test)

