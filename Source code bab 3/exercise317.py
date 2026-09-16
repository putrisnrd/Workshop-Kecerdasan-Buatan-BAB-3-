# Example 3.24 Ensemble1.py (Modified with KNeighborsClassifier)
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 1. Load Iris Dataset
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

# 2. Inisialisasi Classifier Individual
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
clf4 = SVC()
clf5 = KNeighborsClassifier(n_neighbors=5)  # Penambahan KNN Classifier

# 3. Inisialisasi Voting Classifier dengan 5 Classifier
eclf = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('gnb', clf3),
        ('svc', clf4),
        ('knn', clf5),
    ],
    voting='hard',
)

# 4. Evaluasi Cross-Validation untuk Setiap Model dan Ensemble
classifiers = [clf1, clf2, clf3, clf4, clf5, eclf]
labels = [
    'LogisticRegression',
    'Random Forest',
    'naive Bayes',
    'SVM',
    'KNN',
    'Ensemble',
]

for clf, label in zip(classifiers, labels):
  scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
  print(
      "Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label)
  )