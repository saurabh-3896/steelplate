##########################              Importing libraries                 #######################################
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class_names=np.arange(6)

# Below function is used to plot confusion matrix

def plot_confusion_matrix(conmat, classes):
    cmap = plt.cm.Blues
    title = 'Confusion matrix'

    print(conmat)

    plt.imshow(conmat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = conmat.max() / 2.
    for i, j in itertools.product(range(conmat.shape[0]), range(conmat.shape[1])):
        plt.text(j, i, format(conmat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conmat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



path='features.xlsx'                         #give path where extracted features are saved

abc=pd.read_excel(path,header=None)

X=np.array((abc.as_matrix())[1:,1:])
Y=X[:,5]                                   # Stores labels in Y variable
X=X[:,0:5]                                  # Stores features in X variable
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)      # test train split is done 80% training and 20% testing
y_train = y_train.astype('int')
y_test = y_test.astype('int')                                           #labels must be of integer type and not float. So are converted to int


K_value = 3
neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')  #KNN model initialisation
neigh.fit(X_train, y_train)                                                             #train model
testt=neigh.predict(X_test)                                                             #predict labels
scoree=neigh.score(X_test, y_test)                                                      #get accuracy
print("KNN testing accuracy=",scoree*100)                                               #print accuracy in %

################################## same pattern of training,testing  is followed below  ####################################


AB = AdaBoostClassifier()
AB.fit(X_train, y_train)
sscore = AB.score(X_test, y_test)
print("Adaboost accuracy=",sscore*100)

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
sscore = RF.score(X_test, y_test)
print("Random forest accuracy=",sscore*100)

SVM = SVC(kernel='rbf')
SVM.fit(X_train, y_train)
sscore = SVM.score(X_test, y_test)
print("SVM accuracy=",sscore*100)

DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
sscore = DT.score(X_test, y_test)
print("Decision Tree Classifier accuracy=",sscore*100)

GB = GradientBoostingClassifier(n_estimators=80)
GB.fit(X_train, y_train)
predicted=GB.predict(X_test)
sscore = GB.score(X_test, y_test)
print("gradient boosting accuracy=",sscore*100)




cnf_matrix=confusion_matrix(y_test,predicted)                               #plot confusion matrix
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_names)
plt.savefig('GBconfusion_matrix')
plt.show()                                                      #Save and show confusion matrix


##################################       models are appended         ####################################

models=[]
models.append(('KNN',KNeighborsClassifier()))
models.append(('GB',GradientBoostingClassifier()))
models.append(('RF',RandomForestClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('SVM',SVC()))
models.append(('AB',AdaBoostClassifier()))


################################## function is defined to get cross validation results  ####################################

def kfoldcv(num_folds,seed):

    num_instances= len(X_train)
    scoring='accuracy'

    result = []
    namee = []
    scoringg = 'accuracy'
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    for name, model in models:

        cv_result = cross_validation.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoringg)
        result.append(cv_result)
        namee.append(name)
        message = "%s: %f " % (name, cv_result.mean()*100)
        print(message)

print("folds=5,random_state=9")
kfoldcv(5,9)                                # prints cross validation results with folds=5 and random_state=9

print("folds=10,random_state=70")
kfoldcv(10,70)                              # prints cross validation results with folds=10 and random_state=70

print("folds=15,random_state=35")
kfoldcv(15,35)                              # prints cross validation results with folds=15 and random_state=35