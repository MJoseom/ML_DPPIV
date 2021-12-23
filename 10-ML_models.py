import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import svm, metrics, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score,  confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


"""Update the data"""
db = pd.read_csv('/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/06-Descriptors/05-Descriptors_DPPIV_final.csv',  sep=",")
print("Shape of df:",  db.shape)
print("INITIAL actives:", len(db[db["Group"] == 1]),  "inactives:", len(db[db["Group"] == 0]),  "proportion:",  len(db[db["Group"] == 0])/len(db[db["Group"] == 1]))


"""Preprocessing: Split the data"""
x = db.iloc[:,  4:]
y = db["Group"]
##Method1
#X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.7, random_state=10)
#print("actives:", len(Y_train[Y_train == 1]),  "inactives:", len(Y_train[Y_train == 0]),  "proportion:",  len(Y_train[Y_train == 0])/len(Y_train[Y_train == 1]))
##Method2 balanced
stratSplit = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=10)
for train_index, test_index in stratSplit.split(x, y):
    X_train, X_test = x.iloc[train_index,:], x.iloc[test_index,:]
    Y_train, Y_test = y[train_index], y[test_index]

    print("TRAINING actives:", len(Y_train[Y_train == 1]),  "inactives:", len(Y_train[Y_train == 0]),  "proportion:",  len(Y_train[Y_train == 0])/len(Y_train[Y_train == 1]))
    print("TEST actives:", len(Y_test[Y_test == 1]),  "inactives:", len(Y_test[Y_test == 0]),  "proportion:",  len(Y_test[Y_test == 0])/len(Y_test[Y_test == 1]))

splits = [X_train, X_test,  Y_train, Y_test]

"""Model Performance"""
def model_performance(ml_model, test_x, test_y, verbose=True):
   
    # Prediction probability on test set
    test_prob = ml_model.predict_proba(test_x)[:, 1]
    # Prediction class on test set
    test_pred = ml_model.predict(test_x)
    # Performance of model on test set
    accuracy = accuracy_score(test_y, test_pred)
    sens = recall_score(test_y, test_pred)
    spec = recall_score(test_y, test_pred, pos_label=0)
    auc = roc_auc_score(test_y, test_prob)
    cm  = confusion_matrix(test_y, test_pred)
    if verbose:
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Sensitivity: {sens:.2f}")
        print(f"Specificity: {spec:.2f}")
        print(f"AUC: {auc:.2f}")
        print("cm = ", cm )
    return accuracy, sens, spec, auc,  cm
    
"""Fitting the train set to the Machine Learning Model"""
def model_training_and_validation(ml_model, name, splits, verbose=True):

    train_x, test_x, train_y, test_y = splits
    # Fit the model
    ml_model.fit(train_x, train_y)
    # Calculate model performance results
    print("Model",  name)
    accuracy, sens, spec, auc,  cm = model_performance(ml_model, test_x, test_y, verbose)
    return accuracy, sens, spec, auc,  cm

"""RANDOM FOREST"""
param = {
    "n_estimators": 100,  # number of trees to grows
    "criterion": "entropy",  # cost function to be optimized for a split
}
model_RF = RandomForestClassifier(**param)
performance_measures = model_training_and_validation(model_RF, "RF", splits)

"""SUPORT VECTOR MACHINE"""
model_SVM = svm.SVC(kernel="rbf", C=1, gamma=0.1, probability=True)
performance_measures = model_training_and_validation(model_SVM, "SVM", splits)

"""ROC curves"""
def plot_roc_curves_for_models(models, test_x, test_y, save_png=True):
 
    fig, ax = plt.subplots()
    for model in models:
        # Select the model
        ml_model = model["model"]
        # Prediction probability on test set
        test_prob = ml_model.predict_proba(test_x)[:, 1]
        # Compute False postive rate and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(test_y, test_prob)
        # Calculate Area under the curve to display on the plot
        auc = roc_auc_score(test_y, test_prob)
        # Plot the computed values
        ax.plot(fpr, tpr, label=(f"{model['label']} AUC = {auc:.2f}"))

    # Custom settings for the plot
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves for DPP-IV inhibitors ML models ")
    ax.legend(loc="lower right",  fontsize=10)
    # Save plot
    if save_png:
        fig.savefig("./Roc_Curves_ML.png", dpi=300, bbox_inches="tight", transparent=True)

"""Cross-validation"""
def crossvalidation(ml_model, x, y, n_folds=5, verbose=False):
    t0 = time.time()
    # Shuffle the indices for the k-fold cross-validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=10)

    # Results for each of the cross-validation folds
    acc_per_fold = []
    sens_per_fold = []
    spec_per_fold = []
    auc_per_fold = []
    cm_per_fold = []

    for train_index, test_index in kf.split(x,  y):
        # clone model 
        fold_model = clone(ml_model)
     
        # Training an testing
        X_train, X_test = x.iloc[train_index,:], x.iloc[test_index,:]
        Y_train, Y_test = y[train_index], y[test_index]
        
        # Fit the model
        fold_model.fit(X_train, Y_train)

        # Performance for each fold
        accuracy, sens, spec, auc, cm = model_performance(fold_model, X_test, Y_test, verbose)

        # Save results
        acc_per_fold.append(accuracy)
        sens_per_fold.append(sens)
        spec_per_fold.append(spec)
        auc_per_fold.append(auc)
        cm_per_fold.append(cm)

    # Print statistics of results
    print(
        f"Mean accuracy: {np.mean(acc_per_fold):.2f} \t"
        f"and std : {np.std(acc_per_fold):.2f} \n"
        f"Mean sensitivity: {np.mean(sens_per_fold):.2f} \t"
        f"and std : {np.std(sens_per_fold):.2f} \n"
        f"Mean specificity: {np.mean(spec_per_fold):.2f} \t"
        f"and std : {np.std(spec_per_fold):.2f} \n"
        f"Mean AUC: {np.mean(auc_per_fold):.2f} \t"
        f"and std : {np.std(auc_per_fold):.2f} \n"
        f"CM: {cm_per_fold} \n"
        f"Time taken : {time.time() - t0:.2f}s\n"
    )

    return acc_per_fold, sens_per_fold, spec_per_fold, auc_per_fold

models = [{"label": "Model_RF", "model": model_RF},  {"label": "Model_SVM", "model": model_SVM}]
plot_roc_curves_for_models(models, X_test, Y_test)

folds = 10
for model in models:
    print("\n======= ")
    print(f"{model['label']}")
    crossvalidation(model["model"], x, y, n_folds=folds)
