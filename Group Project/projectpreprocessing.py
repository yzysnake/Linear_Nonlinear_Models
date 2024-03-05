import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, f1_score

import seaborn as sns
import matplotlib.pyplot as plt


def plot_learning_curves(model, X, y, cv):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=cv, n_jobs=-1,
                                                            train_sizes=np.linspace(.1, 1.0, 10),
                                                            scoring='accuracy')

    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plotting the learning curves
    plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Drawing bands for the standard deviation
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Creating plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_foldwise_scores(model, X, y, cv):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Ensure X and y are numpy arrays to simplify indexing in this context
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    train_scores = []
    val_scores = []

    # Perform cross-validation
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model.fit(X_train_fold, y_train_fold)

        y_train_pred = model.predict(X_train_fold)
        y_val_pred = model.predict(X_val_fold)

        train_f1 = f1_score(y_train_fold, y_train_pred)
        val_f1 = f1_score(y_val_fold, y_val_pred)

        train_scores.append(train_f1)
        val_scores.append(val_f1)

    # Plotting the scores across folds
    folds = range(1, cv.get_n_splits() + 1)
    plt.plot(folds, train_scores, 'o-', color="blue", label="Training Accuracy Score")
    plt.plot(folds, val_scores, 'o-', color="red", label="Validation Accuracy Score")

    # Adding plot details
    plt.title("Accuracy Scores across CV Folds")
    plt.xlabel("Fold"), plt.ylabel("Accuracy Score")
    plt.xticks(list(folds), labels=list(folds))
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    new_cm = [[cm[1, 1], cm[0, 1]], [cm[1, 0], cm[0, 0]]]

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(new_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Yes', 'No'], yticklabels=['Yes', 'No'])
    plt.title('Confusion Matrix (Yes Represents Approve)')
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.show()


def plot_random_forest_feature_importance(rf_model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Extract feature importances
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame for easier handling
    features = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })

    # Sort the features based on importance
    features_sorted = features.sort_values(by='Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(13, 10.4))
    bars = plt.barh(features_sorted['Feature'], features_sorted['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top

    # Adding the importance percentages at the end of each bar
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.0005
        plt.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width * 100:.2f}%', va='center')

    plt.show()
