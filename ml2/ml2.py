import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("C:\\Users\\93241\\OneDrive - University of Southampton\\COMP3217\\ml2\\TrainingDataMulti.csv",header=None)
# Separate the features and the labels
features = df.iloc[:, :128]  # Select all columns except the last one
labels = df.iloc[:, -1]  # Select only the last column


# Step 2: Train a machine learning model
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=11)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Step 4: Predict labels for testing data
test_df = pd.read_csv("C:\\Users\\93241\\OneDrive - University of Southampton\\COMP3217\\ml2\\TestingDataMulti.csv",header=None)
test_features = test_df.iloc[:, :128]
test_features_scaled = scaler.transform(test_features)

def logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val,test_features_scaled,test_df):
    # Choose and train a model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Step 3: Evaluate the model
    train_predictions = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print("Logistic_Training Accuracy:", train_accuracy)

    val_predictions = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print("Logistic_Validation Accuracy:", val_accuracy)


    test_predictions = model.predict(test_features_scaled)

    # Step 5: Generate Testing Results File
    test_results_df = pd.DataFrame(test_df)
    test_results_df['Label'] = test_predictions
    test_results_df.to_csv("C:\\Users\\93241\\OneDrive - University of Southampton\\COMP3217\\ml2\\LogisticTestingDataMulti.csv", index=False)

    # Step 6: Report and analysis
    print("Testing Results:")
    print(test_results_df['Label'])
    classification_report(y_val, val_predictions)

def knn(X_train_scaled, y_train, X_val_scaled, y_val,test_features_scaled,test_df):
    # Choose and train a model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    # Step 3: Evaluate the model
    train_predictions = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print("KNN_Training Accuracy:", train_accuracy)

    val_predictions = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print("KNN_Validation Accuracy:", val_accuracy)

    test_predictions = model.predict(test_features_scaled)

    # Step 5: Generate Testing Results File
    test_results_df = pd.DataFrame(test_df)
    test_results_df['Label'] = test_predictions
    test_results_df.to_csv("C:\\Users\\93241\\OneDrive - University of Southampton\\COMP3217\\ml2\\KnnTestingDataMulti.csv", index=False)

    # Step 6: Report and analysis
    print("Testing Results:")
    print(test_results_df['Label'])
    classification_report(y_val, val_predictions)

def random_forest(X_train_scaled, y_train, X_val_scaled, y_val, test_features_scaled, test_df):
    # Choose and train a model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_scaled, y_train)

    # Step 3: Evaluate the model
    train_predictions = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_precision = precision_score(y_train, train_predictions, average='macro')
    train_recall = recall_score(y_train, train_predictions, average='macro')
    print("RandomForest_Training Accuracy:", train_accuracy)
    print("RandomForest_Training Precision:", train_precision)
    print("RandomForest_Training Recall:", train_recall)

    val_predictions = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions, average='macro')
    val_recall = recall_score(y_val, val_predictions, average='macro')
    print("RandomForest_Validation Accuracy:", val_accuracy)
    print("RandomForest_Validation Precision:", val_precision)
    print("RandomForest_Validation Recall:", val_recall)

    test_predictions = model.predict(test_features_scaled)

    # Step 5: Generate Testing Results File
    test_results_df = pd.DataFrame(test_df)
    test_results_df['Label'] = test_predictions
    test_results_df.to_csv("C:\\Users\\93241\\OneDrive - University of Southampton\\COMP3217\\ml2\\RandomForestTestingDataMulti.csv", index=False)

    # Step 6: Report and analysis
    print("Testing Results:")
    for i in test_results_df['Label']:
        print(i,end='  ')
    
    print("\n")

    # Print confusion matrix
    cm = confusion_matrix(y_val, val_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Print classification report
    report = classification_report(y_val, val_predictions)
    print("Classification Report:")
    print(report)

def svm(X_train_scaled, y_train, X_val_scaled, y_val,test_features_scaled,test_df):
    # Choose and train a model
    model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)

    # Step 3: Evaluate the model
    train_predictions = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print("SVM_Training Accuracy:", train_accuracy)

    val_predictions = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print("SVM_Validation Accuracy:", val_accuracy)

    test_predictions = model.predict(test_features_scaled)

    # Step 5: Generate Testing Results File
    test_results_df = pd.DataFrame(test_df)
    test_results_df['Label'] = test_predictions
    test_results_df.to_csv("C:\\Users\\93241\\OneDrive - University of Southampton\\COMP3217\\ml2\\SVMTestingDataMulti.csv", index=False)

    # Step 6: Report and analysis
    print("Testing Results:")
    print(test_results_df['Label'])
    classification_report(y_val, val_predictions)

def decision_tree(X_train_scaled, y_train, X_val_scaled, y_val,test_features_scaled,test_df):
    # Choose and train a model
    model = DecisionTreeClassifier()
    model.fit(X_train_scaled, y_train)

    # Step 3: Evaluate the model
    train_predictions = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print("DecisionTree_Training Accuracy:", train_accuracy)

    val_predictions = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print("DecisionTree_Validation Accuracy:", val_accuracy)

    test_predictions = model.predict(test_features_scaled)

    # Step 5: Generate Testing Results File
    test_results_df = pd.DataFrame(test_df)
    test_results_df['Label'] = test_predictions
    test_results_df.to_csv("C:\\Users\\93241\\OneDrive - University of Southampton\\COMP3217\\ml2\\DecisionTestingDataMulti.csv", index=False)

    # Step 6: Report and analysis
    print("Testing Results:")
    print(test_results_df['Label'])
    classification_report(y_val, val_predictions)

# logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val,test_features_scaled,test_df)
# knn(X_train_scaled, y_train, X_val_scaled, y_val,test_features_scaled,test_df)
random_forest(X_train_scaled, y_train, X_val_scaled, y_val,test_features_scaled,test_df)
# svm(X_train_scaled, y_train, X_val_scaled, y_val,test_features_scaled,test_df)
# decision_tree(X_train_scaled, y_train, X_val_scaled, y_val,test_features_scaled,test_df)



