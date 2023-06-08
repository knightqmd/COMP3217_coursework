import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the training data
train_df = pd.read_csv("C:\\Users\\93241\\OneDrive - University of Southampton\\COMP3217\\ml2\\TrainingDataMulti.csv",header=None)
features = train_df.iloc[:, :128]
labels = train_df.iloc[:, -1]


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(features)
#X_val_scaled = scaler.transform(labels)


# Step 2: Train a machine learning model with cross-validation
model = RandomForestClassifier()
model.fit(X_train_scaled, labels)
predictions = cross_val_score(model, X_train_scaled, labels, cv=5)
print(predictions)
# # Step 3: Generate cross-validation results
# cross_val_results_df = pd.DataFrame(train_df)
# cross_val_results_df['CrossValLabel'] = predictions
# cross_val_results_df.to_csv("C:\\Users\\93241\\OneDrive - University of Southampton\\COMP3217\\ml2\\CrossValidationResults.csv", index=False)

# # Step 4: Train a model on the entire training data
# final_model = RandomForestClassifier()
# final_model.fit(features, labels)

# # Step 5: Predict labels for testing data
# test_df = pd.read_csv("C:\\Users\\93241\\OneDrive - University of Southampton\\COMP3217\\ml2\\TestingDataMulti.csv")
# test_features = test_df.iloc[:, :128]
# test_predictions = final_model.predict(test_features)

# # Step 6: Generate Testing Results File
# test_results_df = pd.DataFrame(test_df)
# test_results_df['Label'] = test_predictions
# test_results_df.to_csv("C:\\Users\\93241\\OneDrive - University of Southampton\\COMP3217\\ml2\\TestingResultsMulti.csv", index=False)

# # Step 7: Report and analysis
# print("Testing Results:")
# print(test_results_df)

# Step 8: Evaluate the final model (optional)
# train_predictions = model.predict(features)
# train_accuracy = accuracy_score(labels, train_predictions)
# print("Training Accuracy:", train_accuracy)