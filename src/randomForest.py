import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'data/spotify_data_with_features.csv'
spotify_data = pd.read_csv(file_path)

# Handle missing values in 'Nationality' (if any)
spotify_data['Nationality'] = spotify_data['Nationality'].fillna('Unknown')

# One-hot encode 'Nationality'
nationality_encoded = pd.get_dummies(spotify_data['Nationality'], prefix='Nationality')

# Add the one-hot encoded features to the dataset
spotify_data = pd.concat([spotify_data, nationality_encoded], axis=1)


X = spotify_data[['Cluster', 'Is_Collab', 'Num_Artists', 'Popular_Artist'] +
                  list(nationality_encoded.columns)]
y = spotify_data['Is_Popular']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


random_forest = RandomForestClassifier(random_state=42)  
random_forest.fit(X_train, y_train)


y_pred = random_forest.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nModel Metrics Without Tuning:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Cross-Validation Scores
cv_accuracy = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
cv_precision = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='precision', n_jobs=-1)
cv_recall = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='recall', n_jobs=-1)
cv_f1 = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)

print("\nCross-Validation Metrics (5-fold):")
print(f"Mean Accuracy: {cv_accuracy.mean():.2f}")
print(f"Mean Precision: {cv_precision.mean():.2f}")
print(f"Mean Recall: {cv_recall.mean():.2f}")
print(f"Mean F1-Score: {cv_f1.mean():.2f}")

# Feature Importance Analysis
feature_importances = random_forest.feature_importances_
features = list(X_train.columns)

# Aggregate feature importances for Nationalities and other features
aggregated_importances = {}
nationalities_importances = {}

for i, feature in enumerate(features):
    if 'Nationality' in feature:
        nationalities_importances[feature] = feature_importances[i]
        aggregated_importances['Nationalities'] = aggregated_importances.get('Nationalities', 0) + feature_importances[i]
    else:
        aggregated_importances[feature] = feature_importances[i]

# Convert aggregated importances into a sorted list for plotting
sorted_aggregated = sorted(aggregated_importances.items(), key=lambda x: x[1], reverse=True)

# Separate keys (features) and values (importances) for aggregated plotting
aggregated_features = [item[0] for item in sorted_aggregated]
aggregated_values = [item[1] for item in sorted_aggregated]

# Plot aggregated feature importances
plt.figure(figsize=(10, 6))
plt.barh(aggregated_features, aggregated_values)
plt.xlabel('Importance')
plt.title('Aggregated Feature Importance')
plt.show()

# Plot top-k important nationalities
k = 20  # Number of top nationalities to display
sorted_nationalities = sorted(nationalities_importances.items(), key=lambda x: x[1], reverse=True)[:k]
nationality_features = [item[0] for item in sorted_nationalities]
nationality_values = [item[1] for item in sorted_nationalities]

plt.figure(figsize=(10, 6))
plt.barh(nationality_features, nationality_values)
plt.xlabel('Importance')
plt.title(f'Top {k} Feature Importance for Nationalities')
plt.show()
