from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

features = ['popular_artist', 'Cluster', 'artist_num', 'is_collab', 
            'Danceability', 'Energy', 'Loudness', 'Speechiness', 
            'Acousticness', 'Instrumentalness', 'Valence']
target = 'is_top15percent'

df = pd.read_csv('Spotify_DF_Train.csv',sep='^')

df['is_collab'] = (df['artist_num'] > 1).astype(int)

musical_features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 
                    'Acousticness', 'Instrumentalness', 'Valence']
scaler = StandardScaler()
df[musical_features] = scaler.fit_transform(df[musical_features])

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree = decision_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)
y_pred_train = decision_tree.predict(X_train)

accuracy_train = accuracy_score(y_pred_train, y_train)
report_train = classification_report(y_pred_train, y_train)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

print(f"Training Accuracy: {accuracy_train}")
print("Training Classification Report:")
print(report_train)

#DEBUG
print(f"Dimension of X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

feature_importance = decision_tree.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, align='center')
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Decision Tree')
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.show()
