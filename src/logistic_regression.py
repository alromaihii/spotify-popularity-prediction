from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
y_pred_train = logreg.predict(X_train)

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

feature_importance = np.abs(logreg.coef_[0])

normalized_importance = (feature_importance / feature_importance.sum()) * 100

plt.figure(figsize=(10, 6))
plt.barh(features, normalized_importance, align='center')
plt.xlabel('Normalized Feature Importance (%)')
plt.title('Normalized Feature Importance in Logistic Regression')
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.show()