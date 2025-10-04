import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Spotify_Dataset_V3.csv',delimiter=';')

# Set Date as index if not already
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day_name()
df.set_index('Date', inplace=True)

# Separate data for top 10 ranks and all ranks
df_top50 = df[df['Rank'].between(1, 50)]
df_grouped_all = df[['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence']].groupby('Date').mean()
df_grouped_top50 = df_top50[['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence']].groupby('Date').mean()

# Plot subplots
features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence']
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 10))

# Remove the empty subplot (since we have 7 features but 8 subplots)
axes[3, 1].axis('off')

# Plot each feature in a subplot
for i, feature in enumerate(features):
    ax = axes[i//2, i%2]
    # Plot the average for all ranks
    ax.plot(df_grouped_all.index, df_grouped_all[feature], label=f'All ranks', color='blue')
    # Plot the average for top 10 ranks
    ax.plot(df_grouped_top50.index, df_grouped_top50[feature], label=f'Top 50 ranks', color='orange')
    ax.set_title(feature)
    ax.set_xlabel('Date')
    ax.set_ylabel('Average')
    ax.legend()

# Adjust the layout
plt.tight_layout()
plt.show()
