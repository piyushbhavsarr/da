import pandas as pd

# Read the dataset
data = pd.read_csv('https://www.kaggle.com/datasets/datasnaek/youtube-newselect=INvideos.csv')

# Perform data cleaning operations
data = data.dropna(subset=['views', 'likes', 'dislikes', 'comment_count'])

# Find the total views, total likes, total dislikes, and comment count
total_views = data['views'].sum()
total_likes = data['likes'].sum()
total_dislikes = data['dislikes'].sum()
total_comments = data['comment_count'].sum()

print(f'Total views: {total_views}')
print(f'Total likes: {total_likes}')
print(f'Total dislikes: {total_dislikes}')
print(f'Total comments: {total_comments}')