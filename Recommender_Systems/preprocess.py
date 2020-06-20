# Preprocess movie lens 20M dataset
import pandas as pd

# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('../LargeFiles/rating.csv')



# note:
# user ids are ordered sequentially from 1..138493
# with no missing numbers
# movie ids are integers from 1..131262
# NOT all movie ids appear
# there are only 26744 movie ids
# write code to check it yourself!


# Make the user ids go from 0...N-1
df.userId = df.userId - 1

# Create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
  movie2idx[movie_id] = count
  count += 1

# Add them to the data frame
# Takes awhile
df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

df = df.drop(columns=['timestamp'])

df.to_csv('../LargeFiles/edited_rating.csv', index=False)