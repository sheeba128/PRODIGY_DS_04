import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
# Load the dataset
df = pd.read_csv("C:\Sheeba C\Prodigy Infotech\Tasks\Twitter_Data.csv")
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    #  (like @username) and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = text.strip()
    # Convert text to lowercase
    text = text.lower()
    return text
# Replace 'text_column_name' with the actual name of your column
df['cleaned_text'] = df['clean_text'].apply(preprocess_text)
# Check for missing values in the 'clean_text' column
print(df['clean_text'].isnull().sum())
# Fill missing values with an empty string
df['clean_text'] = df['clean_text'].fillna('')
# Display the updated DataFrame
print(df)
# Define a function 
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity
# Apply the sentiment analysis function
df[['polarity', 'subjectivity']] = df['cleaned_text'].apply(lambda x: pd.Series(get_sentiment(x)))

# Plot the sentiment polarity distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['polarity'], bins=30, kde=True, color='blue')
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Polarity')
plt.ylabel('Frequency')
plt.show()

# Example: Sentiment over time (if you have a datetime column)
df['date'] = pd.to_datetime(df['date_column_name'])  # Replace with your date column
df.set_index('date', inplace=True)







