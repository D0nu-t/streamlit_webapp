import tweepy
import re
import streamlit as st
import nltk
import vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt

from textblob import TextBlob
print("test")
st.title("Twitter Sentiment Analysis")
consumer_key = "VPTkxzaRcirlNpqc23cKhxF0Q"
consumer_secret = "gha8zh85M5jlyyXfyPbyjDJhsN0shLh8U8FrBDLi0Rv4DtQrBR"
access_token = "1425347499477270528-eFClqGMbOFuPOZPg5nEe9SH6mmBL2g"
access_token_secret = "0h7e1B8Ix4MIRzWq4K7NtPHzi1l3TGd6sLTcBIzVtzwsT"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

tweets = tweepy.Cursor(api.search_tweets, q="covid").items(5000)
#print([t for t in tweets])
def preprocess_tweet(tweet):
        ##Preprocessing a tweet##

        link_ex = "https?:\/\/[A-Za-z0-9./]+"
        handle_ex = "@[A-Za-z0-9_]+"
        hashtag_ex = "#[A-Za-z0-9_]+"
        spcl_ex = "[^0-9A-Za-z ]"

        rt_ex = "RT"

        tweet = re.sub(link_ex, " ", tweet)         #removing links
        tweet = re.sub(handle_ex, " ", tweet)       #removing twitter handles
        tweet = re.sub(hashtag_ex, " ", tweet)      #removing hashtags
        tweet = re.sub(spcl_ex, " ", tweet)         #removing special characters
        tweet = re.sub(rt_ex, " ", tweet)           #removing retweet characters
        
        tweet = ''.join([i if ord(i) < 128 else ' ' for i in tweet])    #removing non-ascii characters

        tweet = ' '.join(tweet.split())             #removing extra white spaces
        return tweet
def get_tweets(topic, tweet_count):
  tweets = list()
  try:
    matched_tweets = api.search_tweets(q = topic, lang='en',count = tweet_count)
    for tweet in matched_tweets:
      parsed_tweet = {}
      parsed_tweet['text'] = preprocess_tweet(tweet.text)
                
      #Checking if the tweet is already present in our list because of it being retweeted
      if tweet.retweet_count > 0:
          if parsed_tweet not in tweets:
              tweets.append(parsed_tweet)
      else:
          tweets.append(parsed_tweet)
    return tweets
  except :
    print("Error: " +str(1))

st.write("Hit the button to access the search bar")

if(st.button("Press to search")):
    a=st.text_input("")
else:
    a="test"
collected_tweets = get_tweets(a,50000) 
#print(collected_tweets[:5])  
def get_textblob_results(text):
  results = {
      'subjectivity': TextBlob(text).sentiment.subjectivity,
      'polarity': TextBlob(text).sentiment.polarity,
  }
  return results

def classify_sentiment_textblob(text):
  results = get_textblob_results(text)
  if results['polarity'] > 0:
    sentiment = 1
  elif results['polarity'] == 0:
    sentiment = 0
  else:
    sentiment = -1
  return sentiment

sentiments = []
examples = []
eg_gap = 10

for i,tweet in enumerate(collected_tweets):
  sentiments.append(classify_sentiment_textblob(tweet['text']))

  if i%eg_gap == 0:
    examples.append([tweet['text'],classify_sentiment_textblob(tweet['text'])])

sentiment_count = Counter(sentiments)
sentiment_count
idx2sentiment = {
    -1 : 'negative',
    0 : 'neutral',
    1 : 'positive'
}

for t,s in examples:
  pass
  #print(t, " --> ", idx2sentiment[s])
analyzer = SentimentIntensityAnalyzer()

def get_vader_results(text):
  vs = analyzer.polarity_scores(text)
  return vs

def classify_sentiment_vader(text):
  results = get_vader_results(text)
  if results['compound'] >= 0.05:
    sentiment = 1
  elif results['compound'] <= -0.05:
    sentiment = -1
  else:
    sentiment = 0
  return sentiment

sentiments = []
examples = []
eg_gap = 10

for i,tweet in enumerate(collected_tweets):
  sentiments.append(classify_sentiment_vader(tweet['text']))

  if i%eg_gap == 0:
    examples.append([tweet['text'],classify_sentiment_vader(tweet['text'])])

sentiment_count = Counter(sentiments)
#sentiment_count
st.write("Percentage of positive tweets = ", (sentiment_count[1]/len(collected_tweets))*100, "%")
st.write("Percentage of negative tweets = ", (sentiment_count[-1]/len(collected_tweets))*100, "%")
st.write("Percentage of neutral tweets = ", (sentiment_count[0]/len(collected_tweets))*100, "%")
sentlist=[(sentiment_count[1]/len(collected_tweets))*100,(sentiment_count[-1]/len(collected_tweets))*100,(sentiment_count[0]/len(collected_tweets))*100]
if(sentlist[0]==max(sentlist)):
    st.write("keyword ",a, "mostly had positive results")
elif(sentlist[1]==max(sentlist)):
    st.write("keyword ",a, "mostly had negative results")
if(sentlist[2]==max(sentlist)):
    st.write("keyword ",a, "mostly had neutral results")
data={"positive":sentlist[0],"negative":sentlist[1],"neutral":sentlist[2]}
xv=list(data.keys())
yv=list(data.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(xv, yv, color ='maroon',
        width = 0.4)
plt.show()
st.pyplot(fig)
