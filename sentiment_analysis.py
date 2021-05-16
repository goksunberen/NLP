from textblob import TextBlob
import pandas as pd
import os
from google_trans_new import google_translator
from emot.emo_unicode import EMOTICONS
import re

data_list = []
file_names = []

#Reading files
for root, dirs, files in os.walk(r"PATH TO DATASET"):
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                file_names.append(f.name[42:])
                data_list.append(text)

#Function to remove emoticons such as ':)' from text
def remove_emoticon(text):
    pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return pattern.sub(r'', text)

#We clean the data from emoticons by calling remove_emoticon function and then append the clean text to new_list
new_list = []
for i in data_list:
    text = remove_emoticon(i)
    new_list.append(text)

#Function that returns polarity of the parameter
# -1 is far negative, 1 is far positive, 0 is neutral
def polarity_result(text):
   return TextBlob(text).sentiment.polarity

positive = []
negative = []
neutral = []
emotions = []
polarity = []

pos_count = 0
neg_count = 0
neu_count = 0

#We translate all texts to English and classify them according to their polarity
translator = google_translator()
for txt in new_list:
    translation = translator.translate(txt,lang_tgt='en')
    result = polarity_result(translation)

    if result > 0:
        positive.append(txt)
        emotions.append("Positive")
        polarity.append(result)
        pos_count += 1
    elif result < 0:
        negative.append(txt)
        emotions.append("Negative")
        polarity.append(result)
        neg_count += 1
    else:
        neutral.append(txt)
        emotions.append("Neutral")
        polarity.append(result)
        neu_count += 1

#print(pos_count)
#print(neg_count)
#print(neu_count)

#We convert our lists into a dataframe for better visualization
df = pd.DataFrame({"File Name" : file_names, "Emotion" : emotions, "Polarity" : polarity})
#print(df)

#We save the dataframe as a txt file
df.to_csv(r'PATH TO WHERE YOU WOULD LIKE TO SAVE IT', header=None, index=None, sep='\t', mode='a')

#WORD SENTIMENT ANALYSIS
word = input("Please enter a word: ")
eng = translator.translate(word, lang_tgt = 'en')
polarity = polarity_result(eng)

if polarity > 0:
    print("Positive")
    print(polarity)
elif polarity < 0:
    print("Negative")
    print(polarity)
else:
    print("Neutral")
    print(polarity)
