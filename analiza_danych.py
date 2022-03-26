import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import nltk
from wordcloud import WordCloud



data = pd.read_csv("data/spam.csv", encoding="ISO-8859-1")

data = data[data["Unnamed: 2"].isna()]

data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

data["v1"].hist(weights=np.ones(len(data)) / len(data))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.savefig("imgs/histogram.png")
plt.close()

data["length"] = [len(text) for text in data["v2"]]
data["length"].hist(bins=100)
plt.savefig("imgs/histogram_length.png")
plt.close()

stopwords = nltk.corpus.stopwords.words('english')
stopwords = set(stopwords)
text = " ".join(data[data["v1"]=="ham"]["v2"])
allWords = nltk.tokenize.word_tokenize(text)
allWordDist = nltk.FreqDist(w.lower() for w in allWords)
allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w.lower() not in stopwords and w.isalnum())
allWordExceptStopDist.most_common(30)
text = " ".join([w.lower() for w in allWords if w.lower() not in stopwords and w.isalnum()])
plt.close()


wordcloud = WordCloud(max_words=200,width=800,height=400,background_color="white").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("imgs/wordcloud_good.png")


text = " ".join(data[data["v1"]=="spam"]["v2"])
allWords = nltk.tokenize.word_tokenize(text)
allWordDist = nltk.FreqDist(w.lower() for w in allWords)
allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w.lower() not in stopwords and w.isalnum())
allWordExceptStopDist.most_common(30)
spam_text = " ".join([w.lower() for w in allWords if w.lower() not in stopwords and w.isalnum()])


wordcloud = WordCloud(max_words=200,width=800,height=400,background_color="white",colormap="plasma").generate(spam_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("imgs/wordcloud_spam.png")
plt.close()
