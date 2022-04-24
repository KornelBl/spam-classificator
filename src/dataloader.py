import nltk
import pandas as pd
import string
from dataclasses import dataclass
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

tqdm.pandas()

nltk.download('wordnet')
nltk.download('omw-1.4')


@dataclass
class DataConfig:
    remove_stopwords: bool
    remove_punctuation: bool
    remove_nonalpha: bool
    lemmatization: bool


class DataLoader:

    def __init__(self, data_config: DataConfig, data_path: str = "C:\Studia\II\sem_3\mpjn\data\spam.csv"):
        data = pd.read_csv(data_path, encoding="ISO-8859-1")
        self.stopwords = nltk.corpus.stopwords.words('english')
        data = data[data["Unnamed: 2"].isna()]
        data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
        data = data.rename(columns={"v1": "label", "v2": "text"})
        self.data = data
        self.config = data_config
        self.lemmatizer = WordNetLemmatizer()

    def get_data(self):
        print("Preprocessing text")
        X = self.data["text"].progress_apply(self.preprocess_text)
        y = self.data["label"].replace({"ham": 0, "spam": 1})
        return X, y

    def preprocess_text(self, text: str):
        text = word_tokenize(text, "english")
        text = [word.casefold() for word in text]
        if self.config.remove_punctuation:
            text = [word for word in text if word not in set(string.punctuation)]
        if self.config.remove_nonalpha:
            text = [word for word in text if word.isalpha()]
        if self.config.remove_stopwords:
            text = [word for word in text if word not in set(self.stopwords)]
        if self.config.lemmatization:
            text = [self.lemmatizer.lemmatize(word) for word in text]
        return " ".join(text)


if __name__ == '__main__':
    data_config = DataConfig(
        remove_stopwords=True,
        remove_nonalpha=True,
        remove_punctuation=True,
        lemmatization=True
    )
    dl = DataLoader(data_config)
    x, y = dl.get_data()
