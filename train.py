from sklearn.model_selection import RepeatedStratifiedKFold
from src.dataloader import DataLoader, DataConfig
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from imblearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
import time
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5,
                               random_state=43474692)

data_config = DataConfig(
    remove_stopwords=True,
    remove_nonalpha=True,
    remove_punctuation=True,
    lemmatization=True
)
dl = DataLoader(data_config)
X, y = dl.get_data()



n_estimators = [int(x) for x in np.linspace(start=20, stop=50, num=2)]
max_depth = [int(x) for x in np.linspace(20, 90, num=3)]
vectorizers = {
    "countvectorizer": CountVectorizer,
    "tfidfvectorizer": TfidfVectorizer
}
params = {'model__n_estimators': n_estimators,
          'model__max_depth': max_depth,
          'vectorizer__max_features':[256,512,1024]}


for key, vectorizer in vectorizers.items():

    pipe = Pipeline([("vectorizer",vectorizer()),
                         ("model",RandomForestClassifier())], memory="cached_pipe")
    grid = GridSearchCV(pipe, params, cv=rskf, scoring='balanced_accuracy',
                        verbose=1, return_train_score=True, error_score='raise',
                        n_jobs=8)
    start = time.time()
    grid.fit(X,y)
    end = time.time()
    print(f"Grid Search took {end-start} seconds")
    print((grid.best_estimator_, grid.best_score_))

    models = grid.cv_results_["params"]
    scores = grid.cv_results_["mean_test_score"]


