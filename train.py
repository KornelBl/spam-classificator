from sklearn.model_selection import RepeatedStratifiedKFold
from src.dataloader import DataLoader, DataConfig
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_rel
# from imblearn.pipeline import make_pipeline
from tabulate import tabulate
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
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

parametry = [16, 32, 64, 128, 256, 512, 1024, 2048]

n_estimators = [20]
max_depth = [20]
vectorizers = {
    "Bag of Words": CountVectorizer,
    "TFIDF": TfidfVectorizer
}
params = {'model__n_estimators': n_estimators,
          'model__max_depth': max_depth,
          'vectorizer__max_features': parametry}

results = {}
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

    _results = []
    for i in range(10):
        _results.append(grid.cv_results_[f"split{i}_test_score"])

    _results = np.array(_results).transpose()
    results[key] = _results

for key, values in results.items():
    plt.boxplot(values.T)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8],parametry)
    plt.xlabel("liczba cech")
    plt.ylabel("accuracy")
    plt.savefig(f"imgs/{key}_boxplot.png")
    plt.close()

for key, values in results.items():
    plt.plot(parametry, values.mean(axis=1))
plt.xlabel("liczba cech")
plt.ylabel("accuracy")
plt.legend(results.keys())
plt.xscale('log')
plt.xticks(parametry,parametry)
plt.ylim(0.7, 0.9)
plt.savefig("imgs/log_mean_accuracies.png")
plt.close()


def is_better(ttest_results, alpha=0.05):
    if ttest_results.statistic > 0 and ttest_results.pvalue < alpha:
        return True
    else:
        return False

dfs = {}
for key, values in results.items():
    df = pd.DataFrame()
    df["num"] = parametry
    df["results"] = list(values)
    df["mean_acc"] = [np.mean(r) for r in df["results"]]
    df["std_acc"] = [np.std(r) for r in df["results"]]
    df["better"] = \
        [[i for i, r in enumerate(df["results"]) if is_better(ttest_rel(result, r), alpha=0.05)] for result in df["results"]]

    dfs[key] = df
    df = df.drop("results", axis=1)

    print(f"Wyniki testów statystycznych dla {key}:")
    print(tabulate(df, headers='keys'))

bests = {}
for key, values in dfs.items():
    best_idx = values["mean_acc"].idxmax()
    bests[key] = values.loc[best_idx]

print(ttest_rel(bests["TFIDF"]["results"], bests["Bag of Words"]["results"]))
