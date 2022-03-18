
from numpy import mean
from numpy import std
from numpy import arange
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
 
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
	return X, y
 
# liste des models à evaluer
def get_models():
	models = dict()
	# ratio de 10% à 100%
	for i in arange(0.1, 1.1, 0.1):
		key = '%.1f' % i
		# max_samples
		if i == 1.0:
			i = None
		models[key] = RandomForestClassifier(max_samples=i)
	return models
 

def evaluate_model(model, X, y):
	
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
 
# definition dataset
X, y = get_dataset()
# model à evaluer
models = get_models()
results, names = list(), list()
for name, model in models.items():
	# evaluation du model
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	# performance
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# perfomance mais en graphique
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()