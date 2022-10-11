from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def get_model_grid(model_name: str):
    model_name = model_name.lower()
    if model_name == "nb":
        text_clf = Pipeline([('mnb_clf', MultinomialNB())])
        parameters = {"mnb_clf__alpha": [0.5, 1]}
    elif model_name == "lr":
        text_clf = Pipeline([('lr_clf', LogisticRegression())])
        parameters = {"lr_clf__C": [1, 10, 100], "lr_clf__max_iter": [10000]}
    elif model_name == "svm":
        text_clf = Pipeline([('svc_clf', SVC())])
        parameters = {"svc_clf__C": [1, 10, 100], "svc_clf__kernel": ["poly", "rbf"]}
    elif model_name == "rf":
        text_clf = Pipeline([('rf_clf', RandomForestClassifier())])
        parameters = {"rf_clf__n_estimators": [50, 100], "rf_clf__criterion": ["gini", "entropy"],
                      'rf_clf__max_depth': [4, 6, 8]}
    elif model_name == "mlp":
        text_clf = Pipeline([('mlp_clf', MLPClassifier())])
        parameters = {
            'mlp_clf__hidden_layer_sizes': [(50, 50, 50), (50, 50)],
            'mlp_clf__activation': ['relu'],
            'mlp_clf__solver': ['adam'],
            'mlp_clf__alpha': [0.0001, 0.005],
            'mlp_clf__learning_rate': ['constant', 'adaptive']
        }
    else:
        raise Exception(f"Can't find model name: {model_name}. Please choose from `nb`, `svc`, `rf` or `mlp`!")

    return {"parameters": parameters, "pipe": text_clf}
