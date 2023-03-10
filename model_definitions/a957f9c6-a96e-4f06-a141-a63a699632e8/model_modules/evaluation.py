from sklearn import metrics
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame

import os
import joblib
import json


def save_plot(title):
    import matplotlib.pyplot as plt

    plt.title(title)
    fig = plt.gcf()
    filename = title.replace(" ", "_").lower()
    fig.savefig('artifacts/output/{}'.format(filename), dpi=500)
    plt.clf()


def evaluate(data_conf, model_conf, **kwargs):
    model = joblib.load('artifacts/input/xgb_model.joblib')

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database="AOA_MASTER")

    # Read test dataset from Teradata
    # As this is for demo purposes, we simulate the test dataset changing between executions
    # by introducing a random sample. Note that the sampling is performed in Teradata!
    test_df = DataFrame("ACC_TEST")
    test_df = test_df.to_pandas()

    X_test = test_df[model.feature_names]
    y_test = test_df[model.target_name]

    print("Scoring")
    y_pred = model.predict(test_df[model.feature_names])

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)
        
    metrics.plot_confusion_matrix(model, X_test, y_test)
    save_plot('Confusion Matrix', context=context)

    metrics.plot_roc_curve(model, X_test, y_test)
    save_plot('ROC Curve', context=context)


