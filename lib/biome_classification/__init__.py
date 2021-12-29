import shap
import os
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def waterfall(model, X_test):
    pred_idx = np.where(model.classes_ == model.predict(X_test)[0][0])[0][0] # first sample in test set is classified as pred_idx in model_classes_
    shap.waterfall_plot(shap.Explanation(values=shap_values[pred_idx][row],
                                              base_values=explainer.expected_value[pred_idx], data=X_test.iloc[row],
                                         feature_names=X_test.columns.tolist()),
                   max_display=30)


def load_model():
    model = CatBoostClassifier(
            loss_function='MultiClass',
            custom_metric='Accuracy',
            learning_rate=0.15,
            random_seed=42,
            l2_leaf_reg=3,
            iterations=3)
    model_path = os.path.join('/kb/module/data', 'model_app.json')
    model.load_model(model_path, format='json')
    return model


def load_inference_data():
    # inference data set's first column is data sample id
    inference_data_path = os.path.join('/kb/module/data', 'enigma.tsv')
    with open(inference_data_path, 'r') as f:
        df = pd.read_csv(f, sep="\t")
    X = df.iloc[:, 1:]
    sample_id = df[df.columns[0]]
    return sample_id, X


def inference(model, sample_ids, inference_data):
    prediction = model.predict(inference_data)
    return prediction




