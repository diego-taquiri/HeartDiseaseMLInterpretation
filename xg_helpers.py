import urllib.request
import zipfile
import pandas as pd

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Any, Dict, Union
import xgboost as xgb
import pandas as pd

def extract_zip(src, dst, member_name):
    """
    Extract a member file from a zip file and read it into a pandas DataFrame.

    Parameters:
        src (str): URL of the zip file to be downloaded and extracted.
        dst (str): Local file path where the zip file will be written.
        member_name (str): Name of the member file inside the zip file to be read into a DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing the contents of the member file.
    """
    # Download the zip file from the given URL
    url = src
    fname = dst
    fin = urllib.request.urlopen(url)
    data = fin.read()
    
    # Write the downloaded data to the specified local file
    with open(dst, mode='wb') as fout:
        fout.write(data)
    
    # Extract the specified member file from the zip and read it into a DataFrame
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name))
        kag_questions = kag.iloc[0]  # Extracting the first row (might be headers or metadata)
        raw = kag.iloc[1:]  # Extracting all rows after the first one
    
    return raw



def hyperparameter_tuning(
    space: Dict[str, Union[float, int]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    early_stopping_rounds: int = 50,
    metric: callable = accuracy_score
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for an XGBoost classifier.

    This function takes a dictionary of hyperparameters, training and test data, and an optional value
    for early stopping rounds, and returns a dictionary with the loss and model resulting from the tuning process.
    The model is trained using the training data and evaluated on the test data. The loss is computed as the negative
    of the accuracy score.

    Parameters
    ----------
    space : Dict[str, Union[float, int]]
        A dictionary of hyperparameters for the XGBoost classifier.
    X_train : pd.DataFrame
        The training data.
    y_train : pd.Series
        The training target.
    X_test : pd.DataFrame
        The test data.
    y_test : pd.Series
        The test target.
    early_stopping_rounds : int, optional
        The number of early stopping rounds to use. The default value is 50.
    metric : callable, optional
        Metric to maximize. Default is accuracy_score.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the loss and model resulting from the tuning process.
        The loss is a float, and the model is an XGBoost classifier.
    """
    
    int_vals = ['max_depth', 'reg_alpha']
    space = {k: (int(val) if k in int_vals else val) for k, val in space.items()}
    space['early_stopping_rounds'] = early_stopping_rounds
    
    model = xgb.XGBClassifier(**space)
    evaluation = [(X_train, y_train), (X_test, y_test)]
    
    model.fit(X_train, y_train, eval_set=evaluation, verbose=False)
    pred = model.predict(X_test)
    
    score = metric(y_test, pred)
    
    return {'loss': -score, 'status': STATUS_OK, 'model': model}
