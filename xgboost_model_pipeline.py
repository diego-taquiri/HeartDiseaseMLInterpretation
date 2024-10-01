"""
Instructions

- Fill the methods and functions that currently raise NotImplementedError.
- The data should be split in test (20%) and training (80%) sets outside the Model
- The Model should predict the target column from all the remaining variables.
- For the modeling methods `train`, `predict` and `eval` you can use any appropriate method.
- Use an appropriate metric based on the data and try to get the best results
- Your solution will be judged on both correctness and code quality.
- Do not modify the part of the code which is below this message -> # You should not have to modify the code below this point
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials
import shap
import yellowbrick.model_selection as ms
from yellowbrick import classifier

import xg_helpers as xhelp

class Model:
    def __init__(self, train_df: pl.DataFrame = None, test_df: pl.DataFrame = None) -> None:
        """
        Initialize the Model as necessary

        Args:
            train_df (pl.DataFrame): training data
            test_df (pl.DataFrame): test data
        """
        print("#### Initializing Model ####")
        self.train_df = train_df
        self.test_df = test_df
        self.model = None
        print("Model initialized.\n")

    
    def process_data(self, df: pl.DataFrame = None) -> pl.DataFrame:
        """
        Prepare the data as needed. If df is None,
        then process the training data passed in the constructor
        and also the test data.

        Args:
            df (pl.DataFrame): data
        """
        # Note: No extensive data preprocessing (e.g., handling null values, scaling numeric columns,
        # or encoding categorical variables) is done here because XGBoost can handle null values, 
        # does not require feature scaling, and can work with categorical features directly. 
        # Performing these preprocessing steps might actually hurt performance.
        
        print("#### Processing Data ####")
        if df is None:
            # Process both train and test data
            self.train_df = self.train_df.with_columns(
                pl.when(pl.col("sex") == "Male").then(1).otherwise(0).alias("sex")
            )
            self.test_df = self.test_df.with_columns(
                pl.when(pl.col("sex") == "Male").then(1).otherwise(0).alias("sex")
            )
            print("Training and test data processed.\n")
        else:
            # If the input df is a pandas DataFrame, convert it to a Polars DataFrame
            if isinstance(df, pd.DataFrame):
                df = pl.from_pandas(df)

            processed_df = df.with_columns(
                pl.when(pl.col("sex") == "Male").then(1).otherwise(0).alias("sex")
            )
            print("Provided data processed.\n")
            return processed_df

    def train(self) -> None:
        """
        Train a Machine Learning model on the training set passed in the constructor
        """
        print("#### Starting Model Training ####")
        
        X_train = self.train_df.select(pl.exclude("target")).to_pandas()
        y_train = self.train_df.select(pl.col("target")).to_pandas()["target"]
        X_test = self.test_df.select(pl.exclude("target")).to_pandas()
        y_test = self.test_df.select(pl.col("target")).to_pandas()["target"]
        
        print("Data split into training and testing sets.")
        print("#### Hyperparameter Tuning ####")
        
        # Hyperparameter tuning using Step-wise Tuning with Hyperopt
        params = {'random_state': 42}
        rounds = [
            {'max_depth': hp.quniform('max_depth', 2, 3, 1), 
             'min_child_weight': hp.loguniform('min_child_weight', -2, 3)},
            {'subsample': hp.uniform('subsample', 0.5, 1),  
             'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)},
            {'reg_alpha': hp.uniform('reg_alpha', 0, 10),
             'reg_lambda': hp.uniform('reg_lambda', 1, 10)},
            {'gamma': hp.loguniform('gamma', -10, 10)}, 
            {'learning_rate': hp.loguniform('learning_rate', -7, 0)}
        ]

        all_trials = []

        for round in rounds:
            params = {**params, **round}
            trials = Trials()
            best = fmin(
                fn=lambda space: xhelp.hyperparameter_tuning(space, X_train, y_train, X_test, y_test),
                space=params,
                algo=tpe.suggest,
                max_evals=20,
                trials=trials
            )
            params = {**params, **best}
            all_trials.append(trials)

        print("Hyperparameter tuning completed.\n")
        print("#### Final Model Training ####")

        # Final training with best parameters previously found with this tuning method.
        step_params = {'random_state': 42,
                        'max_depth': 3,
                        'min_child_weight': np.float64(0.15278032592469923),
                        'subsample': np.float64(0.983566704125629),
                        'colsample_bytree': np.float64(0.7103794547211446),
                        'reg_alpha': np.float64(0.46664213417395684),
                        'reg_lambda': np.float64(4.506746421698594),
                        'gamma': np.float64(0.0002468406479022509),
                        'learning_rate': np.float64(0.31841477918128425)
        }

        # Calculate scale_pos_weight to handle class imbalance
        num_negative = len(y_train[y_train == 0])
        num_positive = len(y_train[y_train == 1])
        scale_pos_weight = num_negative / num_positive

        self.model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                                       **step_params, 
                                       early_stopping_rounds=50,
                                       n_estimators=20,
                                       eval_metric=['auc', 'aucpr'])
        self.model.fit(X_train, y_train,
                       eval_set=[(X_train, y_train), 
                                 (X_test, y_test)
                                ], 
                       verbose=100
                      ) 
        
        print("Training completed. Model is ready.\n")

        plot_dir = 'eval_plots'
        os.makedirs(plot_dir, exist_ok=True)

        print("#### Plotting Learning Curve ####")
        fig, ax = plt.subplots(figsize=(8, 4))
        viz = ms.learning_curve(xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                                       **step_params, 
                                       n_estimators=20,
                                       eval_metric=['auc', 'aucpr']), X_train, y_train, ax=ax)
        ax.set_ylim(0.6, 1)
        plt.savefig(os.path.join(plot_dir, 'learning_curve.png'))
        plt.close()
        print(f"Learning curve plot saved as '{os.path.join(plot_dir, 'learning_curve.png')}'.\n")

        
    def predict(self, df: pd.DataFrame = None, output_path: str = "./heart_dataset_predictions.csv") -> pd.DataFrame:
        """
        Predict outcomes with the model on the data passed as argument.
        Assumed the data has been processed by the function process_data
        If the argument is None, work on the test data passed in the constructor.
        
        Args:
            df (pd.DataFrame): data
        """
        # If no dataframe is provided, use the test set
        if df is None:
            df = self.test_df

        # Ensure the DataFrame is a Polars DataFrame and convert to Pandas
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Make predictions using the trained model
        predictions = self.model.predict(df)
        prediction_df = pd.DataFrame(predictions, columns=['prediction'])

        # Append predictions to the original DataFrame
        df_with_predictions = df.copy()
        df_with_predictions['prediction'] = prediction_df['prediction']

        if output_path:
            df_with_predictions.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")

        return prediction_df
    
    def eval(self, df: pd.DataFrame = None) -> None:
        """
        Evaluate the model in the data passed as argument and print proper metrics.
        And create a short summary of the model you have trained.
        If df is None, then eval the test data passed in the constructor
        
        Args:
            df (pd.DataFrame): data
        """
        print("#### Starting Model Evaluation ####")

         # Determine which dataset to use
        if df is None:
            X_eval = self.test_df.select(pl.exclude("target")).to_pandas()
            y_eval = self.test_df.select(pl.col("target")).to_pandas()["target"]
        else:
            X_eval = df.select(pl.exclude("target")).to_pandas()
            y_eval = df.select(pl.col("target")).to_pandas()["target"]

        # Predict outcomes
        predictions = self.model.predict(X_eval)
        
        # Calculate metrics
        accuracy = self.model.score(X_eval, y_eval)
        cm = metrics.confusion_matrix(y_eval, predictions)
        precision = metrics.precision_score(y_eval, predictions)
        recall = metrics.recall_score(y_eval, predictions)
        f1 = metrics.f1_score(y_eval, predictions)
        auc = metrics.roc_auc_score(y_eval, predictions)

        # Print metrics in a table-like format
        print("#### Evaluation Metrics ####")
        print(f"{'Metric':<12} {'Value':<10}")
        print(f"{'-'*12} {'-'*10}")
        print(f"{'Accuracy':<12} {accuracy:.4f}")
        print(f"{'Precision':<12} {precision:.4f}")
        print(f"{'Recall':<12} {recall:.4f}")
        print(f"{'F1 Score':<12} {f1:.4f}")
        print(f"{'AUC':<12} {auc:.4f}")
        print()
        
        # Print the confusion matrix in a readable format
        print("Confusion Matrix:")
        print(f"{'':<12} {'Predicted: 0':<15} {'Predicted: 1':<15}")
        print(f"{'Actual: 0':<12} {cm[0][0]:<15} {cm[0][1]:<15}")
        print(f"{'Actual: 1':<12} {cm[1][0]:<15} {cm[1][1]:<15}")
        print()
        
        # Create directory for plots
        print("#### Saving Evaluation Plots ####")
        plot_dir = 'eval_plots'
        os.makedirs(plot_dir, exist_ok=True)

        # Save confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 4))
        cm = metrics.confusion_matrix(y_eval, predictions,
                                      normalize='true')
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
        disp.plot(ax=ax, cmap='Blues')
        plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
        plt.close()

        # Save F1 Score Report plot
        fig, ax = plt.subplots(figsize=(8, 4))
        classifier.classification_report(self.model, X_eval, y_eval, ax=ax)
        plt.savefig(os.path.join(plot_dir, 'classification_report.png'))
        plt.close()

        # Save ROC curve plot
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics.RocCurveDisplay.from_estimator(self.model, X_eval, y_eval, ax=ax, label='Test ROC Curve')
        metrics.RocCurveDisplay.from_estimator(self.model, self.train_df.select(pl.exclude("target")).to_pandas(), 
                                            self.train_df.select(pl.col("target")).to_pandas()["target"], 
                                            ax=ax, label='Train ROC Curve')
        ax.set(title='ROC plots for the model (Train vs. Test)')
        plt.savefig(os.path.join(plot_dir, 'roc_curve_train_test.png'))

        # Feature Importance Plot
        fig, ax = plt.subplots(figsize=(8, 4)) 
        (pd.Series(self.model.feature_importances_, index=X_eval.columns)
        .sort_values()
        .plot.barh(ax=ax))
        plt.savefig(os.path.join(plot_dir, 'feature_importance.png'))
        plt.close()
        
        # SHAP Values
        shap.initjs()
        shap_ex = shap.TreeExplainer(self.model)
        vals = shap_ex(X_eval)

        # SHAP Waterfall Plot
        fig = plt.figure(figsize=(8, 4)) 
        shap.plots.waterfall(vals[1], show=False)
        plt.savefig(os.path.join(plot_dir, 'shap_waterfall.png'))
        plt.close()

        # SHAP Beeswarm Plot
        fig = plt.figure(figsize=(8, 4)) 
        shap.plots.beeswarm(vals, max_display=len(X_eval.columns))
        plt.savefig(os.path.join(plot_dir, 'shap_beeswarm.png'))
        plt.close()

        print(f"Evaluation plots are saved in the '{plot_dir}' directory.")
        print("Model evaluation completed.")
    
    def save(self, path: str) -> None:
        """
        Save the model so it can be reused

        Args:
            path (str): path to save the model
        """
        print(f"#### Saving Model to {path} ####")
        self.model.save_model(path)
        print(f"Model saved to {path}\n")

    @staticmethod
    def load(path: str) -> Model:
        """
        Reload the Model from the saved path so it can be re-used.
        
        Args:
            path (str): path where the model was saved to
        """
        print(f"#### Loading Model from {path} ####")

        loaded_model = Model(train_df=None, test_df=None)

        loaded_model.model = xgb.XGBClassifier()
        loaded_model.model.load_model(path)

        print(f"Model loaded from {path}\n")

        return loaded_model



def main():
    
    data_path = "./heart_dataset.csv"
    path_to_save = "./xgb_trained.ubj"
    predict_path = "./heart_dataset_inference.csv"
    output_path = "" 
    
    # Read data 
    df = pl.read_csv(data_path)

    # Split data into training and test set by using df
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    #################################################################################
    # You should not have to modify the code below this point
    
    # Define the model
    model = Model(train_df, test_df)
    
    # Process data
    model.process_data()
    
    # Train model
    model.train()
    
    # Evaluate performance
    model.eval()
    
    # Save model
    model.save(path_to_save)
    
    # Load model
    loaded_model = Model.load(path_to_save)
    
    # Predict results of the predict data
    predict_df = pd.read_csv(predict_path)
    
    predict_df = loaded_model.process_data(predict_df)
    outcomes = loaded_model.predict(predict_df)
    print(f"Predicted on predict data: {outcomes}\n")

if __name__ == '__main__':
    main()
    

