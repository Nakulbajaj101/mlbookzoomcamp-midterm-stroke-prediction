#!/usr/bin/env python
# coding: utf-8

import logging
import os
from typing import List

import bentoml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from processing.preprocessors import (BinaryEncoder, CleanStrings,
                                      ColumnDropperTransformer)

logging.basicConfig(level=logging.INFO)
MODEL_NAME = os.getenv("MODEL_NAME", "stroke_detection_model")

# Importing the data
def read_data(filepath=""): 
    """Function to read data"""
    
    data = pd.read_csv(filepath_or_buffer=filepath)
    return data



# Cleaning the columns
def make_columns_consistent(df: pd.DataFrame) -> pd.DataFrame:
    """Function to make column names consistent"""

    data = df.copy()
    columns = [cols.lower().replace(" ","_") for cols in data]
    data.columns = columns

    return data


# Defining the Pipeline Objects
def get_processing_pipelines(id: str, binary_encoder_column: str, categorical_columns: List):
    """Function to define and return model pipelines"""


    preprocess_pipeline = Pipeline(
        [
            ("dropping_id_column",
            
                ColumnDropperTransformer(
                    column_list=[id]
                )
            ),
            ("binary_encoder",
                BinaryEncoder(
                    column_name=binary_encoder_column
                )
            ),
            ("cleaning_strings",
                CleanStrings(
                    column_list=categorical_columns
                )
            )
        ])

    transform_pipeline = Pipeline(
        [   
            ("dict_vectorizer",
                DictVectorizer(sparse=False)
            ),
            ("scaling_data",
                MinMaxScaler()
            ),
            ("multiple_numeric_values_imputation",
                KNNImputer(add_indicator=True)
            )
        ]
    )

    return preprocess_pipeline, transform_pipeline


def preprocessing_and_transforming(df_train: pd.DataFrame, df_test: pd.DataFrame, preprocessor: Pipeline, transformer: Pipeline, target: str):
    """Function to preprocess and transform data"""

    train = df_train.copy()
    test = df_test.copy()
    
    # Preprocessed data
    preprocessed_train = preprocessor.fit_transform(train.drop([target],1))
    preprocessed_test = preprocessor.transform(test.drop([target],1))

    # Converting dataframes to dict objects
    train_dict = preprocessed_train.to_dict(orient='records')
    test_dict = preprocessed_test.to_dict(orient='records')

    # Transforming the data and getting ready for training the pipeline
    X_train = transformer.fit_transform(train_dict)
    X_test = transformer.transform(test_dict)

    y_train = train[target]
    y_test = test[target]

    return X_train, y_train, X_test, y_test, preprocessor, transformer


def model_training(X_train: dict, y_train: pd.Series):
    """Function to train and optimise the three models"""

    X_train = X_train.copy()
    y_train = y_train.copy()

    majority_class_instances = len(y_train) - y_train.sum()
    minority_class_instances = y_train.sum()
    weight_adjust_pos = majority_class_instances/minority_class_instances

    # Decision Tree
    d_param_grid = {
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8,10,20],
        'min_samples_leaf' : [1,3,5,10,20],
        'criterion' : ['gini', 'entropy'],
        'random_state' : [1], 
        'class_weight' : ['balanced']
    }
    d_clf = DecisionTreeClassifier(random_state=1, class_weight='balanced')


    # Logistic Regression
    lr_param_grid = {
        "C":np.logspace(-3,3,7), 
        "max_iter": [500, 1000,2000, 5000],
        'class_weight' : ['balanced'],
        'random_state' : [1]
        } 
        
    lr_clf = LogisticRegression()

    # Xgboost
    xgb_params = {
        'eta': [0.05, 0.1, 0.2],
        'max_depth': [4,5,6,7,8,10,20],
        'min_child_weight': [1,3,5,10,20],
        'n_estimators': [5, 10, 20, 50],
        'scale_pos_weight': [weight_adjust_pos],
        'objective':['binary:logistic'],
        'seed': [1],
        'verbosity': [1]
    }

    xgb_clf = xgb.XGBClassifier()


    # Training the models
    d_clf_cv = GridSearchCV(estimator=d_clf, param_grid=d_param_grid, cv=5, scoring='roc_auc')
    d_clf_cv.fit(X_train, y_train)

    logging.info("Decision tree optimised")

    lr_clf_cv = GridSearchCV(estimator=lr_clf, param_grid=lr_param_grid, cv=5, scoring='roc_auc')
    lr_clf_cv.fit(X_train, y_train)

    logging.info("Logistic regression optimised")


    xgb_clf_cv = GridSearchCV(estimator=xgb_clf, param_grid=xgb_params, cv=5, scoring='roc_auc')
    xgb_clf_cv.fit(X_train, y_train)

    logging.info("xgboost classifier optimised")

    lr_best_params = lr_clf_cv.best_params_
    d_best_params = d_clf_cv.best_params_
    xgb_best_params = xgb_clf_cv.best_params_


    # Training the best models
    lr_best_clf = LogisticRegression(**lr_best_params)
    d_best_clf = DecisionTreeClassifier(**d_best_params)
    xgb_best_clf = xgb.XGBClassifier(**xgb_best_params)

    lr_best_clf.fit(X_train, y_train)
    d_best_clf.fit(X_train, y_train)
    xgb_best_clf.fit(X_train, y_train)

    return d_best_clf, lr_best_clf, xgb_best_clf


def select_best_model(d_tree_clf: DecisionTreeClassifier, 
                      log_reg_clf: LogisticRegression,
                      xgb_clf: xgb.XGBClassifier, 
                      X_test: dict, 
                      y_test: pd.Series) -> tuple:
    
    """Function to evaluate models and return best model"""
    
    def evaluate(model, X_val, y_val):
        """Evaluation function to return recall"""

        predictions = model.predict_proba(X_val)[:,1]
        roc_auc = roc_auc_score(y_val, predictions)
        return roc_auc

    X_test = X_test.copy()
    y_test = y_test.copy()

    d_roc_auc = evaluate(d_tree_clf, X_val=X_test, y_val=y_test)
    lr_roc_auc = evaluate(log_reg_clf, X_val=X_test, y_val=y_test)
    xgb_roc_auc = evaluate(xgb_clf, X_val=X_test, y_val=y_test)
 
    # Models and scores dict
    model_performances = {
        "decision_tree" : {
            "model" : d_tree_clf,
            "roc_auc" : d_roc_auc
        },
        "xgboost" : {
            "model" : xgb_clf,
            "roc_auc" : xgb_roc_auc
        },
        "logistic_regression" : {
            "model" : log_reg_clf,
            "roc_auc" : lr_roc_auc
        }
    }

    logging.info(f"Models and their best performance scores \n {model_performances}")

    best_model = sorted(model_performances.items(), reverse=True, key=lambda score: score[1]['roc_auc'])[0]

    return best_model


def create_bento(best_model: tuple, preprocessor: Pipeline, transformer: Pipeline):
    """Function to create bento"""

    if best_model[0] == 'xgboost':
        logging.info(f"Should use bentoml xgboost framework")
        
        model = best_model[1]['model']
        bentoml.xgboost.save_model(
        name=f'{MODEL_NAME}',
        model=model,
        custom_objects={
            "preprocessor": preprocessor,
            "transformer": transformer
        },
        signatures={
            "predict_proba":{
                "batchable": True,
                "batch_dim": 0
            }
        }
        )

    else:
        logging.info(f"Should use bentoml scikit learn framework")
        model = best_model[1]['model']
        bentoml.sklearn.save_model(
        name=f'{MODEL_NAME}',
        model=model,
        custom_objects={
            "preprocessor": preprocessor,
            "transformer": transformer
        },
        signatures={
            "predict_proba":{
                "batchable": True,
                "batch_dim": 0
            }
        },
        )
    
    logging.info(f"Bento created")


if __name__ == "__main__":

    # Storing the column definitions so can run them through the pipeline
    id = 'id'
    target = 'stroke'
    
    categorical_columns = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                           'work_type', 'residence_type', 'smoking_status']
    
    binary_encoder_column = "ever_married"
    
    logging.info("Reading the data")
    data = read_data(filepath="healthcare-dataset-stroke-data.csv")

    logging.info("Making columns consistent")
    data = make_columns_consistent(df=data)

    logging.info("Splitting the data into train and test")
    train, test = train_test_split(data, test_size=0.1, random_state=1)

    logging.info("Defining preprocessing and data transformation pipelines")
    preprocess_pipeline, transform_pipeline = get_processing_pipelines(
        id=id,
        binary_encoder_column=binary_encoder_column,
        categorical_columns=categorical_columns
    )

    logging.info("Preprocessing and transforming data")
    X_train, y_train, X_test, y_test, preprocessor, transformer = preprocessing_and_transforming(
        df_train=train,
        df_test=test,
        preprocessor=preprocess_pipeline,
        transformer=transform_pipeline,
        target=target
    )

    
    logging.info("Training the three models and hypertuning")
    d_tree_clf, log_reg_clf, xgb_clf = model_training(X_train=X_train, y_train=y_train)


    logging.info("Selecting the best model")
    best_model = select_best_model(d_tree_clf=d_tree_clf,
                                   log_reg_clf=log_reg_clf,
                                   xgb_clf=xgb_clf,
                                   X_test=X_test,
                                   y_test=y_test)

    logging.info("Preparing the bento")
    create_bento(best_model=best_model, preprocessor=preprocessor, transformer=transformer)
