import pandas as pd
from pandas import DataFrame, Series
import sklearn as skl
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from numpy import ndarray


def setup() -> tuple[DataFrame, DataFrame]:
        train_data = pd.read_csv('data/train.csv', header=0)
        test_data = pd.read_csv('data/test.csv', header=0)
        return train_data, test_data
    
def prepare_train_data(data: DataFrame) -> tuple[DataFrame, DataFrame]:
        filtered_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        features = pd.get_dummies(data[filtered_features])
        label = data['Survived']
        return features, label
    
def fit_model_and_predict(model: BaseEstimator,
                          features: DataFrame,
                          label: Series,
                          test_data: DataFrame) -> ndarray:
      model.fit(features, label)
      predictions = model.predict(pd.get_dummies(test_data))
      return predictions

class RandomForestModel:
    
    @staticmethod
    def create_model() -> RandomForestClassifier:
        model = RandomForestClassifier(n_estimators=100, 
                                       max_depth=5, 
                                       min_samples_split=2, 
                                       min_samples_leaf=1,
                                       random_state=42,)
        return model
    

class LogRegressionModel:


class GradientBoostingModel:
