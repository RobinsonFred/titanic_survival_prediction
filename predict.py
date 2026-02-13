import sys
import pandas as pd
from pandas import DataFrame, Series
import sklearn as skl
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
    # current Kaggle score: 0.78947
    @staticmethod
    def create_model() -> Pipeline:
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
            ))
        ])
        return model
    

class LogRegressionModel:
    # current Kaggle score: 0.75598
    @staticmethod
    def create_model() -> Pipeline:
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=42,
            ))
        ])
        return model


class GradientBoostingModel:
    # current Kaggle score: 0.76555
    @staticmethod
    def create_model() -> Pipeline:
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
            ))
        ])
        return model


MODELS = {
    'random_forest': RandomForestModel,
    'logistic_regression': LogRegressionModel,
    'gradient_boosting': GradientBoostingModel,
}

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        print(f"Usage: python predict.py <{'|'.join(MODELS.keys())}>")
        sys.exit(1)

    model_name = sys.argv[1]
    train_data, test_data = setup()
    features, label = prepare_train_data(train_data)
    filtered_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

    model = MODELS[model_name].create_model()
    predictions = fit_model_and_predict(model, features, label, test_data[filtered_features])

    output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
    output_path = f'data/{model_name}.csv'
    output.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    main()
