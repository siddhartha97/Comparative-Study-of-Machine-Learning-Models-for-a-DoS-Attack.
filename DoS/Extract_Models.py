import pandas as pd
from Models.xgboost_model import XGBoost
from Models.random_forest_model import RandomForest
from Models.linear_regression_model import LinRegression


class ExtractModels:
    @staticmethod
    def extract_models():
        df = pd.read_csv('Dataset/Finalset.csv')
        df = df.ix[:, 2:]
        score = {
            'Linear_Regression_score': LinRegression(df).fit_model(),
            'Random_Forest_score': RandomForest(df).fit_model(),
            'XG_Boost_score': XGBoost(df).fit_model()
        }
        print(score)


if __name__ == '__main__':

    ExtractModels().extract_models()
