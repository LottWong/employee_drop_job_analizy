#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:lott
Email:
===========================================
LogisticRegression...
===========================================
"""


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV

class Logistic():
    def read_data(self, file_path="./员工离职预测训练赛/pfm_train.csv",sep=","):
        data = pd.read_table(file_path, sep)
        return data

    def build_features(slef, data):
        data["AgeEducation"] = data["Education"] * 100 / data["Age"]
        data["AgeIncome"] = data["MonthlyIncome"] * data["JobInvolvement"] * data["TotalWorkingYears"] // data["Age"]
        data["Satisfaction"] = data['JobSatisfaction'] + data['EnvironmentSatisfaction'] + data['RelationshipSatisfaction']
        return data

    def pre_data(self, data):
        data = self.build_features(data)
        drop_cols = []
        for col in data.columns:
            length = len(data[col].unique())
            if length == 1:
                drop_cols.append(col);
        data.drop(drop_cols, axis=1, inplace=True)
        data.drop(["EmployeeNumber", "JobLevel"], axis=1, inplace=True)
        data.drop(["Education", "MonthlyIncome", "JobInvolvement", "TotalWorkingYears", "JobSatisfaction",
                   "EnvironmentSatisfaction", "RelationshipSatisfaction"], axis=1, inplace=True)
        # data.drop(["BusinessTravel", "Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"], axis=1, inplace=True)
        data = pd.get_dummies(data)
        return data

    def train(self, train_file_path):
        data = self.pre_data(self.read_data(train_file_path))
        y = data.Attrition
        X = data.drop(["Attrition"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y,random_state=27)
        self.model = linear_model.LogisticRegression(solver="liblinear", penalty="l2", C=2.4, max_iter=100)
        self.model.fit(self.X_train, self.y_train)
        pred = self.model.predict(X=self.X_test)
        score = metrics.accuracy_score(self.y_test, pred)
        score_true = metrics.recall_score(self.y_test, pred)
        score_false = metrics.recall_score(self.y_test, pred, pos_label=0)
        print("理论准确率：", self.model.score(self.X_train, self.y_train))
        print("实际准确率:", score)
        print("正例覆盖率：", score_true)
        print("负例覆盖率：", score_false)

    def predict(self, predict_file_path):
        data = self.pre_data(self.read_data(predict_file_path))
        self.X_test = data
        pred = self.model.predict(X=self.X_test)
        dataframe = pd.DataFrame({'result':pred})
        dataframe.to_csv("output.csv", index=False, sep=',')


if __name__ == "__main__":
    logis = Logistic()
    logis.train("./员工离职预测训练赛/pfm_train.csv")
    logis.predict("./员工离职预测训练赛/pfm_test.csv")