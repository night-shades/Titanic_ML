import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


def main():
    # 專案根目錄：.../Titanic_ML
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 你的 csv 在 titanic/ 內
    train_path = os.path.join(project_root, "titanic", "train.csv")
    test_path = os.path.join(project_root, "titanic", "test.csv")

    # 讀取資料
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # y 是答案欄位（是否生存）
    y = train_df["Survived"]

    # baseline 特徵欄位（先挑乾淨常用的）
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # 切訓練/驗證集（固定 random_state 讓結果可重現）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 數值/類別欄位分開處理
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            # 用中位數補缺失（Age/Fare 常缺）
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            # 用眾數補缺失（Embarked 可能缺）
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # one-hot：把文字類別轉成 0/1 欄位
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 模型：邏輯迴歸（baseline）
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    # 訓練
    clf.fit(X_train, y_train)

    # 驗證
    val_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"[Validation] Accuracy = {acc:.4f}")
    print(classification_report(y_val, val_pred, digits=4))

    # 對 test 做預測 → 產生 Kaggle submission
    test_pred = clf.predict(X_test)
    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": test_pred.astype(int)}
    )

    os.makedirs(os.path.join(project_root, "reports"), exist_ok=True)
    submission_path = os.path.join(project_root, "reports", "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission file to: {submission_path}")

    # 存模型
    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    joblib.dump(clf, os.path.join(project_root, "models", "baseline_model.joblib"))
    print("Saved model to: models/baseline_model.joblib")


if __name__ == "__main__":
    main()
