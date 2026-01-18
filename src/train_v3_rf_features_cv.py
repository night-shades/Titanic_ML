import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib


# -----------------------------
# 1) 名字稱謂：Title
# -----------------------------
def extract_title(name: str) -> str:
    """
    從 Name 欄位擷取稱謂，例如 Mr/Mrs/Miss/Master。
    """
    if pd.isna(name):
        return "Unknown"
    m = re.search(r",\s*([A-Za-z]+)\.", name)
    return m.group(1) if m else "Unknown"


def normalize_title(title: str) -> str:
    """
    合併稀有稱謂，避免 one-hot 後太稀疏、過擬合。
    """
    common = {"Mr", "Mrs", "Miss", "Master"}
    return title if title in common else "Rare"


# -----------------------------
# 2) Cabin 甲板：Deck
# -----------------------------
def extract_deck(cabin: str) -> str:
    """
    Cabin 通常像 'C85'，取第一個字母當 Deck（甲板）。
    缺失很多，所以缺失記為 'Unknown'。
    """
    if pd.isna(cabin) or cabin == "":
        return "Unknown"
    return str(cabin)[0]


# -----------------------------
# 3) 特徵工程
# -----------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 家庭規模與是否獨自
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # Title
    out["Title"] = out["Name"].apply(extract_title).apply(normalize_title)

    # Deck
    out["Deck"] = out["Cabin"].apply(extract_deck)

    # 同票號群組大小：TicketGroupSize
    # 直覺：同一張票/同一組人可能一起行動，影響生存率
    # 注意：要用整個資料算 count，再回填每一列
    out["TicketGroupSize"] = out.groupby("Ticket")["Ticket"].transform("count")

    # Fare 每人平均（同票號可能多人）
    out["FarePerPerson"] = out["Fare"] / out["TicketGroupSize"]

    # Age、Fare 做分箱（讓樹模型更容易用門檻切）
    # qcut 依分位數分箱，對偏態分佈通常更穩
    # 缺失先留著，後面會由 SimpleImputer 補
    out["AgeBin"] = pd.qcut(out["Age"], q=5, duplicates="drop")
    out["FareBin"] = pd.qcut(out["Fare"], q=5, duplicates="drop")

    return out


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(project_root, "titanic", "train.csv")
    test_path = os.path.join(project_root, "titanic", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 特徵工程（train/test 都做）
    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    y = train_df["Survived"]

    # 這一版只專注在「常見有效」的一組特徵
    feature_cols = [
        # 原始（基礎強特徵）
        "Pclass", "Sex", "Embarked",
        # 連續型
        "Age", "Fare", "FarePerPerson",
        # 家庭與群組
        "SibSp", "Parch", "FamilySize", "IsAlone", "TicketGroupSize",
        # 從文字抽出來的訊號
        "Title", "Deck",
        # 分箱（類別）
        "AgeBin", "FareBin",
    ]

    X = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # 類別 vs 數值欄位
    numeric_features = [
        "Age", "Fare", "FarePerPerson", "SibSp", "Parch", "FamilySize", "IsAlone", "TicketGroupSize"
    ]
    categorical_features = [
        "Pclass", "Sex", "Embarked", "Title", "Deck", "AgeBin", "FareBin"
    ]

    # 數值：中位數補缺
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    # 類別：眾數補缺 + one-hot
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 樹模型：能吃交互、非線性，Titanic 常比線性模型更有利
    rf = RandomForestClassifier(
        n_estimators=600,        # 樹的數量：多一點較穩
        random_state=42,
        min_samples_leaf=2,      # 降低過擬合
        max_features="sqrt",     # 常見穩健設定
        n_jobs=-1
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", rf),
        ]
    )

    # -----------------------------
    # 5-fold 交叉驗證：比單次切分可信
    # -----------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"[CV] Accuracy mean = {cv_scores.mean():.4f}, std = {cv_scores.std():.4f}")
    print("CV scores:", np.round(cv_scores, 4))

    # -----------------------------
    # 最終：用全 train 重新訓練，再對 test 產生 submission
    # -----------------------------
    clf.fit(X, y)

    test_pred = clf.predict(X_test)
    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": test_pred.astype(int)}
    )

    os.makedirs(os.path.join(project_root, "reports"), exist_ok=True)
    sub_path = os.path.join(project_root, "reports", "submission_v3.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission file to: {sub_path}")

    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    model_path = os.path.join(project_root, "models", "model_v3_rf.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
