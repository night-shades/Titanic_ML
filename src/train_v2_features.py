import os
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


def extract_title(name: str) -> str:
    """
    從乘客名字欄位 Name 抽出稱謂 Title。
    Titanic 的 Name 格式通常像：
        "Braund, Mr. Owen Harris"
        "Cumings, Mrs. John Bradley (Florence Briggs Thayer)"
    我們用正則找出 'Mr', 'Mrs', 'Miss' 這類稱謂。

    如果抓不到（極少數），回傳 'Unknown'。
    """
    if pd.isna(name):
        return "Unknown"

    # 以正則抓出 ", <Title>." 之間的 Title
    m = re.search(r",\s*([A-Za-z]+)\.", name)
    if not m:
        return "Unknown"
    return m.group(1)


def normalize_title(title: str) -> str:
    """
    把稀有稱謂合併成 'Rare'，避免 one-hot 後太稀疏、且提升泛化能力。
    這是一種常見的「類別合併」技巧。
    """
    common = {"Mr", "Mrs", "Miss", "Master"}
    if title in common:
        return title
    return "Rare"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    對資料表新增特徵工程欄位：
    - FamilySize = SibSp + Parch + 1
      （+1 是把自己也算進家庭人數）
    - IsAlone: 是否獨自一人（FamilySize==1）
    - Title: 從 Name 抽出的稱謂（再做 normalize）
    注意：這些特徵只使用資料本身，不使用 Survived，避免資料洩漏。
    """
    out = df.copy()

    # 家庭規模：同船兄弟姊妹/配偶 + 父母/子女 + 自己
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1

    # 是否獨自：家庭規模等於 1 表示沒家人同行
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # 從 Name 抽出稱謂，再把稀有稱謂合併
    out["Title"] = out["Name"].apply(extract_title).apply(normalize_title)

    return out


def main():
    # =========================
    # 1) 取得專案根目錄
    # =========================
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(project_root, "titanic", "train.csv")
    test_path = os.path.join(project_root, "titanic", "test.csv")

    # =========================
    # 2) 讀取資料
    # =========================
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # =========================
    # 3) 特徵工程（對 train/test 都做一樣處理）
    # =========================
    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    # =========================
    # 4) 準備標籤與特徵
    # =========================
    y = train_df["Survived"]

    # baseline 欄位 + 新增特徵工程欄位
    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
        "FamilySize", "IsAlone", "Title"
    ]
    X = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # =========================
    # 5) 切訓練/驗證集
    # stratify=y：保持生存/死亡比例一致
    # =========================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # 6) 前處理
    # - 數值：中位數補缺
    # - 類別：眾數補缺 + One-Hot
    #
    # 注意：
    # 這些「補值規則 / One-Hot 類別集合」只會從 X_train 學到
    # 再套用到 X_val / X_test，這就是 Pipeline 的價值：避免資料洩漏
    # =========================
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    categorical_features = ["Sex", "Embarked", "Title"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

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

    # =========================
    # 7) 模型
    # Logistic Regression：
    # - 基線強、訓練快、可解釋
    # - 對 one-hot 後的高維稀疏特徵也很常用
    # =========================
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    # =========================
    # 8) 訓練 + 驗證
    # =========================
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, val_pred)
    print(f"[Validation] Accuracy = {acc:.4f}")
    print(classification_report(y_val, val_pred, digits=4))

    # =========================
    # 9) 產生提交檔（Kaggle 格式）
    # =========================
    test_pred = clf.predict(X_test)
    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": test_pred.astype(int)}
    )

    os.makedirs(os.path.join(project_root, "reports"), exist_ok=True)
    submission_path = os.path.join(project_root, "reports", "submission_v2.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission file to: {submission_path}")

    # =========================
    # 10) 存模型
    # =========================
    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    model_path = os.path.join(project_root, "models", "model_v2_features.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
