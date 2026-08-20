import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def preprocess_data(train_df, test_df):
    cat_cols = [
        "Gender",
        "Ever_Married",
        "Graduated",
        "Profession",
        "Spending_Score",
        "Var_1",
    ]
    num_cols = ["Age", "Work_Experience", "Family_Size"]

    # Impute missing values
    for col in cat_cols:
        train_df[col] = train_df[col].fillna("Missing")
        test_df[col] = test_df[col].fillna("Missing")

    for col in num_cols:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(train_df[col].median())

    # Categorical encoding
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[cat_cols] = oe.fit_transform(train_df[cat_cols])
    test_df[cat_cols] = oe.transform(test_df[cat_cols])

    # Feature Engineering
    for df in [train_df, test_df]:
        df["Age_Group"] = pd.cut(
            df["Age"], bins=[0, 25, 40, 60, 100], labels=[0, 1, 2, 3]
        ).astype(float)
        df["Family_Size_Large"] = (df["Family_Size"] > 4).astype(int)
        df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)

    features = (
        cat_cols + num_cols + ["Age_Group", "Family_Size_Large", "Is_Alone"]
    )
    return train_df, test_df, features


def main():
    os.makedirs("output", exist_ok=True)

    train = pd.read_csv("data/Train.csv")
    test = pd.read_csv("data/Test.csv")

    train_proc, test_proc, features = preprocess_data(train, test)

    le = LabelEncoder()
    y_train = le.fit_transform(train_proc["Segmentation"])
    X_train = train_proc[features]
    X_test = test_proc[features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train_proc), 4))
    test_preds = np.zeros((len(test_proc), 4))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

        model = HistGradientBoostingClassifier(
            random_state=42, max_iter=200, learning_rate=0.05, max_depth=6
        )
        model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(X_test) / skf.n_splits

    oof_classes = np.argmax(oof_preds, axis=1)
    print(f"5-Fold CV Accuracy: {accuracy_score(y_train, oof_classes):.4f}")
    print(
        f"5-Fold CV Macro F1: {f1_score(y_train, oof_classes, average='macro'):.4f}"
    )

    final_classes = np.argmax(test_preds, axis=1)
    submission = pd.DataFrame(
        {"ID": test["ID"], "Segmentation": le.inverse_transform(final_classes)}
    )

    submission.to_csv("output/final_submission.csv", index=False)
    print(
        "Successfully saved final predictions to output/final_submission.csv"
    )


if __name__ == "__main__":
    main()