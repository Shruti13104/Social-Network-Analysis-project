import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "influencer_marketing.csv"
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nCleaned columns:")
print(df.columns.tolist())


# ============================================================
# 3. DATA CLEANING
# ============================================================

df = df.drop_duplicates()

df = df.dropna(
    subset=[
        "platform",
        "influencer_category",
        "campaign_type",
        "engagements",
        "estimated_reach",
        "product_sales",
        "campaign_duration_days"
    ]
)

# Convert numerical columns
numeric_columns = [
    "engagements",
    "estimated_reach",
    "product_sales",
    "campaign_duration_days"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=numeric_columns
)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# Engagement rate relative to reach
df["engagement_rate"] = (
    df["engagements"] /
    df["estimated_reach"].replace(0, 1)
) * 100

# Sales efficiency relative to reach
df["sales_per_reach"] = (
    df["product_sales"] /
    df["estimated_reach"].replace(0, 1)
)

print("\nAfter cleaning:")
print(df.shape)


# ============================================================
# 5. CREATE MODEL FOLDER
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 6. FEATURES
# ============================================================

features = [
    "platform",
    "influencer_category",
    "campaign_type",
    "campaign_duration_days"
]

X = df[features]


# ============================================================
# 7. PREPROCESSING
# ============================================================

categorical_features = [
    "platform",
    "influencer_category",
    "campaign_type"
]

numerical_features = [
    "campaign_duration_days"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 8. FUNCTION TO TRAIN MODEL
# ============================================================

def train_model(target, model_name):

    print("\n================================")
    print("Training:", model_name)
    print("Target:", target)
    print("================================")

    X_model = X
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X_model,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    model_path = (
        f"models/{model_name}.pkl"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    print(
        "Saved:",
        model_path
    )

    return pipeline


# ============================================================
# 9. TRAIN THREE MODELS
# ============================================================

engagement_model = train_model(
    "engagements",
    "engagement_model"
)

reach_model = train_model(
    "estimated_reach",
    "reach_model"
)

sales_model = train_model(
    "product_sales",
    "sales_model"
)


# ============================================================
# 10. SAVE DATA FOR RECOMMENDATION
# ============================================================

df.to_csv(
    "models/cleaned_data.csv",
    index=False
)


# ============================================================
# 11. SAVE METADATA
# ============================================================

metadata = {
    "features": features,
    "platforms": sorted(
        df["platform"].unique().tolist()
    ),
    "influencer_categories": sorted(
        df["influencer_category"].unique().tolist()
    ),
    "campaign_types": sorted(
        df["campaign_type"].unique().tolist()
    )
}

joblib.dump(
    metadata,
    "models/metadata.pkl"
)


print("\n================================")
print("ALL MODELS TRAINED SUCCESSFULLY")
print("================================")

print("\nCreated files:")

print("models/engagement_model.pkl")
print("models/reach_model.pkl")
print("models/sales_model.pkl")
print("models/cleaned_data.csv")
print("models/metadata.pkl")