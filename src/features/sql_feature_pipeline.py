import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://localhost/financial_risk_db"

engine = create_engine(DATABASE_URL)


FEATURE_QUERY = """

SELECT

    customer_id,

    COUNT(*) AS transaction_count,

    AVG(amount) AS avg_transaction_amount,

    MAX(amount) AS max_transaction_amount,

    MIN(amount) AS min_transaction_amount,

    SUM(amount) AS total_transaction_amount,

    COUNT(CASE WHEN transaction_type = 'Credit'
        THEN 1 END) AS total_credit_transactions,

    COUNT(CASE WHEN transaction_type = 'Debit'
        THEN 1 END) AS total_debit_transactions,

    COUNT(CASE WHEN is_fraud = 1
        THEN 1 END) AS fraud_transactions,

    ROUND(
        COUNT(CASE WHEN is_fraud = 1 THEN 1 END)::numeric
        / COUNT(*)::numeric,
        4
    ) AS fraud_ratio

FROM transactions

GROUP BY customer_id

"""


def build_feature_store():

    print("Running SQL analytics pipeline")

    df = pd.read_sql(FEATURE_QUERY, engine)

    print("Feature store created")

    print(df.head())

    return df


if __name__ == "__main__":

    features_df = build_feature_store()

    features_df.to_csv(
        "data/processed/customer_features.csv",
        index=False
    )

    print("Feature dataset exported")