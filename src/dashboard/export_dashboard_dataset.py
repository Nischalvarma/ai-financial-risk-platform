import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://localhost/financial_risk_db"

engine = create_engine(DATABASE_URL)


QUERY = """

SELECT

    transaction_id,

    customer_id,

    transaction_date,

    amount,

    merchant_category,

    transaction_type,

    location,

    is_fraud

FROM transactions

"""


def export_dashboard_dataset():

    print("Exporting dashboard dataset")

    df = pd.read_sql(QUERY, engine)

    df.to_csv(
        "data/processed/dashboard_dataset.csv",
        index=False
    )

    print("Dashboard dataset exported")


if __name__ == "__main__":

    export_dashboard_dataset()