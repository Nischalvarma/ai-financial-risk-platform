import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import random

DATABASE_URL = "postgresql://localhost/financial_risk_db"

engine = create_engine(DATABASE_URL)

NUM_ROWS = 5000

merchant_categories = [
    "Retail",
    "Food",
    "Travel",
    "Electronics",
    "Healthcare",
    "Entertainment"
]

transaction_types = [
    "Credit",
    "Debit",
    "Transfer"
]

locations = [
    "Mumbai",
    "Delhi",
    "Hyderabad",
    "Bangalore",
    "Chennai"
]


def generate_transactions(n_rows):

    data = []

    start_date = datetime(2022, 1, 1)

    for i in range(n_rows):

        transaction = {

            "customer_id":
                random.randint(1000, 5000),

            "transaction_date":
                start_date + timedelta(
                    days=random.randint(0, 730)
                ),

            "amount":
                round(random.uniform(10, 5000), 2),

            "merchant_category":
                random.choice(merchant_categories),

            "transaction_type":
                random.choice(transaction_types),

            "location":
                random.choice(locations),

            "is_fraud":
                np.random.choice(
                    [0, 1],
                    p=[0.97, 0.03]
                )
        }

        data.append(transaction)

    return pd.DataFrame(data)


def load_to_database(df):

    df.to_sql(
        "transactions",
        engine,
        if_exists="append",
        index=False
    )

    print("Data successfully loaded into PostgreSQL")


if __name__ == "__main__":

    print("Generating synthetic transactions")

    transactions_df = generate_transactions(NUM_ROWS)

    print("Loading data into database")

    load_to_database(transactions_df)

    print("Pipeline execution completed")