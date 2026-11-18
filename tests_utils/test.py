import pandas as pd
from sqlalchemy import create_engine

POSTGRES_URI = "postgresql://dst_admin:dst_password123@localhost:5432/dst_db"
engine = create_engine(POSTGRES_URI)

query = "SELECT * FROM gold_ml_features;"

df = pd.read_sql(query, engine)
df.head(5)

