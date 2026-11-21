"""Delete the database and re-seed it with slots for the next 7 days.

    python -m app.reset_db
"""
import os

from dotenv import load_dotenv

from .db import DEFAULT_DB_URL, init_db, make_engine

if __name__ == "__main__":
    load_dotenv()
    url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
        if os.path.exists(path):
            os.remove(path)
    init_db(make_engine(url))
    print(f"Database reset and seeded: {url}")
