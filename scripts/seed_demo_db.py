#!/usr/bin/env python3
"""Create a local SQLite demo database for trying the agent without Postgres/MySQL."""

from pathlib import Path

from sqlalchemy import create_engine, text

DB_PATH = Path(__file__).resolve().parent.parent / "demo.db"


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE customers (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    country TEXT,
                    created_at TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    customer_id INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL,
                    ordered_at TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO customers (id, name, email, country, created_at) VALUES
                (1, 'Ada Lovelace', 'ada@example.com', 'UK', '2024-01-10'),
                (2, 'Alan Turing', 'alan@example.com', 'UK', '2024-02-15'),
                (3, 'Grace Hopper', 'grace@example.com', 'US', '2024-03-01'),
                (4, 'Edsger Dijkstra', 'edsger@example.com', 'NL', '2024-03-20')
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO orders (id, customer_id, amount, status, ordered_at) VALUES
                (1, 1, 120.50, 'paid', '2024-04-01'),
                (2, 1, 40.00, 'refunded', '2024-04-10'),
                (3, 2, 999.99, 'paid', '2024-05-01'),
                (4, 3, 15.25, 'pending', '2024-05-12'),
                (5, 3, 220.00, 'paid', '2024-06-01'),
                (6, 4, 50.00, 'paid', '2024-06-15')
                """
            )
        )

    print(f"Demo database ready: {DB_PATH}")
    print("Example:")
    print(
        f'  python main.py ask "total paid revenue by country" '
        f'--sql-url "sqlite:///{DB_PATH}" --sql-user demo --sql-password ""'
    )


if __name__ == "__main__":
    main()
