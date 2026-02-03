import os
import sys
from urllib.parse import urlparse, parse_qs

import pymysql


def _parse_db_url(db_url: str) -> dict:
    parsed = urlparse(db_url)
    if parsed.scheme not in {"mysql", "mariadb"}:
        raise ValueError("Unsupported database URL scheme")
    database = parsed.path.lstrip("/")
    query = parse_qs(parsed.query)
    return {
        "host": parsed.hostname,
        "user": parsed.username,
        "password": parsed.password,
        "database": database,
        "port": parsed.port or 3306,
        "query": query,
    }


def get_db_connection():
    db_url = os.getenv("DATABASE_URL") or os.getenv("MYSQL_URL") or os.getenv("DB_URL")
    if db_url:
        cfg = _parse_db_url(db_url)
        host = cfg["host"]
        user = cfg["user"]
        password = cfg["password"]
        database = cfg["database"]
        port = int(cfg["port"])
        query = cfg["query"]
    else:
        host = os.getenv("DB_HOST")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        database = os.getenv("DB_NAME")
        port = int(os.getenv("DB_PORT", "3306"))
        query = {}

    if not host:
        raise RuntimeError("Database host is not set (DB_HOST or DATABASE_URL).")

    ssl_disabled = os.getenv("DB_SSL_DISABLED", "0") == "1"
    sslmode = (query.get("sslmode") or [""])[0].lower()
    ssl_query = (query.get("ssl") or [""])[0].lower()
    if sslmode == "disable" or ssl_query in {"0", "false", "no"}:
        ssl_disabled = True

    connect_kwargs = dict(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port,
        connect_timeout=10,
        read_timeout=10,
        write_timeout=10,
    )
    if not ssl_disabled:
        connect_kwargs["ssl"] = {"ssl": {}}
    return pymysql.connect(**connect_kwargs)


def ensure_products_schema(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'products'
              AND COLUMN_NAME = 'seller'
            """
        )
        row = cur.fetchone()
        has_col = bool(row and row[0] > 0)
        if not has_col:
            cur.execute("ALTER TABLE products ADD COLUMN seller VARCHAR(120) NULL")
    conn.commit()


def main():
    seller = "Bigoh Official"
    if len(sys.argv) > 1 and sys.argv[1].strip():
        seller = sys.argv[1].strip()

    conn = get_db_connection()
    try:
        ensure_products_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE products
                SET seller = %s
                WHERE seller IS NULL OR seller = ''
                """,
                (seller,),
            )
            updated = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    print(f"Backfilled seller for {updated} product(s).")


if __name__ == "__main__":
    main()
