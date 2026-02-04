import os
import re
import sys
from urllib.parse import urlparse, parse_qs

import pymysql


COLOR_PATTERNS = [
    ("Rose Gold", [r"rose\s*gold", r"ros[eé]\s*gold"]),
    ("Navy Blue", [r"navy\s*blue", r"navy"]),
    ("Gold", [r"\bgold\b"]),
    ("Silver", [r"\bsilver\b"]),
    ("Black", [r"\bblack\b", r"\bnoir\b"]),
    ("White", [r"\bwhite\b", r"\bivory\b"]),
    ("Gray", [r"\bgray\b", r"\bgrey\b"]),
    ("Blue", [r"\bblue\b"]),
    ("Red", [r"\bred\b"]),
    ("Green", [r"\bgreen\b"]),
    ("Yellow", [r"\byellow\b"]),
    ("Orange", [r"\borange\b"]),
    ("Brown", [r"\bbrown\b"]),
    ("Pink", [r"\bpink\b"]),
    ("Purple", [r"\bpurple\b", r"\bviolet\b"]),
    ("Beige", [r"\bbeige\b", r"\bcream\b"]),
    ("Transparent", [r"\btransparent\b", r"\bclear\b"]),
    ("Multi", [r"\bmulti(?:-|\s)?color\b", r"\bmulticolor\b", r"\bmixed\b"]),
]


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


def ensure_color_column(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'products'
              AND COLUMN_NAME = 'color'
            """
        )
        row = cur.fetchone()
        has_col = bool(row and row[0] > 0)
        if not has_col:
            cur.execute("ALTER TABLE products ADD COLUMN color VARCHAR(80) NULL")
    conn.commit()


def infer_color(text: str) -> str:
    text = (text or "").lower()
    for label, patterns in COLOR_PATTERNS:
        for pat in patterns:
            if re.search(pat, text):
                return label
    return ""


def main():
    conn = get_db_connection()
    try:
        ensure_color_column(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT product_id, product_name, description, brand
                FROM products
                WHERE color IS NULL OR color = ''
                """
            )
            rows = cur.fetchall() or []

            updated = 0
            for row in rows:
                product_id = row[0]
                name = row[1] or ""
                desc = row[2] or ""
                brand = row[3] or ""
                text = f"{name} {desc} {brand}"
                color = infer_color(text)
                if not color:
                    continue
                cur.execute(
                    "UPDATE products SET color=%s WHERE product_id=%s",
                    (color, product_id),
                )
                updated += 1

        conn.commit()
    finally:
        conn.close()

    print(f"Inferred color for {updated} product(s).")


if __name__ == "__main__":
    main()
