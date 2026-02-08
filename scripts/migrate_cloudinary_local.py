import argparse
import os
from urllib.parse import urlparse, parse_qs

import pymysql

try:
    import cloudinary
    import cloudinary.uploader
except Exception:
    cloudinary = None


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
        return pymysql.connect(
            host=cfg["host"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            port=cfg["port"],
            charset="utf8mb4",
            autocommit=False,
        )
    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    database = os.getenv("DB_NAME")
    port = int(os.getenv("DB_PORT", "3306"))
    if not host or not user or not database:
        raise RuntimeError("DB_HOST/DB_USER/DB_NAME (or DATABASE_URL) is required.")
    return pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port,
        charset="utf8mb4",
        autocommit=False,
    )


def cloudinary_ready() -> bool:
    return bool((os.getenv("CLOUDINARY_URL") or os.getenv("CLOUDINARY_CLOUD_NAME")) and cloudinary)


def cloudinary_config():
    if not cloudinary_ready():
        raise RuntimeError("Cloudinary is not configured.")
    if os.getenv("CLOUDINARY_URL"):
        cloudinary.config(secure=True)
    else:
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True,
        )


def resolve_local_path(path_value: str, images_root: str) -> str | None:
    if not path_value:
        return None
    if path_value.startswith(("http://", "https://")):
        return None
    normalized = path_value.lstrip("/").replace("\\", "/")
    if normalized.startswith("static/"):
        normalized = normalized[len("static/") :]
    candidate = os.path.join(images_root, os.path.basename(normalized))
    if os.path.isfile(candidate):
        return candidate
    # try full relative path
    candidate = os.path.join(images_root, normalized)
    if os.path.isfile(candidate):
        return candidate
    return None


def upload_file(path: str, folder: str, public_id: str) -> str | None:
    try:
        result = cloudinary.uploader.upload(
            path,
            folder=folder,
            public_id=public_id,
            resource_type="image",
            overwrite=True,
            unique_filename=False,
        )
        return result.get("secure_url") or result.get("url")
    except Exception:
        return None


def migrate_table(
    conn,
    table: str,
    id_col: str,
    url_col: str,
    folder: str,
    images_root: str,
    limit: int | None,
    dry_run: bool,
):
    processed = 0
    copied = 0
    skipped = 0
    errors = 0
    with conn.cursor() as cur:
        cur.execute(f"SELECT {id_col}, {url_col} FROM {table}")
        rows = cur.fetchall() or []
        for row in rows:
            if limit and processed >= limit:
                break
            row_id = row[0]
            url = row[1] or ""
            if url.startswith(("http://", "https://")):
                skipped += 1
                processed += 1
                continue
            local_path = resolve_local_path(str(url), images_root)
            if not local_path:
                errors += 1
                processed += 1
                continue
            new_url = upload_file(local_path, folder, str(row_id))
            if not new_url:
                errors += 1
                processed += 1
                continue
            if not dry_run:
                cur.execute(
                    f"UPDATE {table} SET {url_col}=%s WHERE {id_col}=%s",
                    (new_url, row_id),
                )
            copied += 1
            processed += 1
    if not dry_run:
        conn.commit()
    return {"processed": processed, "copied": copied, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Migrate local product images to Cloudinary and update DB.")
    parser.add_argument("--images-root", default="static/images", help="Local images folder")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows per table (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true", help="Do not update database")
    args = parser.parse_args()

    if not cloudinary_ready():
        raise SystemExit("Cloudinary is not configured. Set CLOUDINARY_URL or CLOUDINARY_* env vars.")
    cloudinary_config()

    images_root = args.images_root
    if not os.path.isdir(images_root):
        raise SystemExit(f"Images root not found: {images_root}")

    conn = get_db_connection()
    try:
        limit = args.limit if args.limit > 0 else None
        prod = migrate_table(
            conn,
            table="products",
            id_col="product_id",
            url_col="image_url",
            folder="bigoh/products",
            images_root=images_root,
            limit=limit,
            dry_run=args.dry_run,
        )
        rev = migrate_table(
            conn,
            table="product_reviews",
            id_col="id",
            url_col="review_photo",
            folder="bigoh/reviews",
            images_root=images_root.replace("images", "review_photos"),
            limit=limit,
            dry_run=args.dry_run,
        )
    finally:
        conn.close()

    print("Products:", prod)
    print("Reviews:", rev)


if __name__ == "__main__":
    main()
