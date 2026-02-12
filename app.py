from flask import *
from werkzeug.exceptions import HTTPException
from urllib.parse import quote, urlparse, parse_qs
import re
import os
import json
import uuid
import shutil
import base64
import io
import hashlib
import shlex
from datetime import timedelta, datetime
from functools import wraps
import secrets
import urllib.request
import urllib.error
import hmac
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
from dotenv import load_dotenv
try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None
try:
    import qrcode
except Exception:
    qrcode = None
try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
except Exception:
    cloudinary = None

load_dotenv()
import mailer


app = Flask(__name__)


app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY is required.")
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER", "+254752370545")
ADMIN_USERS = {
    name.strip().lower()
    for name in os.getenv("ADMIN_USERS", "").split(",")
    if name.strip()
}

from werkzeug.middleware.proxy_fix import ProxyFix

BUSINESS_NAME = os.getenv("BUSINESS_NAME", "Bigoh")
BUSINESS_ADDRESS = os.getenv("BUSINESS_ADDRESS", "Donholm Caltex, Nairobi")
BUSINESS_REG_NO = os.getenv("BUSINESS_REG_NO", "")
BUSINESS_REG_BODY = os.getenv("BUSINESS_REG_BODY", "")
BUSINESS_LOGO = os.getenv("BUSINESS_LOGO", "images/logo.jpeg")
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "riggie0012@gmail.com")
SUPPORT_EMAIL_ADMIN = os.getenv("SUPPORT_EMAIL_ADMIN", "junioronunga8@gmail.com")
SUPPORT_PHONE = os.getenv("SUPPORT_PHONE", "0759 808 915")
SUPPORT_WHATSAPP = os.getenv("SUPPORT_WHATSAPP", "254759808915")
SUPPORT_HOURS = os.getenv("SUPPORT_HOURS", "Daily 8:00am - 8:00pm EAT")
PAYMENT_DETAILS_TITLE = os.getenv("PAYMENT_DETAILS_TITLE", "Payment Details")
PAYMENT_DETAILS_LINES = os.getenv("PAYMENT_DETAILS_LINES", "").strip()
STATIC_CDN_BASE = os.getenv("STATIC_CDN_BASE", "").strip()
USE_CLOUDINARY = bool((os.getenv("CLOUDINARY_URL") or os.getenv("CLOUDINARY_CLOUD_NAME")) and cloudinary)
LOYALTY_ENABLED = os.getenv("LOYALTY_ENABLED", "1") == "1"
LOYALTY_REPEAT_ORDERS_MIN = int(os.getenv("LOYALTY_REPEAT_ORDERS_MIN", "2"))
LOYALTY_REPEAT_DISCOUNT_PCT = float(os.getenv("LOYALTY_REPEAT_DISCOUNT_PCT", "5"))
REFERRAL_ENABLED = os.getenv("REFERRAL_ENABLED", "1") == "1"
REFERRAL_COUPON_EXPIRES_DAYS = int(os.getenv("REFERRAL_COUPON_EXPIRES_DAYS", "30"))
REFERRAL_CODE_LEN = int(os.getenv("REFERRAL_CODE_LEN", "8"))
COUPONS_PER_SIGNUP = int(os.getenv("COUPONS_PER_SIGNUP", "5"))
COUPONS_PER_REFERRAL = int(os.getenv("COUPONS_PER_REFERRAL", "5"))
COUPON_UNIT_AMOUNT = float(os.getenv("COUPON_UNIT_AMOUNT", "100"))
PAYMENT_METHODS = [
    {"label": "M-Pesa", "icon": "fa-solid fa-money-bill-wave"},
    {"label": "Visa", "icon": "fa-brands fa-cc-visa"},
    {"label": "Mastercard", "icon": "fa-brands fa-cc-mastercard"},
    {"label": "Airtel Money", "icon": "fa-solid fa-sim-card"},
    {"label": "T-Kash", "icon": "fa-solid fa-mobile-screen"},
    {"label": "PayPal", "icon": "fa-brands fa-paypal"},
    {"label": "Apple Pay", "icon": "fa-brands fa-apple-pay"},
    {"label": "Google Pay", "icon": "fa-brands fa-google-pay"},
]
PAYMENT_LOGOS = [
    {"label": "M-Pesa", "image": "images/logo_mpesa.webp"},
    {"label": "Airtel Money", "image": "images/loge_airtel.webp"},
    {"label": "Visa", "image": "images/logo_visa.webp"},
    {"label": "Mastercard", "image": "images/logo_mastercard.webp"},
    {"label": "Google Pay", "image": "images/google-pay.png"},
    {"label": "SSL Secure", "image": "images/SSL.webp"},
    {"label": "McAfee Secure", "image": "images/McAfee.webp"},
]
LOW_STOCK_THRESHOLD = int(os.getenv("LOW_STOCK_THRESHOLD", "5"))
LOW_STOCK_ALERT_INTERVAL_HOURS = int(os.getenv("LOW_STOCK_ALERT_INTERVAL_HOURS", "24"))
LOW_STOCK_EMAIL_TO = os.getenv("LOW_STOCK_EMAIL_TO", "") or SUPPORT_EMAIL_ADMIN
ACTIVE_SESSION_WINDOW_MINUTES = int(os.getenv("ACTIVE_SESSION_WINDOW_MINUTES", "5"))
WHATSAPP_ALERTS_ENABLED = os.getenv("WHATSAPP_ALERTS_ENABLED", "0") == "1"
WHATSAPP_ALERT_WEBHOOK = os.getenv("WHATSAPP_ALERT_WEBHOOK", "").strip()
WHATSAPP_ALERT_TOKEN = os.getenv("WHATSAPP_ALERT_TOKEN", "").strip()
WHATSAPP_ALERT_TO = os.getenv("WHATSAPP_ALERT_TO", "").strip() or SUPPORT_WHATSAPP
WHATSAPP_RECEIPTS_ENABLED = os.getenv("WHATSAPP_RECEIPTS_ENABLED", "0") == "1"
WHATSAPP_STATUS_UPDATES_ENABLED = os.getenv("WHATSAPP_STATUS_UPDATES_ENABLED", "1") == "1"
CRON_SECRET = os.getenv("CRON_SECRET", "").strip()
ABANDONED_CART_HOURS = int(os.getenv("ABANDONED_CART_HOURS", "4"))
REVIEW_REQUEST_DELAY_HOURS = int(os.getenv("REVIEW_REQUEST_DELAY_HOURS", "24"))
STALE_PENDING_HOURS = int(os.getenv("STALE_PENDING_HOURS", "48"))
DAILY_SUMMARY_HOUR = int(os.getenv("DAILY_SUMMARY_HOUR", "18"))
REVIEW_AUTO_APPROVE = os.getenv("REVIEW_AUTO_APPROVE", "0") == "1"
REVIEW_TRUSTED_USERS = {
    u.strip().lower()
    for u in os.getenv("REVIEW_TRUSTED_USERS", "").split(",")
    if u.strip()
}
REVIEW_MAX_IMAGE_BYTES = int(os.getenv("REVIEW_MAX_IMAGE_BYTES", "4000000"))
REVIEW_MAX_IMAGE_PX = int(os.getenv("REVIEW_MAX_IMAGE_PX", "1600"))
REVIEW_IMAGE_QUALITY = int(os.getenv("REVIEW_IMAGE_QUALITY", "80"))
PRODUCT_MAX_IMAGE_BYTES = int(os.getenv("PRODUCT_MAX_IMAGE_BYTES", "2500000"))
PRODUCT_MAX_IMAGE_PX = int(os.getenv("PRODUCT_MAX_IMAGE_PX", "1600"))
PRODUCT_IMAGE_QUALITY = int(os.getenv("PRODUCT_IMAGE_QUALITY", "82"))


def _safe_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


DB_CONNECT_TIMEOUT = _safe_float_env("DB_CONNECT_TIMEOUT", 4.0)
DB_READ_TIMEOUT = _safe_float_env("DB_READ_TIMEOUT", 8.0)
DB_WRITE_TIMEOUT = _safe_float_env("DB_WRITE_TIMEOUT", 8.0)
DB_FAILURE_BACKOFF_SECONDS = _safe_float_env("DB_FAILURE_BACKOFF_SECONDS", 20.0)
BRAND_PARTNERS = [
    p.strip()
    for p in os.getenv("BRAND_PARTNERS", "").split(",")
    if p.strip()
]
DEFAULT_MANAGED_CATEGORIES = [
    "Shoes",
    "Ladies Watch",
    "Men Watch",
    "Jersey",
    "Cleaning",
    "Electricals",
]
MANAGED_CATEGORIES = [
    name.strip()
    for name in os.getenv("MANAGED_CATEGORIES", ",".join(DEFAULT_MANAGED_CATEGORIES)).split(",")
    if name.strip()
]

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

if USE_CLOUDINARY:
    if os.getenv("CLOUDINARY_URL"):
        cloudinary.config(secure=True)
    else:
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True,
        )

LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "app.log")
handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3)
handler.setLevel(LOG_LEVEL)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
app.logger.addHandler(handler)
app.logger.setLevel(LOG_LEVEL)

RATE_LIMITS = {
    "signin": (8, 60),
    "signup": (6, 60),
    "add_to_cart": (25, 60),
    "pay_on_delivery": (5, 60),
    "add_product_review": (3, 60),
}
_rate_store = {}


DEFAULT_UPLOAD_ROOT = "/data/uploads" if os.path.isdir("/data/uploads") else "static"
UPLOAD_ROOT = os.getenv("UPLOAD_ROOT", DEFAULT_UPLOAD_ROOT).strip() or DEFAULT_UPLOAD_ROOT
UPLOAD_FOLDER = os.path.join(UPLOAD_ROOT, "images")
REVIEW_UPLOAD_FOLDER = os.path.join(UPLOAD_ROOT, "review_photos")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["UPLOAD_ROOT"] = UPLOAD_ROOT
app.config["REVIEW_UPLOAD_FOLDER"] = REVIEW_UPLOAD_FOLDER
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("FLASK_SESSION_SECURE", "0") == "1"
remember_days_env = os.getenv("REMEMBER_ME_DAYS", "30")
try:
    remember_days = int(remember_days_env)
except ValueError:
    remember_days = 30
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=remember_days)

oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

def cdn_url_for(endpoint, **values):
    if endpoint == "static" and STATIC_CDN_BASE:
        filename = values.get("filename", "")
        return f"{STATIC_CDN_BASE.rstrip('/')}/{filename.lstrip('/')}"
    return url_for(endpoint, **values)

app.jinja_env.globals["url_for"] = cdn_url_for


def generate_csrf_token():
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


app.jinja_env.globals["csrf_token"] = generate_csrf_token


def _client_ip():
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_limit_exceeded():
    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or "application/json" in request.headers.get("Accept", ""):
        return jsonify(ok=False, message="Too many requests. Please slow down.", level="warning"), 429
    set_site_message("Too many requests. Please try again shortly.", "warning")
    return redirect(request.referrer or url_for("home")), 429


def rate_limit(key: str):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
                return view(*args, **kwargs)
            limit, window = RATE_LIMITS.get(key, (10, 60))
            now = time.time()
            ip = _client_ip()
            bucket_key = f"{key}:{ip}"
            bucket = _rate_store.get(bucket_key, [])
            bucket = [t for t in bucket if now - t < window]
            if len(bucket) >= limit:
                _rate_store[bucket_key] = bucket
                return _rate_limit_exceeded()
            bucket.append(now)
            _rate_store[bucket_key] = bucket
            return view(*args, **kwargs)
        return wrapped
    return decorator

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _normalize_user_name(value) -> str:
    return str(value or "").strip().lower()


def _is_trusted_review_user(name: str) -> bool:
    return _normalize_user_name(name) in REVIEW_TRUSTED_USERS


def compress_product_image(path: str) -> None:
    compress_image(
        path,
        max_size=PRODUCT_MAX_IMAGE_PX,
        quality=PRODUCT_IMAGE_QUALITY,
        target_bytes=PRODUCT_MAX_IMAGE_BYTES,
    )


def compress_image(
    path: str,
    max_size: int = None,
    quality: int = None,
    target_bytes: int = None,
) -> None:
    if Image is None:
        return
    try:
        img = Image.open(path)
        if ImageOps is not None:
            img = ImageOps.exif_transpose(img)
        if max_size is None:
            max_size = REVIEW_MAX_IMAGE_PX
        if quality is None:
            quality = REVIEW_IMAGE_QUALITY
        if target_bytes is None:
            target_bytes = REVIEW_MAX_IMAGE_BYTES
        img_format = (img.format or "JPEG").upper()
        img.thumbnail((max_size, max_size))
        if img_format in {"JPEG", "JPG"} and img.mode in {"RGBA", "P"}:
            img = img.convert("RGB")
        save_kwargs = {}
        if img_format in {"JPEG", "JPG"}:
            save_kwargs.update({"optimize": True, "exif": b""})
        elif img_format == "WEBP":
            save_kwargs.update({"method": 6})

        current_quality = max(35, min(quality, 95))
        while True:
            if img_format in {"JPEG", "JPG", "WEBP"}:
                save_kwargs["quality"] = current_quality
            img.save(path, img_format, **save_kwargs)
            if target_bytes <= 0:
                break
            try:
                if os.path.getsize(path) <= target_bytes:
                    break
            except Exception:
                break
            if current_quality <= 40 and max(img.size) <= 800:
                break
            if current_quality > 40:
                current_quality = max(40, current_quality - 10)
            else:
                max_size = max(800, int(max_size * 0.85))
                img.thumbnail((max_size, max_size))
    except Exception:
        return


def _uploads_use_static() -> bool:
    if not UPLOAD_ROOT:
        return True
    if os.path.isabs(UPLOAD_ROOT):
        try:
            return os.path.normpath(UPLOAD_ROOT) == os.path.normpath(app.static_folder)
        except Exception:
            return False
    return os.path.normpath(UPLOAD_ROOT) == "static"


def _cloudinary_upload(file_storage, folder: str) -> Optional[str]:
    if not USE_CLOUDINARY or not cloudinary:
        return None
    try:
        result = cloudinary.uploader.upload(
            file_storage,
            folder=folder,
            resource_type="image",
            overwrite=False,
            unique_filename=True,
        )
        return result.get("secure_url") or result.get("url")
    except Exception:
        return None


def _cloudinary_upload_path(
    path: str,
    folder: str,
    public_id: Optional[str] = None,
    overwrite: bool = False,
) -> Optional[str]:
    if not USE_CLOUDINARY or not cloudinary:
        return None
    if not path or not os.path.isfile(path):
        return None
    try:
        options = {
            "folder": folder,
            "resource_type": "image",
            "overwrite": overwrite,
        }
        if public_id:
            options["public_id"] = public_id
            options["unique_filename"] = False
        else:
            options["unique_filename"] = True
        result = cloudinary.uploader.upload(path, **options)
        return result.get("secure_url") or result.get("url")
    except Exception:
        return None


def image_url(path):
    if not path:
        return url_for("static", filename="images/logo.jpeg")
    try:
        path = path.decode("utf-8")
    except AttributeError:
        path = str(path)

    if path.startswith("http://") or path.startswith("https://"):
        return path

    path = path.lstrip("/")
    if path.startswith("static/"):
        path = path[len("static/") :]

    if not (path.startswith("images/") or path.startswith("review_photos/")):
        path = f"images/{path}"

    if STATIC_CDN_BASE:
        return f"{STATIC_CDN_BASE.rstrip('/')}/{path.lstrip('/')}"
    if _uploads_use_static():
        return url_for("static", filename=path)
    upload_root = app.config.get("UPLOAD_ROOT", UPLOAD_ROOT)
    try:
        if os.path.isfile(os.path.join(upload_root, path)):
            return url_for("uploaded_file", filename=path)
    except Exception:
        pass
    return url_for("static", filename=path)


def _format_hash_chunks(value: str, chunk: int = 4, max_chunks: int = 5) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", value.upper())
    parts = [cleaned[i : i + chunk] for i in range(0, len(cleaned), chunk)]
    return "-".join(parts[:max_chunks])


def _make_qr_data_uri(payload: str) -> Optional[str]:
    if not qrcode:
        return None
    qr = qrcode.QRCode(border=1, box_size=3)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_scu_info(order_id: int, reference: str, issued_at: datetime, items, total: float):
    base = f"{order_id}|{reference}|{issued_at.isoformat()}|{len(items)}|{total:.2f}"
    internal_hash = hashlib.sha1(base.encode("utf-8")).hexdigest().upper()
    signature_hash = hashlib.sha256((base + "|SCU").encode("utf-8")).hexdigest().upper()
    return {
        "date": issued_at.strftime("%Y-%m-%d %H:%M:%S"),
        "scu_id": f"{reference}",
        "receipt_number": f"{order_id}",
        "item_count": f"{len(items)}",
        "internal_data": _format_hash_chunks(internal_hash, chunk=4, max_chunks=5),
        "receipt_signature": _format_hash_chunks(signature_hash, chunk=4, max_chunks=6),
    }


def _migrate_static_uploads():
    upload_root = app.config.get("UPLOAD_ROOT", UPLOAD_ROOT)
    if not upload_root:
        return {"error": "Upload root is not configured."}
    if _uploads_use_static():
        return {"error": "Uploads already use the static folder; migration not needed."}
    if not os.path.isdir(upload_root):
        return {"error": f"Upload root '{upload_root}' is not available."}

    marker_path = os.path.join(upload_root, ".migrated_static_assets")
    if os.path.isfile(marker_path):
        return {"error": "Migration already completed."}

    copied = 0
    skipped = 0
    errors = 0
    for subdir in ("images", "review_photos"):
        src_dir = os.path.join("static", subdir)
        if not os.path.isdir(src_dir):
            continue
        dest_dir = os.path.join(upload_root, subdir)
        os.makedirs(dest_dir, exist_ok=True)
        for root, _, files in os.walk(src_dir):
            rel_root = os.path.relpath(root, src_dir)
            for filename in files:
                src_path = os.path.join(root, filename)
                rel_path = filename if rel_root == "." else os.path.join(rel_root, filename)
                dest_path = os.path.join(dest_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                if os.path.exists(dest_path):
                    skipped += 1
                    continue
                try:
                    shutil.copy2(src_path, dest_path)
                    copied += 1
                except Exception:
                    errors += 1

    try:
        with open(marker_path, "w", encoding="utf-8") as handle:
            handle.write(f"migrated_at={datetime.utcnow().isoformat()}Z\n")
    except Exception:
        pass

    return {"copied": copied, "skipped": skipped, "errors": errors}


def _resolve_local_upload_path(path: str) -> Optional[str]:
    if not path:
        return None
    if str(path).startswith(("http://", "https://")):
        return None
    normalized = str(path).lstrip("/")
    if normalized.startswith("static/"):
        normalized = normalized[len("static/") :]
    upload_root = app.config.get("UPLOAD_ROOT", UPLOAD_ROOT)
    candidate = os.path.join(upload_root, normalized)
    if os.path.isfile(candidate):
        return candidate
    fallback = os.path.join("static", normalized)
    if os.path.isfile(fallback):
        return fallback
    return None


def _migrate_cloudinary_assets(limit: int = 25):
    if not USE_CLOUDINARY:
        return {"error": "Cloudinary is not configured."}

    copied = 0
    skipped = 0
    errors = 0
    processed = 0

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT product_id, image_url FROM products")
                rows = cur.fetchall() or []
            except Exception:
                rows = []

            for row in rows:
                if processed >= limit:
                    break
                product_id = _row_at(row, 0)
                image_url = _row_at(row, 1, "")
                if not image_url or str(image_url).startswith(("http://", "https://")):
                    skipped += 1
                    processed += 1
                    continue
                local_path = _resolve_local_upload_path(image_url)
                if not local_path:
                    errors += 1
                    processed += 1
                    continue
                new_url = _cloudinary_upload_path(
                    local_path,
                    "bigoh/products",
                    public_id=str(product_id),
                    overwrite=True,
                )
                if not new_url:
                    errors += 1
                    processed += 1
                    continue
                try:
                    cur.execute(
                        "UPDATE products SET image_url = %s WHERE product_id = %s",
                        (new_url, product_id),
                    )
                    copied += 1
                except Exception:
                    errors += 1
                processed += 1

            review_rows = []
            if processed < limit:
                try:
                    cur.execute("SELECT id, review_photo FROM product_reviews WHERE review_photo IS NOT NULL")
                    review_rows = cur.fetchall() or []
                except Exception:
                    review_rows = []

            for row in review_rows:
                if processed >= limit:
                    break
                review_id = _row_at(row, 0)
                review_photo = _row_at(row, 1, "")
                if not review_photo or str(review_photo).startswith(("http://", "https://")):
                    skipped += 1
                    processed += 1
                    continue
                local_path = _resolve_local_upload_path(review_photo)
                if not local_path:
                    errors += 1
                    processed += 1
                    continue
                new_url = _cloudinary_upload_path(
                    local_path,
                    "bigoh/reviews",
                    public_id=str(review_id),
                    overwrite=True,
                )
                if not new_url:
                    errors += 1
                    processed += 1
                    continue
                try:
                    cur.execute(
                        "UPDATE product_reviews SET review_photo = %s WHERE id = %s",
                        (new_url, review_id),
                    )
                    copied += 1
                except Exception:
                    errors += 1
                processed += 1
        conn.commit()
    finally:
        conn.close()

    return {
        "copied": copied,
        "skipped": skipped,
        "errors": errors,
        "processed": processed,
        "limit": limit,
    }


def _cloudinary_list_folder(folder: str, max_results: int = 200):
    if not USE_CLOUDINARY or not cloudinary:
        return {"error": "Cloudinary is not configured."}
    resources = []
    next_cursor = None
    try:
        while True:
            params = {
                "type": "upload",
                "resource_type": "image",
                "prefix": f"{folder}/",
                "max_results": min(max_results, 500),
            }
            if next_cursor:
                params["next_cursor"] = next_cursor
            result = cloudinary.api.resources(**params)
            batch = result.get("resources", []) or []
            resources.extend(batch)
            next_cursor = result.get("next_cursor")
            if not next_cursor or len(resources) >= max_results:
                break
        return {"resources": resources[:max_results]}
    except Exception as exc:
        return {"error": str(exc)}


def _cloudinary_find_duplicates(max_results: int = 200):
    # Duplicates are assets not following numeric public_id convention
    duplicates = []
    scanned = 0
    for folder in ("bigoh/products", "bigoh/reviews"):
        resp = _cloudinary_list_folder(folder, max_results=max_results)
        if "error" in resp:
            return {"error": resp["error"]}
        for res in resp.get("resources", []):
            public_id = res.get("public_id", "")
            scanned += 1
            leaf = public_id.split("/")[-1]
            if not leaf.isdigit():
                duplicates.append(public_id)
            if len(duplicates) >= max_results:
                break
        if len(duplicates) >= max_results:
            break
    return {"duplicates": duplicates, "scanned": scanned, "limit": max_results}


def _cloudinary_delete_public_ids(public_ids, batch_size: int = 100):
    if not USE_CLOUDINARY or not cloudinary:
        return {"error": "Cloudinary is not configured."}
    deleted = 0
    errors = 0
    for i in range(0, len(public_ids), batch_size):
        chunk = public_ids[i : i + batch_size]
        try:
            result = cloudinary.api.delete_resources(
                chunk,
                resource_type="image",
                type="upload",
            )
            deleted += sum(1 for _, status in (result.get("deleted") or {}).items() if status == "deleted")
        except Exception:
            errors += len(chunk)
    return {"deleted": deleted, "errors": errors}


def _cloudinary_public_id_from_url(url: str) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    if "res.cloudinary.com" not in url or "/image/upload/" not in url:
        return None
    try:
        tail = url.split("/image/upload/", 1)[1]
        # strip version segment like v123/
        if tail.startswith("v") and "/" in tail:
            version_part, rest = tail.split("/", 1)
            if version_part[1:].isdigit():
                tail = rest
        # drop extension
        if "." in tail:
            tail = ".".join(tail.split(".")[:-1])
        return tail.strip("/")
    except Exception:
        return None


def _cloudinary_db_public_ids(conn) -> set:
    ids = set()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT image_url FROM products WHERE image_url IS NOT NULL")
                for row in cur.fetchall() or []:
                    pid = _cloudinary_public_id_from_url(_row_at(row, 0, ""))
                    if pid:
                        ids.add(pid)
            except Exception:
                pass
            try:
                cur.execute("SELECT review_photo FROM product_reviews WHERE review_photo IS NOT NULL")
                for row in cur.fetchall() or []:
                    pid = _cloudinary_public_id_from_url(_row_at(row, 0, ""))
                    if pid:
                        ids.add(pid)
            except Exception:
                pass
    except Exception:
        pass
    return ids


def _cloudinary_delete_duplicate_content(max_results: int = 500):
    if not USE_CLOUDINARY or not cloudinary:
        return {"error": "Cloudinary is not configured."}
    conn = get_db_connection()
    try:
        referenced = _cloudinary_db_public_ids(conn)
    finally:
        conn.close()

    resources = []
    for folder in ("bigoh/products", "bigoh/reviews"):
        resp = _cloudinary_list_folder(folder, max_results=max_results)
        if "error" in resp:
            return {"error": resp["error"]}
        resources.extend(resp.get("resources", []))

    # group by content etag if available, else by bytes+dimensions
    groups = {}
    for res in resources:
        etag = res.get("etag")
        if etag:
            key = f"etag:{etag}"
        else:
            key = f"sig:{res.get('bytes')}:{res.get('width')}:{res.get('height')}"
        groups.setdefault(key, []).append(res)

    to_delete = []
    kept = 0
    for items in groups.values():
        if len(items) <= 1:
            kept += len(items)
            continue
        # Prefer keeping any referenced public_id
        referenced_items = [r for r in items if r.get("public_id") in referenced]
        if referenced_items:
            keep_ids = {r.get("public_id") for r in referenced_items}
        else:
            # keep the first item (older first if created_at present)
            items_sorted = sorted(items, key=lambda r: r.get("created_at") or "")
            keep_ids = {items_sorted[0].get("public_id")}
        for r in items:
            pid = r.get("public_id")
            if pid in keep_ids:
                kept += 1
            else:
                to_delete.append(pid)

    if not to_delete:
        return {"deleted": 0, "errors": 0, "kept": kept, "scanned": len(resources)}

    delete_result = _cloudinary_delete_public_ids(to_delete)
    if "error" in delete_result:
        return delete_result
    delete_result.update({"kept": kept, "scanned": len(resources)})
    return delete_result

def slugify_category(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or "category"

WATCH_CATEGORY_ALIASES = {
    "watch",
    "watches",
    "waches",
    "men watch",
    "ladies watch",
    "women watch",
    "female watch",
    "gents watch",
    "male watch",
}


def normalize_category_label(name: str) -> str:
    cleaned = str(name or "").strip()
    if not cleaned:
        return ""
    return cleaned


def get_managed_categories():
    categories = []
    seen = set()
    for raw in MANAGED_CATEGORIES:
        name = str(raw or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        categories.append(name)
    return categories


def coerce_allowed_category(raw_category: str, allowed_categories):
    category = str(raw_category or "").strip()
    if not category:
        return ""
    lookup = {
        str(name).strip().lower(): str(name).strip()
        for name in (allowed_categories or [])
        if str(name).strip()
    }
    return lookup.get(category.lower(), "")


def get_category_overview(conn=None, limit=None):
    owns_conn = False
    if conn is None:
        try:
            conn = get_db_connection()
        except Exception:
            return []
        owns_conn = True
    try:
        ensure_products_visibility_column(conn)
        with conn.cursor() as cur:
            base_sql = """
                SELECT c.category, c.total, p.image_url
                FROM (
                    SELECT category, COUNT(*) AS total, MAX(product_id) AS latest_id
                    FROM products
                    WHERE category IS NOT NULL AND category <> ''
                      AND (is_hidden IS NULL OR is_hidden = 0)
                    GROUP BY category
                ) c
                LEFT JOIN products p ON p.product_id = c.latest_id
                ORDER BY c.total DESC, c.category ASC
            """
            params = []
            if limit is not None and int(limit) > 0:
                base_sql += " LIMIT %s"
                params.append(int(limit))
            cur.execute(base_sql, params)
            rows = cur.fetchall() or []
        allowed_keys = {name.lower() for name in get_managed_categories()}
        categories = []
        for row in rows:
            raw_name = str(_row_at(row, 0, "") or "").strip()
            if not raw_name:
                continue
            if allowed_keys and raw_name.lower() not in allowed_keys:
                continue
            display_name = normalize_category_label(raw_name)
            categories.append(
                {
                    "name": display_name,
                    "db_name": raw_name,
                    "slug": slugify_category(raw_name),
                    "count": int(_row_at(row, 1, 0) or 0),
                    "image": _row_at(row, 2, "") or "images/hero.jpg",
                }
            )
        return categories
    except Exception:
        return []
    finally:
        if owns_conn:
            conn.close()
def _row_at(row, idx, default=None):
    if row is None:
        return default
    if isinstance(row, dict):
        try:
            return list(row.values())[idx]
        except Exception:
            return default
    try:
        return row[idx]
    except Exception:
        return default


def _is_safe_url(target):
    if not target:
        return False
    try:
        ref = urlparse(request.host_url)
        test = urlparse(target)
        return (
            test.scheme in ("http", "https", "")
            and ref.netloc == test.netloc
        )
    except Exception:
        return False


def _remember_next_url():
    next_param = request.args.get("next", "")
    candidate = next_param or (request.referrer or "")
    if not _is_safe_url(candidate): 
        return 
    path = urlparse(candidate).path or ""
    if path in ("/signin", "/signup"):
        return
    session["next_url"] = candidate


def _pop_next_url():
    next_url = session.pop("next_url", None)
    if not _is_safe_url(next_url):
        return None
    path = urlparse(next_url).path or ""
    if path in ("/signin", "/signup"):
        return None
    return next_url


def _cron_authorized() -> bool:
    if not CRON_SECRET:
        return False
    token = request.headers.get("X-Task-Secret") or request.args.get("token", "")
    return bool(token and hmac.compare_digest(str(token), str(CRON_SECRET)))


def _build_status_message(status: str, order_id: int, user_name: str, receipt_link: str) -> str:
    status = (status or "").upper()
    templates = {
        "PENDING": (
            "Hello {name},\n"
            "We have received your order #{order_id} and it is now pending confirmation.\n"
            "We will update you shortly.\n"
            "Need help? WhatsApp {support}\n"
            "- {business}"
        ),
        "PROCESSING": (
            "Hello {name},\n"
            "Your order #{order_id} is now being processed.\n"
            "We are preparing your items for dispatch.\n"
            "Need help? WhatsApp {support}\n"
            "- {business}"
        ),
        "COMPLETED": (
            "Hello {name},\n"
            "Your order #{order_id} has been completed and payment has been confirmed.\n"
            "Invoice/Receipt: {receipt}\n"
            "For any queries, contact us on WhatsApp {support}.\n"
            "Regards,\n"
            "{business}"
        ),
        "DELIVERED": (
            "Hello {name},\n"
            "Your order #{order_id} has been delivered.\n"
            "Payment is due after delivery (Cash on Delivery).\n"
            "Your final paid receipt will be shared once payment is confirmed.\n"
            "Thank you for your business.\n"
            "Regards,\n"
            "{business}"
        ),
        "CANCELLED": (
            "Hello {name},\n"
            "Your order #{order_id} has been cancelled.\n"
            "If this is unexpected, please contact us.\n"
            "WhatsApp {support}\n"
            "- {business}"
        ),
    }
    template = templates.get(status)
    if not template:
        template = (
            "Hello {name},\n"
            "Your order #{order_id} status has been updated to {status}.\n"
            "Need help? WhatsApp {support}\n"
            "- {business}"
        )
    return template.format(
        name=user_name,
        order_id=order_id,
        status=status.title(),
        receipt=receipt_link,
        support=SUPPORT_WHATSAPP,
        business=BUSINESS_NAME,
    )


@app.before_request
def csrf_protect():
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        token = (
            request.form.get("csrf_token")
            or request.headers.get("X-CSRF-Token")
            or request.headers.get("X-CSRFToken")
        )
        session_token = session.get("_csrf_token", "")
        if not token or not session_token or not hmac.compare_digest(str(token), str(session_token)):
            app.logger.warning("CSRF blocked: %s %s from %s", request.method, request.path, _client_ip())
            if request.headers.get("X-Requested-With") == "XMLHttpRequest" or "application/json" in request.headers.get("Accept", ""):
                return jsonify(ok=False, message="Invalid CSRF token.", status=400), 400
            return render_template(
                "error.html",
                title="Request blocked",
                message="Invalid CSRF token. Please refresh the page and try again.",
                status_code=400,
            ), 400


@app.before_request
def track_active_session():
    if request.path.startswith("/static/") or request.path == "/favicon.ico":
        return
    if request.method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"):
        return
    try:
        session_id = session.get("active_session_id")
        if not session_id:
            session_id = uuid.uuid4().hex
            session["active_session_id"] = session_id

        user_id = session.get("username")
        username = session.get("key")
        ip_address = _client_ip()
        user_agent = request.headers.get("User-Agent", "")[:255]
        current_path = (request.path or "")[:255]
        now = datetime.now()

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                if not ensure_active_sessions_table(cur):
                    return
                conn.commit()
                cur.execute(
                    """
                    INSERT INTO active_sessions
                    (session_id, user_id, username, ip_address, user_agent, last_seen, current_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        user_id=VALUES(user_id),
                        username=VALUES(username),
                        ip_address=VALUES(ip_address),
                        user_agent=VALUES(user_agent),
                        last_seen=VALUES(last_seen),
                        current_path=VALUES(current_path)
                    """,
                    (
                        session_id,
                        user_id,
                        username,
                        ip_address,
                        user_agent,
                        now,
                        current_path,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        return


@app.after_request
def add_cache_headers(response):
    if request.path.startswith("/static/"):
        response.headers.setdefault(
            "Cache-Control",
            "public, max-age=31536000, immutable",
        )
    if request.path.startswith("/uploads/"):
        response.headers.setdefault(
            "Cache-Control",
            "public, max-age=31536000, immutable",
        )
    return response


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config.get("UPLOAD_ROOT", UPLOAD_ROOT),
        filename,
    )


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        app.static_folder,
        "favicon.ico",
        mimetype="image/x-icon",
    )


@app.route("/tasks/low-stock", methods=["GET", "POST"])
def task_low_stock():
    if not _cron_authorized():
        return "Unauthorized", 401
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT product_id, product_name, category, price, stock, image_url
                FROM products
                WHERE stock <= %s
                ORDER BY stock ASC, product_id DESC
                """,
                (LOW_STOCK_THRESHOLD,),
            )
            low_stock = cur.fetchall() or []
        send_low_stock_alerts(conn, low_stock)
    finally:
        conn.close()
    return jsonify(ok=True, low_stock=len(low_stock))


@app.route("/tasks/back-in-stock", methods=["GET", "POST"])
def task_back_in_stock():
    if not _cron_authorized():
        return "Unauthorized", 401
    conn = get_db_connection()
    try:
        processed = process_back_in_stock_alerts(conn)
    finally:
        conn.close()
    return jsonify(ok=True, processed=processed)


@app.route("/tasks/abandoned-carts", methods=["GET", "POST"])
def task_abandoned_carts():
    if not _cron_authorized():
        return "Unauthorized", 401
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if not ensure_abandoned_carts_table(cur):
                return jsonify(ok=False, error="table"), 500
            conn.commit()
            cur.execute(
                """
                SELECT a.user_id, a.cart_json, u.phone, u.username
                FROM abandoned_carts a
                JOIN users u ON u.id = a.user_id
                WHERE a.notified_at IS NULL
                  AND a.updated_at <= (NOW() - INTERVAL %s HOUR)
                LIMIT 200
                """,
                (ABANDONED_CART_HOURS,),
            )
            rows = cur.fetchall() or []
            sent = 0
            for row in rows:
                user_id = _row_at(row, 0)
                cart_json = _row_at(row, 1, "")
                phone = _row_at(row, 2, "")
                name = _row_at(row, 3, "") or "Customer"
                if not cart_json or cart_json == "{}":
                    continue
                cart_link = url_for("cart", _external=True)
                message = (
                    f"Hello {name},\n"
                    f"You have items waiting in your cart.\n"
                    f"Complete your order here: {cart_link}\n"
                    f"Need help? WhatsApp {SUPPORT_WHATSAPP}\n"
                    f"- {BUSINESS_NAME}"
                )
                if _send_whatsapp_message(phone, message):
                    cur.execute(
                        "UPDATE abandoned_carts SET notified_at = NOW() WHERE user_id = %s",
                        (user_id,),
                    )
                    sent += 1
            conn.commit()
    finally:
        conn.close()
    return jsonify(ok=True, sent=sent)


@app.route("/tasks/review-requests", methods=["GET", "POST"])
def task_review_requests():
    if not _cron_authorized():
        return "Unauthorized", 401
    conn = get_db_connection()
    try:
        ensure_orders_delivery_columns(conn)
        with conn.cursor() as cur:
            if not ensure_order_review_requests_table(cur):
                return jsonify(ok=False, error="table"), 500
            conn.commit()
            cur.execute(
                """
                SELECT o.order_id, o.user_id, u.phone, u.username, o.delivered_at
                FROM orders o
                JOIN users u ON u.id = o.user_id
                LEFT JOIN order_review_requests r ON r.order_id = o.order_id
                WHERE o.status = 'DELIVERED'
                  AND o.delivered_at IS NOT NULL
                  AND o.delivered_at <= (NOW() - INTERVAL %s HOUR)
                  AND r.order_id IS NULL
                LIMIT 200
                """,
                (REVIEW_REQUEST_DELAY_HOURS,),
            )
            rows = cur.fetchall() or []
            sent = 0
            for row in rows:
                order_id = _row_at(row, 0)
                phone = _row_at(row, 2, "")
                name = _row_at(row, 3, "") or "Customer"
                orders_link = url_for("my_orders", _external=True)
                message = (
                    f"Hello {name},\n"
                    f"Thank you for shopping with {BUSINESS_NAME}.\n"
                    f"Please leave a review for your order #{order_id}.\n"
                    f"Track your order here: {orders_link}\n"
                    f"We appreciate your feedback.\n"
                    f"- {BUSINESS_NAME}"
                )
                if _send_whatsapp_message(phone, message):
                    cur.execute(
                        "INSERT INTO order_review_requests (order_id, notified_at) VALUES (%s, NOW())",
                        (order_id,),
                    )
                    sent += 1
            conn.commit()
    finally:
        conn.close()
    return jsonify(ok=True, sent=sent)


@app.route("/tasks/stale-pending", methods=["GET", "POST"])
def task_stale_pending():
    if not _cron_authorized():
        return "Unauthorized", 401
    conn = get_db_connection()
    try:
        ensure_orders_delivery_columns(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT o.order_id, o.user_id, u.phone, u.username
                FROM orders o
                JOIN users u ON u.id = o.user_id
                WHERE o.status IN ('PENDING', 'PROCESSING')
                  AND o.created_at <= (NOW() - INTERVAL %s HOUR)
                LIMIT 200
                """,
                (STALE_PENDING_HOURS,),
            )
            rows = cur.fetchall() or []
            updated = 0
            for row in rows:
                order_id = _row_at(row, 0)
                phone = _row_at(row, 2, "")
                name = _row_at(row, 3, "") or "Customer"
                receipt_link = url_for("order_receipt", order_id=order_id, _external=True)
                message = _build_status_message("CANCELLED", order_id, name, receipt_link)
                cur.execute(
                    "UPDATE orders SET status='CANCELLED', status_updated_at=NOW() WHERE order_id=%s",
                    (order_id,),
                )
                if WHATSAPP_STATUS_UPDATES_ENABLED:
                    _send_whatsapp_message(phone, message)
                updated += 1
            conn.commit()
    finally:
        conn.close()
    return jsonify(ok=True, updated=updated)


@app.route("/tasks/auto-hide", methods=["GET", "POST"])
def task_auto_hide():
    if not _cron_authorized():
        return "Unauthorized", 401
    conn = get_db_connection()
    try:
        ensure_products_visibility_column(conn)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE products SET is_hidden=1 WHERE stock <= 0 AND (is_hidden IS NULL OR is_hidden = 0)"
            )
            hidden = cur.rowcount
            cur.execute(
                "UPDATE products SET is_hidden=0 WHERE stock > 0 AND is_hidden = 1"
            )
            unhidden = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return jsonify(ok=True, hidden=hidden, unhidden=unhidden)


@app.route("/tasks/daily-summary", methods=["GET", "POST"])
def task_daily_summary():
    if not _cron_authorized():
        return "Unauthorized", 401
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(subtotal), 0)
                FROM orders
                WHERE DATE(created_at) = CURDATE()
                """
            )
            row = cur.fetchone()
            orders_count = int(_row_at(row, 0, 0) or 0)
            revenue = float(_row_at(row, 1, 0) or 0)

            cur.execute(
                "SELECT COUNT(*) FROM orders WHERE DATE(created_at)=CURDATE() AND status='PENDING'"
            )
            pending = int(_row_at(cur.fetchone(), 0, 0) or 0)
            cur.execute(
                "SELECT COUNT(*) FROM orders WHERE DATE(created_at)=CURDATE() AND status='PROCESSING'"
            )
            processing = int(_row_at(cur.fetchone(), 0, 0) or 0)
            cur.execute(
                "SELECT COUNT(*) FROM orders WHERE DATE(created_at)=CURDATE() AND status='DELIVERED'"
            )
            delivered = int(_row_at(cur.fetchone(), 0, 0) or 0)
            cur.execute(
                "SELECT COUNT(*) FROM orders WHERE DATE(created_at)=CURDATE() AND status='COMPLETED'"
            )
            completed = int(_row_at(cur.fetchone(), 0, 0) or 0)
            cur.execute(
                "SELECT COUNT(*) FROM orders WHERE DATE(created_at)=CURDATE() AND status='CANCELLED'"
            )
            cancelled = int(_row_at(cur.fetchone(), 0, 0) or 0)

            cur.execute(
                """
                SELECT oi.product_name, SUM(oi.quantity) AS qty
                FROM order_items oi
                JOIN orders o ON o.order_id = oi.order_id
                WHERE DATE(o.created_at) = CURDATE()
                GROUP BY oi.product_name
                ORDER BY qty DESC
                LIMIT 3
                """
            )
            top = cur.fetchall() or []

            cur.execute(
                "SELECT COUNT(*) FROM products WHERE stock <= %s",
                (LOW_STOCK_THRESHOLD,),
            )
            low_stock = int(_row_at(cur.fetchone(), 0, 0) or 0)
    finally:
        conn.close()

    top_lines = "\n".join([f"- {row[0]} ({int(row[1])})" for row in top]) or "No sales yet."
    message = (
        f"{BUSINESS_NAME} Daily Summary (today)\n"
        f"Orders: {orders_count}\n"
        f"Revenue: KES {revenue:,.2f}\n"
        f"Pending: {pending} | Processing: {processing} | Delivered: {delivered} | Completed: {completed} | Cancelled: {cancelled}\n"
        f"Low stock items: {low_stock}\n"
        f"Top products:\n{top_lines}"
    )
    _send_whatsapp_alert(message)
    return jsonify(ok=True, orders=orders_count, revenue=revenue)

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


_db_connect_block_until = 0.0


def _db_connect_block_remaining_seconds() -> float:
    if DB_FAILURE_BACKOFF_SECONDS <= 0:
        return 0.0
    return max(0.0, _db_connect_block_until - time.monotonic())


def _mark_db_connect_failure() -> None:
    global _db_connect_block_until
    if DB_FAILURE_BACKOFF_SECONDS <= 0:
        return
    _db_connect_block_until = time.monotonic() + DB_FAILURE_BACKOFF_SECONDS


def _clear_db_connect_failure() -> None:
    global _db_connect_block_until
    _db_connect_block_until = 0.0


def get_db_connection():
    blocked_for = _db_connect_block_remaining_seconds()
    if blocked_for > 0:
        wait_seconds = int(blocked_for) + 1
        raise RuntimeError(
            f"Database temporarily unavailable. Retry in about {wait_seconds}s."
        )

    db_url = os.getenv("DATABASE_URL") or os.getenv("MYSQL_URL") or os.getenv("DB_URL")
    if db_url:
        try:
            cfg = _parse_db_url(db_url)
        except Exception as exc:
            raise RuntimeError(f"Invalid DATABASE_URL/MYSQL_URL: {exc}") from exc
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

    running_on_railway = bool(
        os.getenv("RAILWAY_PROJECT_ID")
        or os.getenv("RAILWAY_ENVIRONMENT")
        or os.getenv("RAILWAY_ENVIRONMENT_NAME")
        or os.getenv("RAILWAY_STATIC_URL")
    )
    if host.endswith(".railway.internal") and not running_on_railway:
        raise RuntimeError(
            "Database host uses Railway internal DNS but this app is not running on Railway. "
            "Use the public MySQL host/port or DATABASE_URL."
        )

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
        connect_timeout=max(1, int(DB_CONNECT_TIMEOUT)),
        read_timeout=max(1, int(DB_READ_TIMEOUT)),
        write_timeout=max(1, int(DB_WRITE_TIMEOUT)),
    )
    if not ssl_disabled:
        connect_kwargs["ssl"] = {"ssl": {}}

    try:
        conn = pymysql.connect(**connect_kwargs)
        _clear_db_connect_failure()
        return conn
    except pymysql.err.OperationalError as exc:
        _mark_db_connect_failure()
        hint = ""
        if host.endswith(".railway.internal"):
            hint = (
                " The host looks like a Railway internal address. "
                "Use the public MySQL host/port (or DATABASE_URL) when deploying outside Railway."
            )
        raise pymysql.err.OperationalError(
            exc.args[0], f"{exc.args[1]}{hint}"
        ) from exc
    except OSError as exc:
        _mark_db_connect_failure()
        hint = ""
        if host.endswith(".railway.internal"):
            hint = (
                " The host looks like a Railway internal address. "
                "Use the public MySQL host/port (or DATABASE_URL) when deploying outside Railway."
            )
        raise RuntimeError(
            f"Database connection failed to {host}:{port} ({exc}).{hint}"
        ) from exc


def _scalar(cur, query, params=None, default=0):
    try:
        cur.execute(query, params or ())
        row = cur.fetchone()
        if not row:
            return default
        value = _row_at(row, 0, default)
        return default if value is None else value
    except Exception:
        return default


def get_loyalty_discount(conn, user_id: int, subtotal: float):
    if not LOYALTY_ENABLED or not user_id or subtotal <= 0:
        return 0.0, "", 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM orders
                WHERE user_id = %s
                  AND status IN ('DELIVERED', 'COMPLETED')
                """,
                (user_id,),
            )
            count = int(_row_at(cur.fetchone(), 0, 0) or 0)
    except Exception:
        return 0.0, "", 0

    if count < LOYALTY_REPEAT_ORDERS_MIN or LOYALTY_REPEAT_DISCOUNT_PCT <= 0:
        return 0.0, "", count

    discount = round(subtotal * (LOYALTY_REPEAT_DISCOUNT_PCT / 100.0), 2)
    reason = f"Loyalty {LOYALTY_REPEAT_DISCOUNT_PCT:.0f}% (repeat customer)"
    return discount, reason, count


REFERRAL_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _normalize_referral_code(value: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return cleaned[:24]


def ensure_coupon_schema(conn) -> bool:
    if app.config.get("COUPON_SCHEMA_READY"):
        return True
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_coupons (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    coupon_code VARCHAR(40) NOT NULL,
                    amount DECIMAL(10,2) NOT NULL DEFAULT 0.00,
                    reason VARCHAR(160) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
                    source_user_id INT NULL,
                    issued_for_user_id INT NULL,
                    used_order_id INT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    used_at DATETIME NULL,
                    expires_at DATETIME NULL,
                    UNIQUE KEY uniq_coupon_code (coupon_code),
                    KEY idx_coupon_user_status (user_id, status, expires_at),
                    KEY idx_coupon_used_order (used_order_id)
                )
                """
            )
        conn.commit()
        app.config["COUPON_SCHEMA_READY"] = True
        return True
    except Exception:
        app.config["COUPON_SCHEMA_READY"] = False
        return False


def ensure_referral_schema(conn) -> bool:
    if not REFERRAL_ENABLED:
        return False
    if app.config.get("REFERRAL_SCHEMA_READY"):
        return True
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_referrals (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    referrer_user_id INT NOT NULL,
                    referred_user_id INT NOT NULL,
                    referral_code VARCHAR(24) NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uniq_referred_user (referred_user_id),
                    UNIQUE KEY uniq_referral_pair (referrer_user_id, referred_user_id),
                    KEY idx_referrer_user_id (referrer_user_id),
                    KEY idx_referral_code (referral_code)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_coupons (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    coupon_code VARCHAR(40) NOT NULL,
                    amount DECIMAL(10,2) NOT NULL DEFAULT 0.00,
                    reason VARCHAR(160) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
                    source_user_id INT NULL,
                    issued_for_user_id INT NULL,
                    used_order_id INT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    used_at DATETIME NULL,
                    expires_at DATETIME NULL,
                    UNIQUE KEY uniq_coupon_code (coupon_code),
                    KEY idx_coupon_user_status (user_id, status, expires_at),
                    KEY idx_coupon_used_order (used_order_id)
                )
                """
            )
            if not table_has_column(conn, "users", "referral_code"):
                cur.execute("ALTER TABLE users ADD COLUMN referral_code VARCHAR(24) NULL")
                app.config[_schema_cache_key("users", "referral_code")] = True
            try:
                cur.execute("CREATE UNIQUE INDEX uniq_users_referral_code ON users (referral_code)")
            except Exception:
                pass
        conn.commit()
        app.config["REFERRAL_SCHEMA_READY"] = True
        return True
    except Exception:
        app.config["REFERRAL_SCHEMA_READY"] = False
        return False


def _generate_referral_code(length: int = 8) -> str:
    code_len = max(6, min(int(length or 8), 16))
    return "".join(secrets.choice(REFERRAL_ALPHABET) for _ in range(code_len))


def ensure_user_referral_code(conn, user_id: int, username: str = "") -> str:
    if not REFERRAL_ENABLED or not user_id:
        return ""
    if not app.config.get("REFERRAL_SCHEMA_READY") and not ensure_referral_schema(conn):
        return ""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT referral_code FROM users WHERE id=%s LIMIT 1", (user_id,))
            row = cur.fetchone()
            existing = _normalize_referral_code(_row_at(row, 0, ""))
            if existing:
                return existing

            base = _normalize_referral_code(username)[:6]
            for _ in range(24):
                suffix = _generate_referral_code(REFERRAL_CODE_LEN)
                candidate = f"{base}{suffix}"[:24] if base else suffix
                cur.execute("SELECT 1 FROM users WHERE referral_code=%s LIMIT 1", (candidate,))
                if cur.fetchone():
                    continue
                cur.execute("UPDATE users SET referral_code=%s WHERE id=%s", (candidate, user_id))
                return candidate
    except Exception:
        return ""
    return ""


def _coupon_expires_at():
    if REFERRAL_COUPON_EXPIRES_DAYS <= 0:
        return None
    return _now_utc() + timedelta(days=REFERRAL_COUPON_EXPIRES_DAYS)


def _generate_coupon_code(prefix: str = "BIGOH") -> str:
    clean = _normalize_referral_code(prefix)[:8] or "BIGOH"
    return f"{clean}-{secrets.token_hex(4).upper()}"


def create_user_coupon(
    cur,
    user_id: int,
    amount: float,
    reason: str,
    source_user_id: int = None,
    issued_for_user_id: int = None,
    prefix: str = "BIGOH",
):
    if not user_id:
        return ""
    try:
        amount_value = round(float(amount or 0), 2)
    except Exception:
        amount_value = 0.0
    if amount_value < 0:
        amount_value = 0.0

    expires_at = _coupon_expires_at()
    for _ in range(8):
        coupon_code = _generate_coupon_code(prefix)
        try:
            cur.execute(
                """
                INSERT INTO user_coupons
                (user_id, coupon_code, amount, reason, status, source_user_id, issued_for_user_id, expires_at)
                VALUES (%s, %s, %s, %s, 'ACTIVE', %s, %s, %s)
                """,
                (
                    user_id,
                    coupon_code,
                    amount_value,
                    str(reason or "Signup coupon")[:160],
                    source_user_id,
                    issued_for_user_id,
                    expires_at,
                ),
            )
            return coupon_code
        except pymysql.err.IntegrityError:
            continue
        except Exception:
            return ""
    return ""


def issue_user_coupons(
    cur,
    user_id: int,
    count: int,
    reason: str,
    source_user_id: int = None,
    issued_for_user_id: int = None,
    prefix: str = "BIGOH",
    amount: float = None,
):
    try:
        coupon_count = max(0, int(count or 0))
    except Exception:
        coupon_count = 0
    if not user_id or coupon_count <= 0:
        return []

    coupon_amount = COUPON_UNIT_AMOUNT if amount is None else amount
    issued = []
    for _ in range(coupon_count):
        code = create_user_coupon(
            cur,
            user_id=user_id,
            amount=coupon_amount,
            reason=reason,
            source_user_id=source_user_id,
            issued_for_user_id=issued_for_user_id,
            prefix=prefix,
        )
        if code:
            issued.append(code)
    return issued


def apply_signup_coupon_rewards(conn, user_id: int) -> dict:
    result = {
        "applied": False,
        "message": "",
        "issued_count": 0,
    }
    if not user_id:
        return result
    if not ensure_coupon_schema(conn):
        return result

    reward_reason = "Signup reward (new account)"
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM user_coupons
                WHERE user_id = %s
                  AND reason = %s
                """,
                (user_id, reward_reason),
            )
            already_issued = int(_row_at(cur.fetchone(), 0, 0) or 0)
            target_count = max(0, int(COUPONS_PER_SIGNUP))
            missing_count = max(0, target_count - already_issued)
            issued_codes = issue_user_coupons(
                cur,
                user_id=user_id,
                count=missing_count,
                reason=reward_reason,
                issued_for_user_id=user_id,
                prefix="WELCOME",
            )
        issued_count = len(issued_codes)
        result["issued_count"] = issued_count
        result["applied"] = issued_count > 0
        if issued_count > 0:
            result["message"] = f"Signup bonus applied: {issued_count} coupons added to your account."
    except Exception:
        return result
    return result


def apply_signup_referral_rewards(conn, referred_user_id: int, referral_input: str) -> dict:
    result = {
        "applied": False,
        "message": "",
        "referrer_name": "",
        "referrer_coupon_count": 0,
    }
    if not REFERRAL_ENABLED or not referred_user_id:
        return result

    raw_ref = str(referral_input or "").strip()
    if not raw_ref:
        return result

    if not ensure_referral_schema(conn):
        result["message"] = "Referral rewards are unavailable right now."
        return result

    try:
        with conn.cursor() as cur:
            normalized = _normalize_referral_code(raw_ref)
            referrer_row = None
            if normalized:
                cur.execute(
                    "SELECT id, username, referral_code FROM users WHERE referral_code=%s LIMIT 1",
                    (normalized,),
                )
                referrer_row = cur.fetchone()

            if not referrer_row:
                cur.execute(
                    "SELECT id, username, referral_code FROM users WHERE username=%s OR email=%s LIMIT 1",
                    (raw_ref, raw_ref.lower()),
                )
                referrer_row = cur.fetchone()
                if referrer_row:
                    normalized = _normalize_referral_code(_row_at(referrer_row, 2, ""))

            if not referrer_row:
                result["message"] = "Referral code was not recognized."
                return result

            referrer_id = int(_row_at(referrer_row, 0, 0) or 0)
            referrer_name = str(_row_at(referrer_row, 1, "") or "")
            result["referrer_name"] = referrer_name
            if not referrer_id or referrer_id == int(referred_user_id):
                result["message"] = "You cannot use your own referral code."
                return result

            cur.execute("SELECT 1 FROM user_referrals WHERE referred_user_id=%s LIMIT 1", (referred_user_id,))
            if cur.fetchone():
                return result

            if not normalized:
                normalized = ensure_user_referral_code(conn, referrer_id, referrer_name)
            cur.execute(
                """
                INSERT INTO user_referrals (referrer_user_id, referred_user_id, referral_code)
                VALUES (%s, %s, %s)
                """,
                (referrer_id, referred_user_id, normalized or _normalize_referral_code(raw_ref)),
            )

            referrer_coupons = issue_user_coupons(
                cur,
                user_id=referrer_id,
                count=COUPONS_PER_REFERRAL,
                reason="Referral reward (friend completed signup)",
                source_user_id=referred_user_id,
                issued_for_user_id=referred_user_id,
                prefix="REFER",
            )

            referrer_coupon_count = len(referrer_coupons)
            result["referrer_coupon_count"] = referrer_coupon_count
            result["applied"] = referrer_coupon_count > 0
            if referrer_coupon_count > 0:
                result["message"] = (
                    f"Referral applied: your referrer received {referrer_coupon_count} coupons."
                )
    except pymysql.err.IntegrityError:
        return result
    except Exception:
        result["message"] = "Referral rewards could not be issued."
    return result


def get_best_active_coupon(conn, user_id: int, subtotal: float):
    if not user_id or subtotal <= 0:
        return None
    if not ensure_coupon_schema(conn):
        return None
    try:
        with conn.cursor() as cur:
            now = _now_utc()
            cur.execute(
                """
                SELECT id, coupon_code, amount, reason
                FROM user_coupons
                WHERE user_id = %s
                  AND status = 'ACTIVE'
                  AND amount > 0
                  AND (expires_at IS NULL OR expires_at >= %s)
                ORDER BY amount DESC, id ASC
                LIMIT 1
                """,
                (user_id, now),
            )
            row = cur.fetchone()
            if not row:
                return None
            raw_amount = float(_row_at(row, 2, 0) or 0)
            if raw_amount <= 0:
                return None
            return {
                "id": int(_row_at(row, 0, 0) or 0),
                "code": str(_row_at(row, 1, "") or ""),
                "amount": min(round(raw_amount, 2), round(float(subtotal), 2)),
                "reason": str(_row_at(row, 3, "") or ""),
            }
    except Exception:
        return None


def calculate_checkout_discount(conn, user_id: int, subtotal: float):
    subtotal = round(float(subtotal or 0), 2)
    if subtotal <= 0:
        return 0.0, "", 0, None

    coupon = get_best_active_coupon(conn, user_id, subtotal)
    if not coupon:
        return 0.0, "", 0, None

    coupon_discount = min(float(coupon.get("amount", 0) or 0), subtotal)
    coupon_discount = round(coupon_discount, 2)
    if coupon_discount <= 0:
        return 0.0, "", 0, None

    coupon["applied_amount"] = coupon_discount
    reason = f"Coupon {coupon.get('code', '')}".strip()
    return coupon_discount, reason, 0, coupon


def consume_user_coupon(cur, coupon_id: int, order_id: int) -> bool:
    if not coupon_id or not order_id:
        return False
    now = _now_utc()
    try:
        cur.execute(
            """
            UPDATE user_coupons
            SET status = 'USED',
                used_order_id = %s,
                used_at = %s
            WHERE id = %s
              AND status = 'ACTIVE'
              AND (expires_at IS NULL OR expires_at >= %s)
            """,
            (order_id, now, coupon_id, now),
        )
        return cur.rowcount > 0
    except Exception:
        return False


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            session["next_url"] = url_for("upload")
            return redirect(url_for("signin"))
        return view(*args, **kwargs)
    return wrapped


@app.errorhandler(Exception)
def handle_exception(exc):
    wants_json = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
    )
    if isinstance(exc, HTTPException):
        app.logger.warning("HTTP error %s: %s", exc.code, exc)
        if wants_json:
            return jsonify(
                ok=False,
                error=exc.name,
                message=exc.description or "We couldn't complete your request.",
                status=exc.code,
            ), exc.code
        try:
            return (
                render_template(
                    "error.html",
                    title="Something went wrong",
                    message=exc.description or "We couldn't complete your request.",
                    status_code=exc.code,
                ),
                exc.code,
            )
        except Exception:
            return (
                "<h1>Something went wrong</h1><p>Please try again.</p>",
                exc.code,
            )
    app.logger.exception("Unhandled error: %s", exc)
    if wants_json:
        return (
            jsonify(
                ok=False,
                error="Internal Server Error",
                message="Service temporarily unavailable. Please try again later.",
                status=500,
            ),
            500,
        )
    try:
        return (
            render_template(
                "error.html",
                title="Failed",
                message="Service temporarily unavailable. Please try again later.",
                status_code=500,
            ),
            500,
        )
    except Exception:
        return "Something went wrong. Please try again.", 500



def verify_password(stored_password, provided_password):
    if stored_password is None or provided_password is None:
        return False, False

    # Hashed password path
    try:
        if check_password_hash(stored_password, provided_password):
            return True, False
    except (ValueError, TypeError):
        pass

    return False, False


def get_user_is_admin(user_id) -> bool:
    if not user_id:
        return False
    if not users_has_is_admin():
        return False
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT is_admin FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            return bool(_row_at(row, 0, 0))
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _fetch_user_email(cur, user_id, fallback: str = "") -> str:
    if not user_id:
        return fallback or ""
    try:
        cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        return str(_row_at(row, 0, fallback or "") or fallback or "")
    except Exception:
        return fallback or ""


def is_admin_identity(user_id, email: str = "") -> bool:
    if email and _normalize_user_name(email) in ADMIN_USERS:
        return True
    return get_user_is_admin(user_id)


def send_login_notifications(user_name, user_email, user_phone):
    if user_phone and validate_phone_number(user_phone):
        try:
            import sms
            sms.send_sms(user_phone, f"Hi {user_name}, you have successfully signed in to Bigoh.")
        except Exception:
            pass

    if user_email and validate_email_format(user_email):
        try:
            forwarded = request.headers.get("X-Forwarded-For", "")
            ip_addr = forwarded.split(",")[0].strip() if forwarded else request.remote_addr
            subject, text_body, html_body = mailer.build_signin_email(
                user_name, ip=ip_addr
            )
            mailer.send_email(user_email, subject, text_body, html_body)
        except Exception:
            pass


def send_signup_confirmation(user_name, user_email):
    if not user_email or not validate_email_format(user_email):
        return
    try:
        subject, text_body, html_body = mailer.build_signup_email(user_name, BUSINESS_NAME)
        mailer.send_email(user_email, subject, text_body, html_body)
    except Exception:
        pass


def validate_password_strength(password):
    if password is None:
        return "Password is required."
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if re.search(r"(.)\1\1", password):
        return "Password cannot contain more than 2 identical characters in a row."

    category_count = 0
    if re.search(r"[a-z]", password):
        category_count += 1
    if re.search(r"[A-Z]", password):
        category_count += 1
    if re.search(r"[0-9]", password):
        category_count += 1
    if re.search(r"[^A-Za-z0-9]", password):
        category_count += 1
    if category_count < 3:
        return "Password must include at least 3 of: lowercase, uppercase, numbers, special characters."
    return None


EMAIL_REGEX = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)


def validate_email_format(email: str) -> bool:
    if not email:
        return False
    if len(email) > 254:
        return False
    if " " in email:
        return False
    if email.count("@") != 1:
        return False
    local_part, domain = email.split("@", 1)
    if not local_part or not domain:
        return False
    if ".." in local_part or ".." in domain:
        return False
    return bool(EMAIL_REGEX.match(email))


def normalize_phone_number(phone: str) -> str:
    if not phone:
        return ""
    cleaned = re.sub(r"[\s\-().]", "", phone)
    if cleaned.startswith("+"):
        digits = cleaned[1:]
        prefix = "+"
    else:
        digits = cleaned
        prefix = ""
    if not digits.isdigit():
        return ""
    if len(digits) < 10 or len(digits) > 15:
        return ""
    return f"{prefix}{digits}"


def validate_phone_number(phone: str) -> bool:
    return bool(normalize_phone_number(phone))


EMAIL_TOKEN_TTL = timedelta(hours=24)
PHONE_OTP_TTL = timedelta(minutes=10)
OTP_MAX_ATTEMPTS = 5
RESET_OTP_TTL = timedelta(minutes=1)
RESET_OTP_MAX_ATTEMPTS = 5


def _now_utc():
    return datetime.utcnow()


def _schema_cache_key(table: str, column: str) -> str:
    return f"SCHEMA_HAS_{table.upper()}_{column.upper()}"


def table_has_column(conn, table: str, column: str) -> bool:
    cache_key = _schema_cache_key(table, column)
    cached = app.config.get(cache_key)
    if cached is not None:
        return cached
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = %s
                  AND COLUMN_NAME = %s
                """,
                (table, column),
            )
            row = cur.fetchone()
            has_col = bool(row and _row_at(row, 0, 0) > 0)
            app.config[cache_key] = has_col
            return has_col
    except Exception:
        app.config[cache_key] = False
        return False


def products_has_seller(conn=None) -> bool:
    cache_key = _schema_cache_key("products", "seller")
    cached = app.config.get(cache_key)
    if cached is not None:
        return cached
    owns_conn = False
    if conn is None:
        conn = get_db_connection()
        owns_conn = True
    try:
        has_col = table_has_column(conn, "products", "seller")
        app.config[cache_key] = has_col
        return has_col
    finally:
        if owns_conn:
            conn.close()


def products_has_color(conn=None) -> bool:
    cache_key = _schema_cache_key("products", "color")
    cached = app.config.get(cache_key)
    if cached is not None:
        return cached
    owns_conn = False
    if conn is None:
        conn = get_db_connection()
        owns_conn = True
    try:
        has_col = table_has_column(conn, "products", "color")
        app.config[cache_key] = has_col
        return has_col
    finally:
        if owns_conn:
            conn.close()


def ensure_products_schema(conn) -> bool:
    try:
        with conn.cursor() as cur:
            if not products_has_seller(conn):
                cur.execute("ALTER TABLE products ADD COLUMN seller VARCHAR(120) NULL")
            if not products_has_color(conn):
                cur.execute("ALTER TABLE products ADD COLUMN color VARCHAR(80) NULL")
        conn.commit()
        app.config[_schema_cache_key("products", "seller")] = True
        app.config[_schema_cache_key("products", "color")] = True
        return True
    except Exception:
        return False


def ensure_user_verification_schema(conn) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_verifications (
                    user_id INT PRIMARY KEY,
                    email_token VARCHAR(120) NULL,
                    email_token_expires DATETIME NULL,
                    email_sent_at DATETIME NULL,
                    phone_otp VARCHAR(10) NULL,
                    phone_otp_expires DATETIME NULL,
                    phone_sent_at DATETIME NULL,
                    phone_otp_attempts INT NOT NULL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX (email_token),
                    INDEX (phone_otp)
                )
                """
            )

            if not table_has_column(conn, "users", "email_verified"):
                cur.execute(
                    "ALTER TABLE users ADD COLUMN email_verified TINYINT(1) NOT NULL DEFAULT 0"
                )
                app.config[_schema_cache_key("users", "email_verified")] = True
            if not table_has_column(conn, "users", "phone_verified"):
                cur.execute(
                    "ALTER TABLE users ADD COLUMN phone_verified TINYINT(1) NOT NULL DEFAULT 0"
                )
                app.config[_schema_cache_key("users", "phone_verified")] = True
            if not table_has_column(conn, "users", "email_verified_at"):
                cur.execute(
                    "ALTER TABLE users ADD COLUMN email_verified_at DATETIME NULL"
                )
                app.config[_schema_cache_key("users", "email_verified_at")] = True
            if not table_has_column(conn, "users", "phone_verified_at"):
                cur.execute(
                    "ALTER TABLE users ADD COLUMN phone_verified_at DATETIME NULL"
                )
                app.config[_schema_cache_key("users", "phone_verified_at")] = True
            if not table_has_column(conn, "user_verifications", "password_reset_otp"):
                cur.execute(
                    "ALTER TABLE user_verifications ADD COLUMN password_reset_otp VARCHAR(10) NULL"
                )
                app.config[_schema_cache_key("user_verifications", "password_reset_otp")] = True
            if not table_has_column(conn, "user_verifications", "password_reset_expires"):
                cur.execute(
                    "ALTER TABLE user_verifications ADD COLUMN password_reset_expires DATETIME NULL"
                )
                app.config[_schema_cache_key("user_verifications", "password_reset_expires")] = True
            if not table_has_column(conn, "user_verifications", "password_reset_attempts"):
                cur.execute(
                    "ALTER TABLE user_verifications ADD COLUMN password_reset_attempts INT NOT NULL DEFAULT 0"
                )
                app.config[_schema_cache_key("user_verifications", "password_reset_attempts")] = True

        conn.commit()
        return True
    except Exception:
        return False


def generate_email_token() -> str:
    return secrets.token_urlsafe(32)


def generate_phone_otp() -> str:
    return f"{secrets.randbelow(1000000):06d}"


def ensure_user_verification_row(cur, user_id: int):
    cur.execute("INSERT IGNORE INTO user_verifications (user_id) VALUES (%s)", (user_id,))


def update_user_verification(cur, user_id: int, fields: dict):
    if not fields:
        return
    columns = ", ".join([f"{key}=%s" for key in fields.keys()])
    params = list(fields.values()) + [user_id]
    cur.execute(
        f"UPDATE user_verifications SET {columns} WHERE user_id=%s",
        params,
    )


def get_verification_state(conn, user_id: int) -> dict:
    state = {"email_verified": True, "phone_verified": True}
    if not table_has_column(conn, "users", "email_verified") or not table_has_column(
        conn, "users", "phone_verified"
    ):
        return state
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT email_verified, phone_verified FROM users WHERE id=%s",
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                return state
            state["email_verified"] = bool(_row_at(row, 0, 1))
            state["phone_verified"] = bool(_row_at(row, 1, 1))
            return state
    except Exception:
        return state


def build_verify_url(token: str) -> str:
    base = os.getenv("APP_BASE_URL", "").rstrip("/")
    if not base:
        base = request.host_url.rstrip("/")
    return f"{base}{url_for('verify_email')}?token={quote(token)}"


def send_email_verification(user_name: str, user_email: str, token: str):
    verify_url = build_verify_url(token)
    subject, text_body, html_body = mailer.build_email_verification(
        user_name, verify_url
    )
    mailer.send_email(user_email, subject, text_body, html_body)


def send_phone_otp_sms(phone: str, otp: str):
    import sms
    sms.send_sms(phone, f"Your Bigoh verification code is {otp}. It expires in 10 minutes.")


def send_password_reset_sms(phone: str, otp: str):
    import sms
    sms.send_sms(phone, f"Your Bigoh password reset code is {otp}. It expires in 1 minute.")


def send_password_reset_email(user_name: str, user_email: str, otp: str):
    subject, text_body, html_body = mailer.build_password_reset_email(
        user_name, otp
    )
    mailer.send_email(user_email, subject, text_body, html_body)


def _random_password() -> str:
    return secrets.token_urlsafe(32)


def _unique_username_from_email(conn, email: str) -> str:
    base = (email.split("@", 1)[0] or "user").strip().lower()
    base = re.sub(r"[^a-z0-9_.-]", "", base) or "user"
    candidate = base
    with conn.cursor() as cur:
        suffix = 0
        while True:
            cur.execute("SELECT 1 FROM users WHERE username=%s LIMIT 1", (candidate,))
            if not cur.fetchone():
                return candidate
            suffix += 1
            candidate = f"{base}{suffix}"



def set_site_message(message, level="warning"):
    session["site_message"] = message
    session["site_message_level"] = level


def users_has_is_admin():
    cached = app.config.get("USERS_HAS_IS_ADMIN")
    if cached is not None:
        return cached

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'users'
                  AND COLUMN_NAME = 'is_admin'
                """
            )
            row = cur.fetchone()
            has_col = bool(row and _row_at(row, 0, 0) > 0)
            app.config["USERS_HAS_IS_ADMIN"] = has_col
            return has_col
    except Exception:
        app.config["USERS_HAS_IS_ADMIN"] = False
        return False
    finally:
        conn.close()


def orders_has_reference():
    cached = app.config.get("ORDERS_HAS_REFERENCE")
    if cached is not None:
        return cached

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'orders'
                  AND COLUMN_NAME = 'order_reference'
                """
            )
            row = cur.fetchone()
            has_col = bool(row and _row_at(row, 0, 0) > 0)
            app.config["ORDERS_HAS_REFERENCE"] = has_col
            return has_col
    except Exception:
        app.config["ORDERS_HAS_REFERENCE"] = False
        return False
    finally:
        conn.close()


def orders_has_discount():
    cached = app.config.get("ORDERS_HAS_DISCOUNT")
    if cached is not None:
        return cached

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'orders'
                  AND COLUMN_NAME = 'discount'
                """
            )
            row = cur.fetchone()
            has_col = bool(row and _row_at(row, 0, 0) > 0)
            app.config["ORDERS_HAS_DISCOUNT"] = has_col
            return has_col
    except Exception:
        app.config["ORDERS_HAS_DISCOUNT"] = False
        return False
    finally:
        conn.close()


def ensure_orders_schema(conn) -> bool:
    try:
        with conn.cursor() as cur:
            if not table_has_column(conn, "orders", "discount"):
                cur.execute(
                    "ALTER TABLE orders ADD COLUMN discount DECIMAL(10,2) NOT NULL DEFAULT 0"
                )
                app.config[_schema_cache_key("orders", "discount")] = True
                app.config["ORDERS_HAS_DISCOUNT"] = True
            if not table_has_column(conn, "orders", "discount_reason"):
                cur.execute(
                    "ALTER TABLE orders ADD COLUMN discount_reason VARCHAR(120) NULL"
                )
                app.config[_schema_cache_key("orders", "discount_reason")] = True
        conn.commit()
        return True
    except Exception:
        return False


def ensure_reviews_table(cur):
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS product_reviews (
                id INT AUTO_INCREMENT PRIMARY KEY,
                product_id INT NOT NULL,
                user_name VARCHAR(80) NOT NULL,
                rating INT NOT NULL,
                comment TEXT NOT NULL,
                review_photo VARCHAR(255) NULL,
                review_photo_approved TINYINT(1) NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_seed TINYINT(1) NOT NULL DEFAULT 0,
                INDEX (product_id)
            )
            """
        )
        conn = cur.connection
        if not table_has_column(conn, "product_reviews", "review_photo"):
            cur.execute(
                "ALTER TABLE product_reviews ADD COLUMN review_photo VARCHAR(255) NULL"
            )
            app.config[_schema_cache_key("product_reviews", "review_photo")] = True
        if not table_has_column(conn, "product_reviews", "review_photo_approved"):
            cur.execute(
                "ALTER TABLE product_reviews ADD COLUMN review_photo_approved TINYINT(1) NOT NULL DEFAULT 0"
            )
            app.config[_schema_cache_key("product_reviews", "review_photo_approved")] = True
        return True
    except Exception:
        return False


def seed_sample_reviews(cur):
    # Samples disabled: do not auto-insert seed reviews.
    return False
    try:
        cur.execute("SELECT COUNT(*) FROM product_reviews")
        row = cur.fetchone()
        if row and _row_at(row, 0, 0) > 0:
            return False

        cur.execute("SELECT product_id FROM products ORDER BY product_id DESC LIMIT 4")
        product_rows = cur.fetchall() or []
        product_ids = [_row_at(r, 0) for r in product_rows]
        if not product_ids:
            return False

        samples = [
            ("Sample review: Delivery was quick and the quality is impressive.", 5),
            ("Sample review: Looks great in person, feels premium for the price.", 4),
            ("Sample review: Packaging was neat and the item matched the photos.", 5),
            ("Sample review: Good value and comfortable to use daily.", 4),
            ("Sample review: Nice finish and solid build, will order again.", 5),
            ("Sample review: Clean design, would recommend to friends.", 4),
        ]
        data = []
        for idx, pid in enumerate(product_ids):
            base = idx * 2
            for offset in range(2):
                text, rating = samples[(base + offset) % len(samples)]
                data.append((pid, f"Sample Buyer {idx + offset + 1}", rating, text, 1))

        if data:
            cur.executemany(
                """
                INSERT INTO product_reviews
                (product_id, user_name, rating, comment, is_seed)
                VALUES (%s, %s, %s, %s, %s)
                """,
                data,
            )
        return True
    except Exception:
        return False


def get_product_reviews(conn, product_id, viewer_name=None):
    reviews = []
    avg_rating = 0.0
    review_count = 0
    has_seed = False
    breakdown = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    try:
        with conn.cursor() as cur:
            if not ensure_reviews_table(cur):
                return reviews, avg_rating, review_count, has_seed, breakdown
            conn.commit()
            viewer_key = _normalize_user_name(viewer_name)

            cur.execute(
                """
                SELECT user_name, rating, comment, created_at, is_seed, review_photo, review_photo_approved
                FROM product_reviews
                WHERE product_id = %s AND TRIM(comment) <> ''
                ORDER BY created_at DESC, id DESC
                """,
                (product_id,),
            )
            raw_reviews = cur.fetchall() or []
            cur.execute(
                """
                SELECT
                    AVG(rating),
                    COUNT(*)
                FROM product_reviews
                WHERE product_id = %s
                """,
                (product_id,),
            )
            row = cur.fetchone()
            avg_rating = float(_row_at(row, 0, 0) or 0)
            review_count = int(_row_at(row, 1, 0) or 0)
            cur.execute(
                """
                SELECT rating, COUNT(*)
                FROM product_reviews
                WHERE product_id = %s
                GROUP BY rating
                """,
                (product_id,),
            )
            for r in cur.fetchall() or []:
                rating_val = int(_row_at(r, 0, 0) or 0)
                if rating_val in breakdown:
                    breakdown[rating_val] = int(_row_at(r, 1, 0) or 0)

            verified_map = {}
            for r in raw_reviews:
                user_name = str(_row_at(r, 0, "") or "").strip()
                if not user_name or user_name in verified_map:
                    continue
                cur.execute("SELECT id FROM users WHERE username=%s LIMIT 1", (user_name,))
                user_row = cur.fetchone()
                user_id = _row_at(user_row, 0, None) if user_row else None
                if not user_id:
                    verified_map[user_name] = False
                    continue
                cur.execute(
                    """
                    SELECT 1
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.order_id
                    WHERE o.user_id = %s
                      AND oi.product_id = %s
                      AND o.status IN ('DELIVERED', 'COMPLETED')
                    LIMIT 1
                    """,
                    (user_id, product_id),
                )
                verified_map[user_name] = bool(cur.fetchone())

            reviews = []
            for r in raw_reviews:
                user_name = _row_at(r, 0, "")
                review_photo = _row_at(r, 5, "")
                photo_approved = bool(_row_at(r, 6, 0))
                is_owner = (
                    bool(viewer_key)
                    and _normalize_user_name(user_name) == viewer_key
                )
                photo_pending = bool(review_photo and (not photo_approved) and is_owner)
                reviews.append(
                    {
                        "user_name": user_name,
                        "rating": _row_at(r, 1, 0),
                        "comment": _row_at(r, 2, ""),
                        "created_at": _row_at(r, 3, ""),
                        "verified": verified_map.get(str(user_name or "").strip(), False),
                        "photo": review_photo if photo_approved else "",
                        "photo_pending": photo_pending,
                        "photo_pending_path": review_photo if photo_pending else "",
                    }
                )
    except Exception:
        return reviews, avg_rating, review_count, has_seed, breakdown

    return reviews, avg_rating, review_count, has_seed, breakdown


def get_ratings_for_products(conn, product_ids):
    if not product_ids:
        return {}
    try:
        with conn.cursor() as cur:
            if not ensure_reviews_table(cur):
                return {}
            conn.commit()

            ids = sorted({int(pid) for pid in product_ids if pid is not None})
            if not ids:
                return {}

            placeholders = ", ".join(["%s"] * len(ids))
            cur.execute(
                f"""
                SELECT
                    product_id,
                    AVG(rating),
                    COUNT(*)
                FROM product_reviews
                WHERE product_id IN ({placeholders})
                GROUP BY product_id
                """,
                tuple(ids),
            )
            rows = cur.fetchall() or []
            rating_map = {}
            for row in rows:
                pid = int(_row_at(row, 0, 0))
                avg_rating = float(_row_at(row, 1, 0) or 0)
                count_rating = int(_row_at(row, 2, 0) or 0)
                rating_map[pid] = {
                    "avg": round(avg_rating, 1),
                    "count": count_rating,
                }
            return rating_map
    except Exception:
        return {}


def ensure_flash_sale_tables(cur):
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS flash_sale_settings (
                id INT PRIMARY KEY,
                is_active TINYINT(1) NOT NULL DEFAULT 0,
                duration_seconds INT NOT NULL DEFAULT 0,
                ends_at DATETIME NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS flash_sale_items (
                product_id INT PRIMARY KEY,
                is_active TINYINT(1) NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute("SELECT COUNT(*) FROM flash_sale_settings WHERE id = 1")
        row = cur.fetchone()
        if not row or _row_at(row, 0, 0) == 0:
            cur.execute(
                """
                INSERT INTO flash_sale_settings (id, is_active, duration_seconds, ends_at)
                VALUES (1, 0, 0, NULL)
                """
            )
        return True
    except Exception:
        return False


def ensure_sponsored_products_table(cur):
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sponsored_products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                product_id INT NOT NULL,
                is_active TINYINT(1) NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_sponsored_product (product_id)
            )
            """
        )
        return True
    except Exception:
        return False


def ensure_stock_alerts_table(cur):
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_alerts (
                product_id INT PRIMARY KEY,
                last_notified DATETIME
            )
            """
        )
        return True
    except Exception:
        return False


def ensure_back_in_stock_alerts_table(cur):
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS back_in_stock_alerts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                product_id INT NOT NULL,
                email VARCHAR(255) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                notified_at DATETIME NULL,
                UNIQUE KEY uniq_back_in_stock (product_id, email),
                INDEX (product_id),
                INDEX (notified_at)
            )
            """
        )
        return True
    except Exception:
        return False


def ensure_abandoned_carts_table(cur):
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS abandoned_carts (
                user_id INT PRIMARY KEY,
                cart_json TEXT NOT NULL,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                notified_at DATETIME NULL,
                INDEX (updated_at)
            )
            """
        )
        return True
    except Exception:
        return False


def ensure_order_review_requests_table(cur):
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS order_review_requests (
                order_id INT PRIMARY KEY,
                notified_at DATETIME NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        return True
    except Exception:
        return False


def ensure_orders_delivery_columns(conn):
    try:
        with conn.cursor() as cur:
            if not table_has_column(conn, "orders", "delivered_at"):
                cur.execute("ALTER TABLE orders ADD COLUMN delivered_at DATETIME NULL")
            if not table_has_column(conn, "orders", "status_updated_at"):
                cur.execute("ALTER TABLE orders ADD COLUMN status_updated_at DATETIME NULL")
        conn.commit()
        return True
    except Exception:
        return False


def ensure_products_visibility_column(conn):
    try:
        with conn.cursor() as cur:
            if not table_has_column(conn, "products", "is_hidden"):
                cur.execute("ALTER TABLE products ADD COLUMN is_hidden TINYINT(1) NOT NULL DEFAULT 0")
        conn.commit()
        return True
    except Exception:
        return False


def ensure_active_sessions_table(cur):
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS active_sessions (
                session_id VARCHAR(64) PRIMARY KEY,
                user_id INT NULL,
                username VARCHAR(80) NULL,
                ip_address VARCHAR(64) NULL,
                user_agent VARCHAR(255) NULL,
                last_seen DATETIME NOT NULL,
                current_path VARCHAR(255) NULL,
                INDEX (last_seen)
            )
            """
        )
        return True
    except Exception:
        return False


def _send_whatsapp_alert(message: str) -> bool:
    if not WHATSAPP_ALERTS_ENABLED or not WHATSAPP_ALERT_WEBHOOK:
        return False
    payload = json.dumps(
        {
            "to": WHATSAPP_ALERT_TO,
            "message": message,
            "token": WHATSAPP_ALERT_TOKEN or None,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        WHATSAPP_ALERT_WEBHOOK,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8):
            return True
    except urllib.error.URLError:
        return False


def _send_whatsapp_message(to: str, message: str) -> bool:
    if not WHATSAPP_RECEIPTS_ENABLED or not WHATSAPP_ALERT_WEBHOOK:
        return False
    to_clean = normalize_phone_number(to)
    if not to_clean:
        return False
    payload = json.dumps(
        {
            "to": to_clean,
            "message": message,
            "token": WHATSAPP_ALERT_TOKEN or None,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        WHATSAPP_ALERT_WEBHOOK,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8):
            return True
    except urllib.error.URLError:
        return False


def send_low_stock_alerts(conn, items):
    if not items or LOW_STOCK_THRESHOLD <= 0:
        return
    try:
        with conn.cursor() as cur:
            if not ensure_stock_alerts_table(cur):
                return
            conn.commit()

            product_ids = [int(_row_at(row, 0, 0)) for row in items if _row_at(row, 0, 0)]
            if not product_ids:
                return

            placeholders = ", ".join(["%s"] * len(product_ids))
            cur.execute(
                f"SELECT product_id, last_notified FROM stock_alerts WHERE product_id IN ({placeholders})",
                tuple(product_ids),
            )
            last_map = {int(_row_at(r, 0, 0)): _row_at(r, 1, None) for r in cur.fetchall() or []}

            now = datetime.now()
            cutoff = now - timedelta(hours=LOW_STOCK_ALERT_INTERVAL_HOURS)
            to_alert = []
            for row in items:
                pid = int(_row_at(row, 0, 0))
                stock = int(_row_at(row, 4, 0) or 0)
                if stock > LOW_STOCK_THRESHOLD:
                    continue
                last = last_map.get(pid)
                if last and last > cutoff:
                    continue
                to_alert.append(row)

            if not to_alert:
                return

            lines = []
            for row in to_alert:
                lines.append(
                    f"- {_row_at(row, 1, 'Item')} (#{_row_at(row, 0, '-')}) stock: {_row_at(row, 4, 0)}"
                )
            subject = f"Low stock alert ({len(to_alert)} items)"
            text_body = "Low stock items:\n" + "\n".join(lines)
            html_body = "<br>".join(lines)
            if LOW_STOCK_EMAIL_TO:
                mailer.send_email(LOW_STOCK_EMAIL_TO, subject, text_body, html_body)

            _send_whatsapp_alert(text_body)

            for row in to_alert:
                pid = int(_row_at(row, 0, 0))
                cur.execute(
                    """
                    INSERT INTO stock_alerts (product_id, last_notified)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE last_notified=VALUES(last_notified)
                    """,
                    (pid, now),
                )
            conn.commit()
    except Exception:
        return


def send_back_in_stock_alerts(conn, product_id: int, product_name: str):
    try:
        with conn.cursor() as cur:
            if not ensure_back_in_stock_alerts_table(cur):
                return
            conn.commit()

            cur.execute(
                """
                SELECT id, email
                FROM back_in_stock_alerts
                WHERE product_id = %s AND notified_at IS NULL
                """,
                (product_id,),
            )
            rows = cur.fetchall() or []
            if not rows:
                return

            link = url_for("single", product_id=product_id, _external=True)
            subject = f"{product_name} is back in stock"
            text_body = (
                f"Good news!\n\n{product_name} is back in stock.\n"
                f"View item: {link}\n\n"
                f"- {BUSINESS_NAME}"
            )
            html_body = (
                f"<p>Good news!</p><p><strong>{product_name}</strong> is back in stock.</p>"
                f"<p><a href=\"{link}\">View item</a></p>"
            )

            sent_ids = []
            for row in rows:
                alert_id = row[0]
                email = row[1]
                ok = mailer.send_email(email, subject, text_body, html_body)
                if ok:
                    sent_ids.append(alert_id)

            if sent_ids:
                placeholders = ", ".join(["%s"] * len(sent_ids))
                cur.execute(
                    f"UPDATE back_in_stock_alerts SET notified_at=NOW() WHERE id IN ({placeholders})",
                    tuple(sent_ids),
                )
                conn.commit()
    except Exception:
        return


def process_back_in_stock_alerts(conn, limit: int = 200):
    try:
        with conn.cursor() as cur:
            if not ensure_back_in_stock_alerts_table(cur):
                return 0
            conn.commit()
            cur.execute(
                """
                SELECT b.product_id, p.product_name
                FROM back_in_stock_alerts b
                JOIN products p ON p.product_id = b.product_id
                WHERE b.notified_at IS NULL
                  AND p.stock > 0
                GROUP BY b.product_id, p.product_name
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall() or []
            for row in rows:
                pid = int(_row_at(row, 0, 0) or 0)
                name = _row_at(row, 1, "") or "Item"
                if pid:
                    send_back_in_stock_alerts(conn, pid, name)
            return len(rows)
    except Exception:
        return 0


def get_sponsored_products(conn, limit: int = 8):
    products = []
    try:
        ensure_products_visibility_column(conn)
        with conn.cursor() as cur:
            if not ensure_sponsored_products_table(cur):
                return products
            conn.commit()
            cur.execute(
                """
                SELECT p.*
                FROM sponsored_products s
                JOIN products p ON p.product_id = s.product_id
                WHERE s.is_active = 1 AND (p.is_hidden IS NULL OR p.is_hidden = 0)
                ORDER BY s.id DESC
                LIMIT %s
                """,
                (limit,),
            )
            products = cur.fetchall() or []
    except Exception:
        return products
    return products

def format_duration(seconds):
    try:
        seconds = int(seconds or 0)
    except (TypeError, ValueError):
        seconds = 0
    if seconds < 0:
        seconds = 0
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}h : {minutes:02d}m : {secs:02d}s"


def get_flash_sale_state(conn):
    state = {
        "active": False,
        "duration_seconds": 0,
        "seconds_left": 0,
        "items": [],
    }
    try:
        ensure_products_visibility_column(conn)
        with conn.cursor() as cur:
            if not ensure_flash_sale_tables(cur):
                return state
            conn.commit()

            cur.execute(
                "SELECT is_active, duration_seconds, ends_at FROM flash_sale_settings WHERE id = 1"
            )
            row = cur.fetchone()
            is_active = bool(_row_at(row, 0, 0)) if row else False
            duration_seconds = int(_row_at(row, 1, 0) or 0) if row else 0
            ends_at = _row_at(row, 2, None) if row else None

            now = datetime.now()
            if is_active and duration_seconds <= 0:
                is_active = False
            if is_active and duration_seconds > 0 and ends_at is None:
                ends_at = now + timedelta(seconds=duration_seconds)
                cur.execute(
                    "UPDATE flash_sale_settings SET ends_at=%s WHERE id=1",
                    (ends_at,),
                )
                conn.commit()

            seconds_left = 0
            if is_active and ends_at:
                seconds_left = int((ends_at - now).total_seconds())
                if seconds_left <= 0:
                    is_active = False
                    seconds_left = 0
                    cur.execute(
                        "UPDATE flash_sale_settings SET is_active=0, ends_at=NULL WHERE id=1"
                    )
                    conn.commit()

            items = []
            if is_active:
                cur.execute(
                    """
                    SELECT p.*
                    FROM flash_sale_items f
                    JOIN products p ON f.product_id = p.product_id
                    WHERE f.is_active = 1 AND (p.is_hidden IS NULL OR p.is_hidden = 0)
                    ORDER BY p.product_id DESC
                    """
                )
                items = cur.fetchall() or []

            state = {
                "active": is_active,
                "duration_seconds": duration_seconds,
                "seconds_left": seconds_left,
                "items": items,
            }
    except Exception:
        return state

    return state

@app.context_processor
def cart_count():
    cart = session.get("cart", {})
    total_items = sum(cart.values())
    msg = session.pop("site_message", None)
    msg_level = session.pop("site_message_level", "warning")
    nav_limit = int(os.getenv("NAV_CATEGORY_LIMIT", "6") or 6)
    try:
        nav_categories = get_category_overview(limit=nav_limit)
    except Exception:
        nav_categories = []
    return dict(
        cart_count=total_items,
        site_message=msg,
        site_message_level=msg_level,
        image_url=image_url,
        nav_categories=nav_categories,
        carousel_categories=nav_categories[:3],
        business_name=BUSINESS_NAME,
        business_address=BUSINESS_ADDRESS,
        business_reg_no=BUSINESS_REG_NO,
        business_reg_body=BUSINESS_REG_BODY,
        support_email=SUPPORT_EMAIL,
        support_email_admin=SUPPORT_EMAIL_ADMIN,
        support_phone=SUPPORT_PHONE,
        support_whatsapp=SUPPORT_WHATSAPP,
        support_hours=SUPPORT_HOURS,
        brand_partners=BRAND_PARTNERS,
        payment_methods=PAYMENT_METHODS,
        payment_logos=PAYMENT_LOGOS,
    )



def get_product(product_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM products WHERE product_id=%s", (product_id,))
    product = cursor.fetchone()
    connection.close()
    return product


def has_verified_purchase(conn, user_id: int, product_id: int) -> bool:
    if not user_id or not product_id:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.order_id
                WHERE o.user_id = %s
                  AND oi.product_id = %s
                  AND o.status IN ('DELIVERED', 'COMPLETED')
                LIMIT 1
                """,
                (user_id, product_id),
            )
            return bool(cur.fetchone())
    except Exception:
        return False



# Default Home route
@app.route("/")
def home():
    try:
        connection = get_db_connection()
    except Exception:
        app.logger.exception("Home route failed to connect to database.")
        return render_template(
            "home.html",
            categories=[],
            new_products=[],
            flash_sales=[],
            flash_sale_active=False,
            flash_sale_seconds=0,
            flash_sale_time_label=format_duration(0),
            flash_sale_duration_seconds=0,
            sponsored_products=[],
            ratings={},
        )

    try:
        category_limit = int(os.getenv("HOME_CATEGORY_LIMIT", "6") or 6)
        items_limit = int(os.getenv("HOME_CATEGORY_ITEMS", "12") or 12)
        new_limit = int(os.getenv("NEW_PRODUCTS_LIMIT", "10") or 10)
        ensure_products_visibility_column(connection)
        cursor = connection.cursor()
        category_rows = get_category_overview(connection, limit=category_limit)
        categories = []
        for row in category_rows:
            category_name = row.get("db_name") or row.get("name", "")
            if not category_name:
                continue
            cursor.execute(
                """
                SELECT * FROM products
                WHERE category = %s AND (is_hidden IS NULL OR is_hidden = 0)
                ORDER BY product_id DESC
                LIMIT %s
                """,
                (category_name, items_limit),
            )
            items = cursor.fetchall() or []
            categories.append(
                {
                    "name": row.get("name") or category_name,
                    "db_name": category_name,
                    "slug": row.get("slug") or slugify_category(category_name),
                    "items": items,
                    "image": row.get("image") or "images/hero.jpg",
                    "count": row.get("count", 0),
                }
            )

        if table_has_column(connection, "products", "created_at"):
            sql5 = "SELECT * FROM products WHERE (is_hidden IS NULL OR is_hidden = 0) ORDER BY created_at DESC, product_id DESC LIMIT %s"
        else:
            sql5 = "SELECT * FROM products WHERE (is_hidden IS NULL OR is_hidden = 0) ORDER BY product_id DESC LIMIT %s"
        cursor = connection.cursor()
        cursor.execute(sql5, (new_limit,))
        new_products = cursor.fetchall()

        sponsored_products = get_sponsored_products(connection, limit=8)

        flash_state = get_flash_sale_state(connection)
        flash_sales = flash_state["items"]
        flash_sale_active = flash_state["active"]
        flash_sale_seconds = flash_state["seconds_left"]
        flash_sale_time_label = format_duration(flash_sale_seconds if flash_sale_active else 0)
        flash_sale_duration_seconds = flash_state["duration_seconds"]

        rating_ids = []
        for group in (new_products, sponsored_products, flash_sales):
            rating_ids.extend([_row_at(row, 0) for row in group] if group else [])
        for category in categories:
            items = category.get("items") or []
            rating_ids.extend([_row_at(row, 0) for row in items] if items else [])
        ratings = get_ratings_for_products(connection, rating_ids)

        return render_template(
            "home.html",
            categories=categories,
            new_products=new_products,
            flash_sales=flash_sales,
            flash_sale_active=flash_sale_active,
            flash_sale_seconds=flash_sale_seconds,
            flash_sale_time_label=flash_sale_time_label,
            flash_sale_duration_seconds=flash_sale_duration_seconds,
            sponsored_products=sponsored_products,
            ratings=ratings,
        )
    finally:
        connection.close()


#Single_item route
@app.route("/single_item/<product_id>")
def single(product_id):
    connection = get_db_connection()
    try:
        sql = "SELECT * FROM products WHERE product_id = %s"
        cursor1 = connection.cursor()
        cursor1.execute(sql, (product_id,))
        product = cursor1.fetchone()
        if not product:
            return redirect(url_for("home"))

        seller = ""
        if products_has_seller(connection):
            seller = _row_at(product, 9, "")

        reviews, avg_rating, review_count, has_seed, rating_breakdown = get_product_reviews(
            connection, product_id, session.get("key")
        )
        can_review = False
        if session.get("username"):
            can_review = has_verified_purchase(connection, session.get("username"), product_id)
    finally:
        connection.close()

    avg_rating = round(avg_rating, 1)
    avg_rating_int = int(round(avg_rating))

    return render_template(
        "single.html",
        product=product,
        seller=seller,
        reviews=reviews,
        avg_rating=avg_rating,
        avg_rating_int=avg_rating_int,
        review_count=review_count,
        has_seed=has_seed,
        rating_breakdown=rating_breakdown,
        can_review=can_review,
    )


@app.route("/product/<int:product_id>/stock-alert", methods=["POST"])
def back_in_stock_alert(product_id):
    email = request.form.get("email", "").strip()
    if not email and session.get("username"):
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                email = _fetch_user_email(cur, session.get("username"), "")
        finally:
            conn.close()

    if not validate_email_format(email):
        set_site_message("Please enter a valid email for restock alerts.", "warning")
        return redirect(url_for("single", product_id=product_id))

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT product_id, product_name, stock FROM products WHERE product_id=%s", (product_id,))
            row = cur.fetchone()
            if not row:
                set_site_message("Product not found.", "warning")
                return redirect(url_for("home"))
            stock = int(_row_at(row, 2, 0) or 0)
            if stock > 0:
                set_site_message("This item is already in stock.", "info")
                return redirect(url_for("single", product_id=product_id))

            if not ensure_back_in_stock_alerts_table(cur):
                set_site_message("Unable to save alert right now.", "danger")
                return redirect(url_for("single", product_id=product_id))

            cur.execute(
                """
                INSERT IGNORE INTO back_in_stock_alerts (product_id, email)
                VALUES (%s, %s)
                """,
                (product_id, email),
            )
        conn.commit()
    finally:
        conn.close()

    set_site_message("We will email you once this item is back in stock.", "success")
    return redirect(url_for("single", product_id=product_id))


@app.route("/product/<int:product_id>/review", methods=["POST"])
@rate_limit("add_product_review")
def add_product_review(product_id):
    if not session.get("key"):
        session["next_url"] = url_for("single", product_id=product_id)
        return redirect(url_for("signin"))

    name = session.get("key", "").strip()
    user_id = session.get("username")
    comment = request.form.get("comment", "").strip()
    try:
        rating = int(request.form.get("rating", "0"))
    except ValueError:
        rating = 0
    photo = request.files.get("photo")

    if not name or not comment or rating not in {1, 2, 3, 4, 5}:
        return redirect(url_for("single", product_id=product_id, review="error"))

    if len(comment) > 500:
        comment = comment[:500]

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if not ensure_reviews_table(cur):
                return redirect(url_for("single", product_id=product_id, review="error"))

            cur.execute("SELECT product_id FROM products WHERE product_id = %s", (product_id,))
            if not cur.fetchone():
                return redirect(url_for("home"))

            if not has_verified_purchase(conn, user_id, product_id):
                return redirect(url_for("single", product_id=product_id, review="verified-only"))

            photo_path = None
            photo_approved = 0
            if photo and photo.filename:
                if not allowed_file(photo.filename):
                    return redirect(url_for("single", product_id=product_id, review="photo-type"))
                photo_path = _cloudinary_upload(photo, "bigoh/reviews") if USE_CLOUDINARY else None
                if not photo_path:
                    try:
                        photo.stream.seek(0)
                    except Exception:
                        pass
                    os.makedirs(REVIEW_UPLOAD_FOLDER, exist_ok=True)
                    ext = os.path.splitext(photo.filename)[1].lower()
                    safe_name = secure_filename(os.path.splitext(photo.filename)[0]) or "review"
                    token = secrets.token_hex(6)
                    filename = f"review_{product_id}_{user_id}_{safe_name}_{token}{ext}"
                    save_path = os.path.join(REVIEW_UPLOAD_FOLDER, filename)
                    photo.save(save_path)
                    compress_image(save_path)
                    photo_path = f"review_photos/{filename}"
                if REVIEW_AUTO_APPROVE or _is_trusted_review_user(name):
                    photo_approved = 1

            cur.execute(
                """
                INSERT INTO product_reviews (product_id, user_name, rating, comment, review_photo, review_photo_approved, is_seed)
                VALUES (%s, %s, %s, %s, %s, %s, 0)
                """,
                (product_id, name, rating, comment, photo_path, photo_approved),
            )
        conn.commit()
    finally:
        conn.close()

    return redirect(url_for("single", product_id=product_id, review="ok"))


# Signup route
@app.route('/signup', methods=['POST', 'GET'])
@rate_limit("signup")
def signup():
    if request.method == 'POST':
        next_from_form = request.form.get("next", "")
        if _is_safe_url(next_from_form):
            session["next_url"] = next_from_form
        try:
            username = request.form['username']
            email = request.form.get('email', '').strip().lower()
            phone_raw = request.form.get('phone', '').strip()
            phone = normalize_phone_number(phone_raw)
            referral_input = (request.form.get("referral_code") or "").strip()
            password1 = request.form['password1']
            password2 = request.form['password2']
            strength_error = validate_password_strength(password1)
            if strength_error:
                return render_template('signup.html', error=strength_error)

            if password1 != password2:
                return render_template('signup.html', error='Password Do Not Match')

            if not email or not validate_email_format(email):
                return render_template('signup.html', error='Please enter a valid email address.', error_field="email")

            if not phone:
                return render_template('signup.html', error='Please enter a valid phone number.', error_field="phone")

            connection = get_db_connection()
            cursor = connection.cursor()
            try:
                ensure_user_verification_schema(connection)
                cursor.execute("SELECT 1 FROM users WHERE email = %s LIMIT 1", (email,))
                if cursor.fetchone():
                    connection.close()
                    return render_template('signup.html', error='Email already registered. Please sign in.', error_field="email")
                cursor.execute("SELECT 1 FROM users WHERE username = %s LIMIT 1", (username,))
                if cursor.fetchone():
                    connection.close()
                    return render_template('signup.html', error='Username already taken. Please choose another.', error_field="username")
            except Exception:
                connection.close()
                return render_template('signup.html', error='Unable to validate account details. Please try again.')

            hashed_password = generate_password_hash(password1)

            now = _now_utc()
            if users_has_is_admin():
                sql = ''' 
                     insert into users(username, password, email, phone, is_admin, email_verified, email_verified_at) 
                     values(%s, %s, %s, %s, %s, %s, %s)
                 '''
                cursor.execute(sql, (username, hashed_password, email, phone, 0, 1, now))
            else:
                sql = ''' 
                     insert into users(username, password, email, phone, email_verified, email_verified_at) 
                     values(%s, %s, %s, %s, %s, %s)
                 '''
                cursor.execute(sql, (username, hashed_password, email, phone, 1, now))

            try:
                connection.commit()
            except pymysql.err.IntegrityError:
                connection.close()
                return render_template('signup.html', error='Email or username already registered.', error_field="both")

            signup_coupon_result = {"applied": False, "message": ""}
            referral_result = {"applied": False, "message": ""}
            user_referral_code = ""
            try:
                user_id = cursor.lastrowid
                user_referral_code = ensure_user_referral_code(connection, user_id, username)
                with connection.cursor() as ver_cur:
                    ver_cur.execute(
                        """
                        UPDATE users
                        SET phone_verified = 1, phone_verified_at = %s
                        WHERE id = %s
                        """,
                        (now, user_id),
                    )
                    ensure_user_verification_row(ver_cur, user_id)
                    update_user_verification(
                        ver_cur,
                        user_id,
                        {
                            "email_token": None,
                            "email_token_expires": None,
                            "email_sent_at": None,
                            "phone_otp": None,
                            "phone_otp_expires": None,
                            "phone_sent_at": None,
                            "phone_otp_attempts": 0,
                        },
                    )
                signup_coupon_result = apply_signup_coupon_rewards(connection, user_id)
                referral_result = apply_signup_referral_rewards(connection, user_id, referral_input)
                connection.commit()
            except Exception:
                connection.close()
                return render_template('signup.html', error='Signup failed. Please try again.')
            connection.close()

            send_signup_confirmation(username, email)
            message_parts = ["Signup successful."]
            if user_referral_code:
                signup_link = f"{request.host_url.rstrip('/')}{url_for('signup')}?ref={quote(user_referral_code)}"
                message_parts.append(
                    f"Your referral code is {user_referral_code}. Share this link to earn coupons: {signup_link}"
                )
            if signup_coupon_result.get("message"):
                message_parts.append(signup_coupon_result["message"])
            if referral_result.get("message"):
                message_parts.append(referral_result["message"])
            next_url = _pop_next_url()
            session.clear()
            session.permanent = False
            session["key"] = username
            session["username"] = user_id
            session["remember_me"] = False
            session["is_admin"] = is_admin_identity(user_id, email)
            if email:
                session["user_email"] = email
            if user_referral_code:
                session["referral_code"] = user_referral_code
            set_site_message(
                " ".join(message_parts),
                "success",
            )
            return redirect(next_url or url_for("home"))
        except Exception:
            app.logger.exception("Signup error")
            return render_template('signup.html', error='Signup failed. Please try again.')
        
    else:
        _remember_next_url()
        prefill_email = (request.args.get("email") or "").strip().lower()
        prefill_name = (request.args.get("name") or "").strip()
        prefill_referral = (request.args.get("ref") or "").strip().upper()
        return render_template(
            'signup.html',
            next_url=session.get("next_url", ""),
            prefill_email=prefill_email,
            prefill_name=prefill_name,
            prefill_referral=prefill_referral,
        )
    

@app.route('/verify-email', methods=['GET'])
def verify_email():
    token = request.args.get("token", "").strip()
    if not token:
        set_site_message("Invalid verification link.", "danger")
        return redirect(url_for("signin"))

    connection = get_db_connection()
    try:
        ensure_user_verification_schema(connection)
        with connection.cursor() as cur:
            cur.execute(
                """
                SELECT v.user_id, v.email_token_expires, u.email, u.phone
                FROM user_verifications v
                JOIN users u ON u.id = v.user_id
                WHERE v.email_token = %s
                """,
                (token,),
            )
            row = cur.fetchone()
            if not row:
                set_site_message("Verification link is invalid or already used.", "danger")
                return redirect(url_for("signin"))
            user_id = _row_at(row, 0)
            expires_at = _row_at(row, 1)
            user_email = _row_at(row, 2, "")
            user_phone = _row_at(row, 3, "")
            if expires_at and expires_at < _now_utc():
                session["verify_user_id"] = user_id
                session["verify_email"] = user_email
                session["verify_phone"] = user_phone
                set_site_message("Verification link has expired. Please request a new one.", "warning")
                return redirect(url_for("verify_phone"))

            cur.execute(
                """
                UPDATE users
                SET email_verified = 1, email_verified_at = %s
                WHERE id = %s
                """,
                (_now_utc(), user_id),
            )
            cur.execute(
                """
                UPDATE user_verifications
                SET email_token = NULL, email_token_expires = NULL
                WHERE user_id = %s
                """,
                (user_id,),
            )
        connection.commit()
    except Exception:
        set_site_message("Unable to verify email. Please try again.", "danger")
        return redirect(url_for("signin"))
    finally:
        connection.close()

    session["verify_user_id"] = user_id
    session["verify_email"] = user_email
    session["verify_phone"] = user_phone

    connection = get_db_connection()
    try:
        verification = get_verification_state(connection, user_id)
    finally:
        connection.close()

    if verification.get("phone_verified"):
        set_site_message("Email verified. You can now sign in.", "success")
        session.pop("verify_user_id", None)
        session.pop("verify_email", None)
        session.pop("verify_phone", None)
        return redirect(url_for("signin"))

    set_site_message("Email verified. Please verify your phone.", "info")
    return redirect(url_for("verify_phone"))


@app.route('/verify-phone', methods=['GET', 'POST'])
def verify_phone():
    user_id = session.get("verify_user_id") or session.get("username")
    if not user_id:
        set_site_message("Please sign in to verify your account.", "warning")
        return redirect(url_for("signin"))

    if request.method == 'GET':
        connection = get_db_connection()
        try:
            ensure_user_verification_schema(connection)
            verification = get_verification_state(connection, user_id)
            if verification.get("email_verified") and verification.get("phone_verified"):
                set_site_message("Your account is already verified.", "info")
                return redirect(url_for("signin"))
        finally:
            connection.close()

    if request.method == 'POST':
        otp = (request.form.get("otp") or "").strip()
        if not otp:
            return render_template('verify_phone.html', error="Please enter the verification code.")

        connection = get_db_connection()
        try:
            ensure_user_verification_schema(connection)
            with connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT phone_otp, phone_otp_expires, phone_otp_attempts
                    FROM user_verifications
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
                if not row:
                    return render_template('verify_phone.html', error="Verification code not found. Please resend.")
                code = _row_at(row, 0, "")
                expires_at = _row_at(row, 1)
                attempts = int(_row_at(row, 2, 0) or 0)
                if attempts >= OTP_MAX_ATTEMPTS:
                    return render_template('verify_phone.html', error="Too many attempts. Please resend a new code.")
                if expires_at and expires_at < _now_utc():
                    return render_template('verify_phone.html', error="Verification code has expired. Please resend.")
                if otp != code:
                    attempts += 1
                    update_user_verification(
                        cur,
                        user_id,
                        {"phone_otp_attempts": attempts},
                    )
                    connection.commit()
                    return render_template('verify_phone.html', error="Incorrect code. Please try again.")

                cur.execute(
                    """
                    UPDATE users
                    SET phone_verified = 1, phone_verified_at = %s
                    WHERE id = %s
                    """,
                    (_now_utc(), user_id),
                )
                update_user_verification(
                    cur,
                    user_id,
                    {
                        "phone_otp": None,
                        "phone_otp_expires": None,
                        "phone_otp_attempts": 0,
                    },
                )
            connection.commit()
        except Exception:
            return render_template('verify_phone.html', error="Unable to verify phone. Please try again.")
        finally:
            connection.close()

        set_site_message("Phone verified. You can now sign in.", "success")
        session.pop("verify_user_id", None)
        session.pop("verify_email", None)
        session.pop("verify_phone", None)
        return redirect(url_for("signin"))

    return render_template('verify_phone.html')


@app.route('/verify/resend', methods=['POST'])
def resend_verification():
    user_id = session.get("verify_user_id") or session.get("username")
    if not user_id:
        set_site_message("Please sign in to resend verification.", "warning")
        return redirect(url_for("signin"))

    connection = get_db_connection()
    try:
        ensure_user_verification_schema(connection)
        verification = get_verification_state(connection, user_id)
        with connection.cursor() as cur:
            cur.execute(
                "SELECT username, email, phone FROM users WHERE id=%s",
                (user_id,),
            )
            user_row = cur.fetchone() or ()
            user_name = _row_at(user_row, 0, "Customer")
            user_email = _row_at(user_row, 1, "")
            user_phone = _row_at(user_row, 2, "")
            session["verify_email"] = user_email
            session["verify_phone"] = user_phone

            ensure_user_verification_row(cur, user_id)
            now = _now_utc()
            email_token = None
            phone_otp = None
            if not verification.get("email_verified"):
                email_token = generate_email_token()
                update_user_verification(
                    cur,
                    user_id,
                    {
                        "email_token": email_token,
                        "email_token_expires": now + EMAIL_TOKEN_TTL,
                        "email_sent_at": now,
                    },
                )
            if not verification.get("phone_verified"):
                phone_otp = generate_phone_otp()
                update_user_verification(
                    cur,
                    user_id,
                    {
                        "phone_otp": phone_otp,
                        "phone_otp_expires": now + PHONE_OTP_TTL,
                        "phone_sent_at": now,
                        "phone_otp_attempts": 0,
                    },
                )
        connection.commit()
    except Exception:
        set_site_message("Unable to resend verification right now.", "danger")
        return redirect(url_for("verify_phone"))
    finally:
        connection.close()

    try:
        if email_token and validate_email_format(user_email or ""):
            send_email_verification(user_name, user_email, email_token)
    except Exception:
        pass
    try:
        if phone_otp and validate_phone_number(user_phone or ""):
            send_phone_otp_sms(user_phone, phone_otp)
    except Exception:
        pass

    set_site_message("Verification messages sent.", "info")
    return redirect(url_for("verify_phone"))


@app.route('/add-phone', methods=['GET', 'POST'])
def add_phone():
    user_id = session.get("pending_user_id")
    if not user_id:
        set_site_message("Please sign in to continue.", "warning")
        return redirect(url_for("signin"))

    if request.method == 'POST':
        phone_raw = (request.form.get("phone") or "").strip()
        phone = normalize_phone_number(phone_raw)
        if not phone:
            return render_template('add_phone.html', error="Please enter a valid phone number.")

        connection = get_db_connection()
        user_referral_code = ""
        try:
            ensure_user_verification_schema(connection)
            with connection.cursor() as cur:
                now = _now_utc()
                cur.execute(
                    """
                    UPDATE users
                    SET phone=%s, phone_verified=1, phone_verified_at=%s
                    WHERE id=%s
                    """,
                    (phone, now, user_id),
                )
                ensure_user_verification_row(cur, user_id)
                update_user_verification(
                    cur,
                    user_id,
                    {
                        "phone_otp": None,
                        "phone_otp_expires": None,
                        "phone_sent_at": None,
                        "phone_otp_attempts": 0,
                    },
                )
                cur.execute(
                    "SELECT username, email, email_verified FROM users WHERE id=%s",
                    (user_id,),
                )
                user_row = cur.fetchone()
            user_referral_code = ensure_user_referral_code(
                connection, user_id, _row_at(user_row, 0, "")
            )
            connection.commit()
        except Exception:
            return render_template('add_phone.html', error="Unable to save phone number. Please try again.")
        finally:
            connection.close()

        username = _row_at(user_row, 0, "Customer") if user_row else "Customer"
        email = _row_at(user_row, 1, "") if user_row else ""
        email_verified = bool(_row_at(user_row, 2, 0)) if user_row else False

        if email and not email_verified:
            token = None
            try:
                connection = get_db_connection()
                ensure_user_verification_schema(connection)
                with connection.cursor() as cur:
                    token = generate_email_token()
                    ensure_user_verification_row(cur, user_id)
                    update_user_verification(
                        cur,
                        user_id,
                        {
                            "email_token": token,
                            "email_token_expires": now + EMAIL_TOKEN_TTL,
                            "email_sent_at": now,
                        },
                    )
                connection.commit()
            except Exception:
                pass
            finally:
                if connection:
                    connection.close()
            try:
                send_email_verification(username, email, token)
            except Exception:
                pass
            set_site_message("Phone saved. Please verify your email to continue.", "warning")
            session.pop("pending_user_id", None)
            return redirect(url_for("signin"))

        remember_me = session.pop("google_remember_me", False)
        next_url = _pop_next_url()
        session.clear()
        session.permanent = remember_me
        session["key"] = username
        session["username"] = user_id
        session["remember_me"] = remember_me
        session["is_admin"] = is_admin_identity(user_id, email)
        if email:
            session["user_email"] = email
        if user_referral_code:
            session["referral_code"] = user_referral_code
        session.pop("pending_user_id", None)
        set_site_message("Phone saved successfully.", "success")
        return redirect(next_url or url_for("home"))

    return render_template('add_phone.html')


@app.route('/login/google')
def login_google():
    if not os.getenv("GOOGLE_CLIENT_ID") or not os.getenv("GOOGLE_CLIENT_SECRET"):
        set_site_message("Google sign-in is not configured yet.", "warning")
        return redirect(url_for("signin"))

    remember_me = request.args.get("remember") == "1"
    intent = (request.args.get("intent") or "signin").strip().lower()
    if intent not in {"signin", "signup"}:
        intent = "signin"
    google_referral_input = (request.args.get("ref") or "").strip()
    session["google_remember_me"] = remember_me
    session["google_intent"] = intent
    if google_referral_input:
        session["google_referral_code"] = google_referral_input
    else:
        session.pop("google_referral_code", None)
    _remember_next_url()
    redirect_uri = url_for("auth_google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route('/auth/google/callback')
def auth_google_callback():
    try:
        token = oauth.google.authorize_access_token()
    except Exception:
        set_site_message("Google sign-in failed. Please try again.", "danger")
        return redirect(url_for("signin"))

    userinfo = token.get("userinfo")
    if not userinfo:
        try:
            userinfo = oauth.google.userinfo()
        except Exception:
            userinfo = None

    if not userinfo:
        set_site_message("Unable to read Google profile.", "danger")
        return redirect(url_for("signin"))

    email = (userinfo.get("email") or "").strip().lower()
    email_verified = bool(userinfo.get("email_verified"))
    display_name = (userinfo.get("name") or "").strip() or email.split("@", 1)[0]

    if not email or not validate_email_format(email):
        set_site_message("Google account email is invalid.", "danger")
        return redirect(url_for("signin"))

    intent = session.pop("google_intent", "signin")
    google_referral_input = (session.pop("google_referral_code", "") or "").strip()
    connection = get_db_connection()
    created_new = False
    signup_coupon_result = {"applied": False, "message": ""}
    referral_result = {"applied": False, "message": ""}
    user_referral_code = ""
    try:
        ensure_user_verification_schema(connection)
        with connection.cursor() as cur:
            cur.execute(
                "SELECT id, username, email, phone FROM users WHERE email=%s LIMIT 1",
                (email,),
            )
            row = cur.fetchone()
            if row:
                user_id = _row_at(row, 0)
                username = _row_at(row, 1, display_name)
                phone = _row_at(row, 3, "") or ""
            else:
                if intent != "signup":
                    set_site_message("No account found for that Google email. Please sign up first.", "warning")
                    params = {"email": email, "name": display_name}
                    return redirect(url_for("signup", **params))

                username = _unique_username_from_email(connection, email)
                password_hash = generate_password_hash(_random_password())
                if users_has_is_admin():
                    cur.execute(
                        """
                        INSERT INTO users (username, password, email, phone, is_admin, email_verified, email_verified_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            username,
                            password_hash,
                            email,
                            "",
                            0,
                            1 if email_verified else 0,
                            _now_utc() if email_verified else None,
                        ),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO users (username, password, email, phone, email_verified, email_verified_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            username,
                            password_hash,
                            email,
                            "",
                            1 if email_verified else 0,
                            _now_utc() if email_verified else None,
                        ),
                    )
                user_id = cur.lastrowid
                phone = ""
                created_new = True
            user_referral_code = ensure_user_referral_code(connection, user_id, username)
            if created_new:
                signup_coupon_result = apply_signup_coupon_rewards(connection, user_id)
                referral_result = apply_signup_referral_rewards(connection, user_id, google_referral_input)
            connection.commit()
    except Exception:
        set_site_message("Unable to complete Google sign-in.", "danger")
        return redirect(url_for("signin"))
    finally:
        connection.close()

    if created_new:
        send_signup_confirmation(username, email)

    if not phone:
        session["pending_user_id"] = user_id
        msg = "Please add your phone number to continue."
        if signup_coupon_result.get("message"):
            msg = f"{signup_coupon_result['message']} {msg}"
        if referral_result.get("message"):
            msg = f"{referral_result['message']} {msg}"
        set_site_message(msg, "warning")
        return redirect(url_for("add_phone"))

    if not email_verified:
        token = None
        try:
            connection = get_db_connection()
            ensure_user_verification_schema(connection)
            with connection.cursor() as cur:
                now = _now_utc()
                token = generate_email_token()
                ensure_user_verification_row(cur, user_id)
                update_user_verification(
                    cur,
                    user_id,
                    {
                        "email_token": token,
                        "email_token_expires": now + EMAIL_TOKEN_TTL,
                        "email_sent_at": now,
                    },
                )
            connection.commit()
        except Exception:
            pass
        finally:
            if connection:
                connection.close()
        try:
            send_email_verification(username, email, token)
        except Exception:
            pass
        session["verify_user_id"] = user_id
        session["verify_email"] = email
        set_site_message("Please verify your email to continue.", "warning")
        return redirect(url_for("verify_phone"))

    remember_me = session.pop("google_remember_me", False)
    next_url = _pop_next_url()
    session.clear()
    session.permanent = remember_me
    session["key"] = username
    session["username"] = user_id
    session["remember_me"] = remember_me
    session["is_admin"] = is_admin_identity(user_id, email)
    if email:
        session["user_email"] = email
    if user_referral_code:
        session["referral_code"] = user_referral_code
    session.pop("pending_user_id", None)

    return redirect(next_url or url_for("home"))

#Signin route
@app.route('/signin', methods=['POST', 'GET'])
@rate_limit("signin")
def signin():
    if request.method == 'POST':
        next_from_form = request.form.get("next", "")
        if _is_safe_url(next_from_form):
            session["next_url"] = next_from_form
        username = request.form['username']
        password = request.form['password']
        remember_me = request.form.get("remember_me") == "1"

        connection = get_db_connection()

        has_admin_col = users_has_is_admin()
        if has_admin_col:
            sql = '''
               select id, username, password, email, phone, is_admin from users where username = %s
            '''
        else:
            sql = '''
               select * from users where username = %s
            '''
        cursor = connection.cursor()
        identifier = (username or "").strip()
        email_identifier = identifier.lower()
        phone_identifier = normalize_phone_number(identifier)
        if has_admin_col:
            sql = '''
               select id, username, password, email, phone, is_admin
               from users
               where username = %s or email = %s or phone = %s
               limit 1
            '''
            cursor.execute(sql, (identifier, email_identifier, phone_identifier))
        else:
            sql = '''
               select * from users
               where username = %s or email = %s or phone = %s
               limit 1
            '''
            cursor.execute(sql, (identifier, email_identifier, phone_identifier))
        user = cursor.fetchone()

        if not user:
            connection.close()
            return render_template('signin.html', error='Invalid Credentials')
        else:
            stored_password = user[2] if len(user) > 2 else None
            valid, needs_update = verify_password(stored_password, password)
            if not valid:
                connection.close()
                return render_template('signin.html', error='Invalid Credentials')

            if needs_update:
                new_hash = generate_password_hash(password)
                cursor.execute("UPDATE users SET password=%s WHERE id=%s", (new_hash, user[0]))
                connection.commit()

            # assume users table columns are (id, username, password, email, phone)
            user_id = user[0]
            user_name = user[1]
            user_email = user[3] if len(user) > 3 else None
            user_phone = user[4] if len(user) > 4 else None
            user_email = _fetch_user_email(cursor, user_id, user_email or "")
            user_referral_code = ""
            try:
                user_referral_code = ensure_user_referral_code(connection, user_id, user_name)
                connection.commit()
            except Exception:
                user_referral_code = ""

            connection.close()
            session.clear()
            session.permanent = remember_me
            session['key'] = user_name
            # `pay_on_delivery` expects `session['username']` to contain user id
            session['username'] = user_id
            session['remember_me'] = remember_me

            session["is_admin"] = is_admin_identity(user_id, user_email or "")
            if user_email:
                session["user_email"] = user_email
            if user_referral_code:
                session["referral_code"] = user_referral_code

            send_login_notifications(user_name, user_email, user_phone)

            next_url = _pop_next_url()
            return redirect(next_url or url_for("home"))

    else:
        _remember_next_url()
        success = "Registered Successfully, You can Signin Now" if request.args.get("signup") == "success" else ""
        return render_template('signin.html', success=success, next_url=session.get("next_url", ""))    


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        phone_raw = (request.form.get("phone") or "").strip()
        email_raw = (request.form.get("email") or "").strip().lower()
        phone = normalize_phone_number(phone_raw)
        email = email_raw if validate_email_format(email_raw) else ""
        if not phone and not email:
            return render_template('forgot_password.html', error="Please enter a valid phone number or email.")

        connection = get_db_connection()
        user_id = None
        user_name = "Customer"
        try:
            ensure_user_verification_schema(connection)
            with connection.cursor() as cur:
                if phone:
                    cur.execute("SELECT id, username FROM users WHERE phone=%s LIMIT 1", (phone,))
                else:
                    cur.execute("SELECT id, username FROM users WHERE email=%s LIMIT 1", (email,))
                row = cur.fetchone()
                if row:
                    user_id = _row_at(row, 0)
                    user_name = _row_at(row, 1, "Customer")
                    otp = generate_phone_otp()
                    ensure_user_verification_row(cur, user_id)
                    update_user_verification(
                        cur,
                        user_id,
                        {
                            "password_reset_otp": otp,
                            "password_reset_expires": _now_utc() + RESET_OTP_TTL,
                            "password_reset_attempts": 0,
                        },
                    )
            connection.commit()
        except Exception:
            return render_template('forgot_password.html', error="Unable to start password reset. Please try again.")
        finally:
            connection.close()

        if user_id:
            try:
                if phone:
                    send_password_reset_sms(phone, otp)
                else:
                    send_password_reset_email(user_name, email, otp)
            except Exception:
                pass
            session["reset_user_id"] = user_id
            session["reset_phone"] = phone
            session["reset_email"] = email
            session["reset_channel"] = "email" if email else "phone"
        set_site_message("If that account is registered, we sent a reset code.", "info")
        return redirect(url_for("reset_password"))

    return render_template('forgot_password.html')


@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    user_id = session.get("reset_user_id")
    phone = session.get("reset_phone")
    email = session.get("reset_email")
    channel = session.get("reset_channel")
    if request.method == 'POST':
        otp = (request.form.get("otp") or "").strip()
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")

        if not user_id:
            return render_template('reset_password.html', error="Please request a reset code first.")

        if not otp:
            return render_template('reset_password.html', error="Please enter the reset code.")

        if password1 != password2:
            return render_template('reset_password.html', error="Passwords do not match.")

        strength_error = validate_password_strength(password1)
        if strength_error:
            return render_template('reset_password.html', error=strength_error)

        connection = get_db_connection()
        try:
            ensure_user_verification_schema(connection)
            with connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT password_reset_otp, password_reset_expires, password_reset_attempts
                    FROM user_verifications
                    WHERE user_id=%s
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
                if not row:
                    return render_template('reset_password.html', error="Reset code not found. Please request a new one.")
                stored_otp = _row_at(row, 0, "")
                expires_at = _row_at(row, 1)
                attempts = int(_row_at(row, 2, 0) or 0)
                if attempts >= RESET_OTP_MAX_ATTEMPTS:
                    return render_template('reset_password.html', error="Too many attempts. Please request a new code.")
                if expires_at and expires_at < _now_utc():
                    return render_template('reset_password.html', error="Reset code has expired. Please request a new one.")
                if otp != stored_otp:
                    update_user_verification(
                        cur,
                        user_id,
                        {"password_reset_attempts": attempts + 1},
                    )
                    connection.commit()
                    return render_template('reset_password.html', error="Incorrect code. Please try again.")

                new_hash = generate_password_hash(password1)
                cur.execute("UPDATE users SET password=%s WHERE id=%s", (new_hash, user_id))
                update_user_verification(
                    cur,
                    user_id,
                    {
                        "password_reset_otp": None,
                        "password_reset_expires": None,
                        "password_reset_attempts": 0,
                    },
                )
            connection.commit()
        except Exception:
            return render_template('reset_password.html', error="Unable to reset password. Please try again.")
        finally:
            connection.close()

        session.pop("reset_user_id", None)
        session.pop("reset_phone", None)
        session.pop("reset_email", None)
        session.pop("reset_channel", None)
        set_site_message("Password updated. You can now sign in.", "success")
        return redirect(url_for("signin"))

    return render_template('reset_password.html', phone=phone or "", email=email or "", channel=channel or "")


#logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/signin')



#Search route
@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return render_template(
            "search_results.html",
            results=[],
            q="",
            ratings={},
            did_you_mean="",
            category_filter="",
            brand_filter="",
            availability_filter="all",
            min_rating=0,
            sort="popularity",
            min_price=0,
            max_price=0,
            base_min=0,
            base_max=0,
            categories=[],
            brands=[],
            total_results=0,
        )

    category_filter = request.args.get("category", "").strip()
    brand_filter = request.args.get("brand", "").strip()
    availability_filter = request.args.get("availability", "all").strip().lower()
    min_rating_raw = request.args.get("min_rating", "").strip()
    sort = request.args.get("sort", "popularity").strip()
    min_price_raw = request.args.get("min_price", "").strip()
    max_price_raw = request.args.get("max_price", "").strip()
    if availability_filter not in {"all", "in_stock"}:
        availability_filter = "all"
    if sort not in {"popularity", "newest", "price_asc", "price_desc", "rating"}:
        sort = "popularity"

    results = advanced_product_search(q, limit=120)
    tokenized_query = _tokenize_search(q)
    plain_query_tokens = [tok for tok in tokenized_query if ":" not in tok]
    corrected_tokens = [
        _canonical_search_term(tok)
        for tok in plain_query_tokens
    ]
    corrected_query = " ".join([tok for tok in corrected_tokens if tok]).strip()
    plain_query_text = " ".join(plain_query_tokens).strip()
    did_you_mean = ""
    if (
        plain_query_text
        and corrected_query
        and corrected_query != _normalize_search_text(plain_query_text)
    ):
        did_you_mean = corrected_query
    if not results and corrected_query and corrected_query != _normalize_search_text(q):
        results = advanced_product_search(corrected_query, limit=120)

    def summarize_categories(rows):
        summary = {}
        for row in rows:
            raw_category = str(_row_at(row, 2, "") or "").strip()
            if not raw_category:
                continue
            display_name = _bucket_category_for_search(raw_category)
            key = _normalize_search_text(display_name)
            if not key:
                continue
            if key not in summary:
                summary[key] = {"name": display_name, "count": 0}
            summary[key]["count"] += 1
        return summary

    def summarize_brands(rows):
        summary = {}
        for row in rows:
            brand_name = str(_row_at(row, 3, "") or "").strip()
            key = _normalize_search_text(brand_name)
            if not key:
                continue
            if key not in summary:
                summary[key] = {"name": brand_name, "count": 0}
            summary[key]["count"] += 1
        return summary

    category_summary = summarize_categories(results)
    brand_summary = summarize_brands(results)

    if not category_filter and not brand_filter:
        query_tokens = tokenized_query
        if len(query_tokens) == 1:
            lowered = _canonical_search_term(query_tokens[0])
            category_match = category_summary.get(lowered)
            brand_match = brand_summary.get(lowered)
            if category_match:
                category_filter = category_match["name"]
            elif brand_match:
                brand_filter = brand_match["name"]

    if category_filter:
        category_key = _normalize_search_text(category_filter)
        category_filter = category_summary.get(category_key, {}).get(
            "name", _bucket_category_for_search(category_filter)
        )
    if brand_filter:
        brand_key = _normalize_search_text(brand_filter)
        brand_filter = brand_summary.get(brand_key, {}).get("name", brand_filter)

    try:
        min_price = float(min_price_raw) if min_price_raw else None
    except ValueError:
        min_price = None
    try:
        max_price = float(max_price_raw) if max_price_raw else None
    except ValueError:
        max_price = None
    try:
        min_rating = float(min_rating_raw) if min_rating_raw else 0.0
    except ValueError:
        min_rating = 0.0

    conn = get_db_connection()
    try:
        ratings = get_ratings_for_products(conn, [_row_at(row, 0) for row in results])
    finally:
        conn.close()

    filtered = []
    selected_brand_key = _normalize_search_text(brand_filter)
    for row in results:
        if category_filter:
            category = str(_row_at(row, 2, "") or "").strip()
            if not _category_matches_filter(category, category_filter):
                continue
        if selected_brand_key:
            brand = str(_row_at(row, 3, "") or "").strip()
            if _normalize_search_text(brand) != selected_brand_key:
                continue
        price_val = _row_at(row, 4, None)
        try:
            price_val = float(price_val)
        except (TypeError, ValueError):
            price_val = None
        if min_price is not None and price_val is not None and price_val < min_price:
            continue
        if max_price is not None and price_val is not None and price_val > max_price:
            continue
        if availability_filter == "in_stock":
            stock_val = _row_at(row, 5, 0)
            try:
                stock_val = int(stock_val)
            except (TypeError, ValueError):
                stock_val = 0
            if stock_val <= 0:
                continue
        if min_rating:
            avg_rating = ratings.get(_row_at(row, 0), {}).get("avg", 0)
            if avg_rating < min_rating:
                continue
        filtered.append(row)

    filtered_category_summary = summarize_categories(filtered)
    filtered_brand_summary = summarize_brands(filtered)

    prices = []
    for row in filtered:
        price_val = _row_at(row, 4, None)
        if price_val is not None:
            try:
                prices.append(float(price_val))
            except (TypeError, ValueError):
                pass

    base_min = int(min(prices)) if prices else 0
    base_max = int(max(prices)) if prices else 0

    if min_price is None:
        min_price = base_min
    if max_price is None:
        max_price = base_max

    def _price_value(row):
        try:
            return float(_row_at(row, 4, 0) or 0)
        except (TypeError, ValueError):
            return 0

    if sort == "newest":
        filtered.sort(key=lambda r: _row_at(r, 0, 0), reverse=True)
    elif sort == "price_asc":
        filtered.sort(key=_price_value)
    elif sort == "price_desc":
        filtered.sort(key=_price_value, reverse=True)
    elif sort == "rating":
        filtered.sort(
            key=lambda r: ratings.get(_row_at(r, 0), {}).get("avg", 0),
            reverse=True,
        )

    categories = sorted(
        list(filtered_category_summary.values()),
        key=lambda item: item["name"].lower(),
    )
    brands = sorted(
        list(filtered_brand_summary.values()),
        key=lambda item: item["name"].lower(),
    )

    return render_template(
        "search_results.html",
        results=filtered,
        q=q,
        ratings=ratings,
        did_you_mean=did_you_mean,
        category_filter=category_filter,
        brand_filter=brand_filter,
        availability_filter=availability_filter,
        min_rating=min_rating,
        sort=sort,
        min_price=min_price,
        max_price=max_price,
        base_min=base_min,
        base_max=base_max,
        categories=categories,
        brands=brands,
        total_results=len(filtered),
    )


@app.route("/search_suggestions")
def search_suggestions():
    q = request.args.get("q", "").strip()
    if len(_normalize_search_text(q)) < 2:
        return jsonify([])
    rows = advanced_product_search(q, limit=8, include_rating_score=False)

    suggestions = []
    for row in rows:
        suggestions.append(
            {
                "id": _row_at(row, 0),
                "name": _row_at(row, 1),
                "category": _bucket_category_for_search(_row_at(row, 2)),
                "brand": _row_at(row, 3),
                "price": _row_at(row, 4),
                "stock": _row_at(row, 5),
                "description": _row_at(row, 6),
                "image_url": image_url(_row_at(row, 7)),
            }
        )
    return jsonify(suggestions)


SEARCH_TERM_CORRECTIONS = {
    "whatch": "watch",
    "whaches": "watches",
    "whatches": "watches",
    "wath": "watch",
    "wacth": "watch",
    "wach": "watch",
    "watche": "watch",
    "waches": "watches",
    "jersy": "jersey",
    "jerze": "jersey",
    "snikers": "sneakers",
    "shoos": "shoes",
    "electrical": "electricals",
    "electronic": "electricals",
    "cleaner": "cleaning",
}


def _normalize_search_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    lowered = re.sub(r"[^\w\s-]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _canonical_search_term(term: str) -> str:
    cleaned = _normalize_search_text(term)
    if not cleaned:
        return ""
    return SEARCH_TERM_CORRECTIONS.get(cleaned, cleaned)


def _tokenize_search(raw_query):
    raw = str(raw_query or "").strip()
    if not raw:
        return []
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = re.findall(r'"[^"]+"|\S+', raw)
    cleaned = []
    for tok in tokens:
        tok = str(tok or "").strip()
        if tok:
            cleaned.append(tok)
    return cleaned


def _expand_synonyms(term: str) -> list:
    synonyms = {
        "jersey": ["jerseys", "kit", "football shirt", "soccer jersey"],
        "jerseys": ["jersey", "kit", "football shirt", "soccer jersey"],
        "kit": ["jersey", "jerseys", "football shirt", "soccer jersey"],
        "watch": ["watches", "wristwatch", "timepiece", "waches", "ladies watch", "men watch"],
        "watches": ["watch", "wristwatch", "timepiece", "waches", "ladies watch", "men watch"],
        "waches": ["watch", "watches", "wristwatch", "timepiece", "ladies watch", "men watch"],
        "ladies watch": ["women watch", "female watch", "watch"],
        "men watch": ["male watch", "gents watch", "watch"],
        "ladies": ["women", "female"],
        "men": ["male", "gents"],
        "cleaning": ["cleaner", "cleaners", "detergent"],
        "cleaners": ["cleaner", "cleaning", "detergent"],
        "sneakers": ["shoes", "trainers"],
        "shoes": ["sneakers", "trainers"],
        "electricals": ["electronics", "electrical"],
    }
    key = _canonical_search_term(term)
    return synonyms.get(key, [])


def _search_term_variants(term: str) -> list:
    initial = _canonical_search_term(term)
    if not initial:
        return []

    variants = []
    pending = [initial]
    seen = set()
    while pending:
        item = _canonical_search_term(pending.pop(0))
        if not item or item in seen:
            continue
        seen.add(item)
        variants.append(item)
        for synonym in _expand_synonyms(item):
            normalized_syn = _canonical_search_term(synonym)
            if normalized_syn and normalized_syn not in seen:
                pending.append(normalized_syn)

    expanded = list(variants)
    for item in variants:
        if " " in item:
            continue
        if len(item) > 3 and item.endswith("s"):
            singular = _canonical_search_term(item[:-1])
            if singular:
                expanded.append(singular)
        elif len(item) > 2 and not item.endswith("s"):
            plural = _canonical_search_term(f"{item}s")
            if plural:
                expanded.append(plural)

    deduped = []
    seen = set()
    for item in expanded:
        normalized = _canonical_search_term(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _bucket_category_for_search(name: str) -> str:
    raw = str(name or "").strip()
    lowered = _normalize_search_text(raw)
    if lowered in WATCH_CATEGORY_ALIASES:
        return "Watches"
    if lowered in {"jersey", "jerseys", "kit"}:
        return "Jerseys"
    if lowered in {"shoe", "shoes", "sneaker", "sneakers"}:
        return "Shoes"
    return normalize_category_label(raw)


def _category_matches_filter(category_name: str, selected_filter: str) -> bool:
    selected_norm = _normalize_search_text(selected_filter)
    if not selected_norm:
        return True

    category_norm = _normalize_search_text(category_name)
    category_bucket = _normalize_search_text(_bucket_category_for_search(category_name))
    selected_bucket = _normalize_search_text(_bucket_category_for_search(selected_filter))

    return (
        category_norm == selected_norm
        or category_bucket == selected_norm
        or category_norm == selected_bucket
        or category_bucket == selected_bucket
    )


def advanced_product_search(raw_query, limit=60, include_rating_score=True):
    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        limit_value = 60
    limit_value = max(1, min(limit_value, 120))

    tokens = _tokenize_search(raw_query)
    raw_lower = _normalize_search_text(raw_query)

    general_terms = []
    name_terms = []
    brand_terms = []
    category_terms = []
    desc_terms = []
    min_price = None
    max_price = None

    field_aliases = {
        "cat": "category",
        "title": "name",
        "desc": "description",
    }

    for tok in tokens:
        if ":" in tok:
            key, value = tok.split(":", 1)
            key = field_aliases.get(_normalize_search_text(key), _normalize_search_text(key))
            value = str(value or "").strip()
            if not value:
                continue
            if key in {"category"}:
                category_terms.append(value)
                continue
            if key in {"brand"}:
                brand_terms.append(value)
                continue
            if key in {"name"}:
                name_terms.append(value)
                continue
            if key in {"description"}:
                desc_terms.append(value)
                continue
            if key in {"price"}:
                if "-" in value:
                    low, high = value.split("-", 1)
                    try:
                        min_price = float(low)
                        max_price = float(high)
                    except ValueError:
                        pass
                else:
                    try:
                        min_price = float(value)
                    except ValueError:
                        pass
                continue
            if key in {"min"}:
                try:
                    min_price = float(value)
                except ValueError:
                    pass
                continue
            if key in {"max"}:
                try:
                    max_price = float(value)
                except ValueError:
                    pass
                continue

        general_terms.append(tok)

    where_parts = []
    where_params = []
    score_parts = []
    score_params = []

    def add_score(expr, term, weight, like=False):
        if like:
            score_parts.append(f"CASE WHEN {expr} LIKE %s THEN {weight} ELSE 0 END")
            score_params.append(term)
        else:
            score_parts.append(f"CASE WHEN {expr} = %s THEN {weight} ELSE 0 END")
            score_params.append(term)

    def add_soundex_score(field, term, weight):
        score_parts.append(f"CASE WHEN SOUNDEX({field}) = SOUNDEX(%s) THEN {weight} ELSE 0 END")
        score_params.append(term)

    def add_term_group(term, weight_scale=1.0):
        variants = _search_term_variants(term)[:10]
        term_clauses = []
        for v in variants:
            like = f"%{v}%"
            term_clauses.append(
                "(LOWER(p.product_name) LIKE %s OR LOWER(p.category) LIKE %s OR LOWER(p.brand) LIKE %s OR LOWER(p.description) LIKE %s "
                "OR SOUNDEX(p.product_name)=SOUNDEX(%s) OR SOUNDEX(p.brand)=SOUNDEX(%s) OR SOUNDEX(p.category)=SOUNDEX(%s))"
            )
            where_params.extend([like, like, like, like, v, v, v])

            w = weight_scale
            add_score("LOWER(p.product_name)", v, int(28 * w))
            add_score("LOWER(p.brand)", v, int(18 * w))
            add_score("LOWER(p.category)", v, int(14 * w))
            add_score("LOWER(p.product_name)", f"{v}%", int(20 * w), like=True)
            add_score("LOWER(p.brand)", f"{v}%", int(12 * w), like=True)
            add_score("LOWER(p.category)", f"{v}%", int(9 * w), like=True)
            add_score("LOWER(p.product_name)", f"%{v}%", int(9 * w), like=True)
            add_score("LOWER(p.brand)", f"%{v}%", int(6 * w), like=True)
            add_score("LOWER(p.category)", f"%{v}%", int(4 * w), like=True)
            add_score("LOWER(p.description)", f"%{v}%", int(2 * w), like=True)
            add_soundex_score("p.product_name", v, int(4 * w))
            add_soundex_score("p.brand", v, int(3 * w))
            add_soundex_score("p.category", v, int(2 * w))
        if term_clauses:
            where_parts.append("(" + " OR ".join(term_clauses) + ")")

    for term in general_terms:
        add_term_group(term, 1.0)

    for term in name_terms:
        add_term_group(term, 1.25)

    for term in brand_terms:
        add_term_group(term, 1.15)

    for term in category_terms:
        add_term_group(term, 1.1)

    for term in desc_terms:
        add_term_group(term, 0.9)

    def add_query_scores(query_text: str, weight_scale=1.0):
        query_term = _canonical_search_term(query_text)
        if not query_term:
            return
        w = weight_scale
        add_score("LOWER(p.product_name)", query_term, int(40 * w))
        add_score("LOWER(p.brand)", query_term, int(24 * w))
        add_score("LOWER(p.category)", query_term, int(18 * w))
        add_score("LOWER(p.product_name)", f"{query_term}%", int(26 * w), like=True)
        add_score("LOWER(p.brand)", f"{query_term}%", int(14 * w), like=True)
        add_score("LOWER(p.category)", f"{query_term}%", int(10 * w), like=True)
        add_score("LOWER(p.product_name)", f"%{query_term}%", int(12 * w), like=True)
        add_score("LOWER(p.brand)", f"%{query_term}%", int(7 * w), like=True)
        add_score("LOWER(p.category)", f"%{query_term}%", int(5 * w), like=True)
        add_score("LOWER(p.description)", f"%{query_term}%", int(3 * w), like=True)
        add_soundex_score("p.product_name", query_term, int(5 * w))
        add_soundex_score("p.brand", query_term, int(3 * w))
        add_soundex_score("p.category", query_term, int(2 * w))

    if raw_lower:
        add_query_scores(raw_lower, 1.0)

    corrected_tokens = [
        _canonical_search_term(tok)
        for tok in (general_terms + name_terms + brand_terms + category_terms + desc_terms)
    ]
    corrected_query = " ".join([tok for tok in corrected_tokens if tok]).strip()
    if corrected_query and corrected_query != raw_lower:
        add_query_scores(corrected_query, 0.9)

    where_parts.append("(p.is_hidden IS NULL OR p.is_hidden = 0)")

    if min_price is not None and max_price is not None:
        where_parts.append("p.price BETWEEN %s AND %s")
        where_params.extend([min_price, max_price])
    elif min_price is not None:
        where_parts.append("p.price >= %s")
        where_params.append(min_price)
    elif max_price is not None:
        where_parts.append("p.price <= %s")
        where_params.append(max_price)

    where_clause = ""
    if where_parts:
        where_clause = "WHERE " + " AND ".join(where_parts)

    score_expr = "0"
    if score_parts:
        score_expr = " + ".join(score_parts)

    if include_rating_score:
        sql = f"""
            SELECT
                p.*,
                COALESCE(r.avg_rating, 0) AS avg_rating,
                COALESCE(r.rating_count, 0) AS rating_count,
                ({score_expr})
                  + (COALESCE(r.avg_rating, 0) * 2)
                  + (LEAST(COALESCE(r.rating_count, 0), 50) * 0.2) AS score
            FROM products p
            LEFT JOIN (
                SELECT product_id, AVG(rating) AS avg_rating, COUNT(*) AS rating_count
                FROM product_reviews
                WHERE is_seed = 0
                GROUP BY product_id
            ) r ON r.product_id = p.product_id
            {where_clause}
            ORDER BY score DESC, rating_count DESC, avg_rating DESC, p.product_name
            LIMIT {limit_value}
        """
    else:
        sql = f"""
            SELECT
                p.*,
                0 AS avg_rating,
                0 AS rating_count,
                ({score_expr}) AS score
            FROM products p
            {where_clause}
            ORDER BY score DESC, p.product_name
            LIMIT {limit_value}
        """

    connection = get_db_connection()
    try:
        ensure_products_visibility_column(connection)
        with connection.cursor() as cursor:
            cursor.execute(sql, score_params + where_params)
            return cursor.fetchall()
    finally:
        connection.close()



def _parse_price(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_products_by_category(category_name, filters=None):
    filters = filters or {}
    connection = get_db_connection()
    try:
        ensure_products_schema(connection)
        ensure_products_visibility_column(connection)
        if isinstance(category_name, (list, tuple, set)):
            category_names = [str(name).strip() for name in category_name if str(name).strip()]
        else:
            category_names = [str(category_name).strip()] if str(category_name).strip() else []
        if not category_names:
            return [], [], []
        category_keys = [name.lower() for name in category_names]
        brand = filters.get("brand", "")
        color = filters.get("color", "")
        min_price = _parse_price(filters.get("min_price"))
        max_price = _parse_price(filters.get("max_price"))
        availability = filters.get("availability", "")

        if len(category_keys) == 1:
            category_clause = "LOWER(category) = %s"
        else:
            placeholders = ", ".join(["%s"] * len(category_keys))
            category_clause = f"LOWER(category) IN ({placeholders})"
        where_parts = [category_clause, "(is_hidden IS NULL OR is_hidden = 0)"]
        params = category_keys[:]
        if brand:
            where_parts.append("brand = %s")
            params.append(brand)
        if color:
            where_parts.append("color = %s")
            params.append(color)
        if min_price is not None:
            where_parts.append("price >= %s")
            params.append(min_price)
        if max_price is not None:
            where_parts.append("price <= %s")
            params.append(max_price)
        if availability == "in_stock":
            where_parts.append("stock > 0")

        where_clause = " AND ".join(where_parts)

        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM products WHERE {where_clause} ORDER BY product_id DESC",
                params,
            )
            products = cursor.fetchall()

            cursor.execute(
                """
                SELECT DISTINCT brand
                FROM products
                WHERE {category_clause} AND brand IS NOT NULL AND brand <> ''
                ORDER BY brand
                """.format(category_clause=category_clause),
                category_keys,
            )
            brands = [row[0] for row in cursor.fetchall() or []]

            if products_has_color(connection):
                cursor.execute(
                    """
                    SELECT DISTINCT color
                    FROM products
                    WHERE {category_clause} AND color IS NOT NULL AND color <> ''
                    ORDER BY color
                    """.format(category_clause=category_clause),
                    category_keys,
                )
                colors = [row[0] for row in cursor.fetchall() or []]
            else:
                colors = []

        return products, brands, colors
    finally:
        connection.close()


@app.route("/categories")
def categories():
    categories_list = get_category_overview()
    return render_template("categories.html", categories=categories_list)


@app.route("/category/men-watch")
def category_men_watch():
    filters = {
        "brand": request.args.get("brand", "").strip(),
        "color": request.args.get("color", "").strip(),
        "min_price": request.args.get("min_price", "").strip(),
        "max_price": request.args.get("max_price", "").strip(),
        "availability": request.args.get("availability", "").strip(),
    }
    products, brands, colors = get_products_by_category("Men Watch", filters)
    conn = get_db_connection()
    try:
        ratings = get_ratings_for_products(conn, [_row_at(row, 0) for row in products])
    finally:
        conn.close()
    return render_template(
        "category_men_watch.html",
        products=products,
        ratings=ratings,
        brands=brands,
        colors=colors,
        filters=filters,
    )


@app.route("/category/ladies-watch")
def category_ladies_watch():
    filters = {
        "brand": request.args.get("brand", "").strip(),
        "color": request.args.get("color", "").strip(),
        "min_price": request.args.get("min_price", "").strip(),
        "max_price": request.args.get("max_price", "").strip(),
        "availability": request.args.get("availability", "").strip(),
    }
    products, brands, colors = get_products_by_category("Ladies Watch", filters)
    conn = get_db_connection()
    try:
        ratings = get_ratings_for_products(conn, [_row_at(row, 0) for row in products])
    finally:
        conn.close()
    return render_template(
        "category_ladies_watch.html",
        products=products,
        ratings=ratings,
        brands=brands,
        colors=colors,
        filters=filters,
    )


@app.route("/category/jersey")
def category_jersey():
    filters = {
        "brand": request.args.get("brand", "").strip(),
        "color": request.args.get("color", "").strip(),
        "min_price": request.args.get("min_price", "").strip(),
        "max_price": request.args.get("max_price", "").strip(),
        "availability": request.args.get("availability", "").strip(),
    }
    products, brands, colors = get_products_by_category("Jersey", filters)
    conn = get_db_connection()
    try:
        ratings = get_ratings_for_products(conn, [_row_at(row, 0) for row in products])
    finally:
        conn.close()
    return render_template(
        "category_jersey.html",
        products=products,
        ratings=ratings,
        brands=brands,
        colors=colors,
        filters=filters,
    )


@app.route("/category/cleaning")
def category_cleaning():
    filters = {
        "brand": request.args.get("brand", "").strip(),
        "color": request.args.get("color", "").strip(),
        "min_price": request.args.get("min_price", "").strip(),
        "max_price": request.args.get("max_price", "").strip(),
        "availability": request.args.get("availability", "").strip(),
    }
    products, brands, colors = get_products_by_category("Cleaning", filters)
    conn = get_db_connection()
    try:
        ratings = get_ratings_for_products(conn, [_row_at(row, 0) for row in products])
    finally:
        conn.close()
    return render_template(
        "category_cleaning.html",
        products=products,
        ratings=ratings,
        brands=brands,
        colors=colors,
        filters=filters,
    )


@app.route("/category/shoes")
def category_shoes():
    filters = {
        "brand": request.args.get("brand", "").strip(),
        "color": request.args.get("color", "").strip(),
        "min_price": request.args.get("min_price", "").strip(),
        "max_price": request.args.get("max_price", "").strip(),
        "availability": request.args.get("availability", "").strip(),
    }
    products, brands, colors = get_products_by_category("Shoes", filters)
    conn = get_db_connection()
    try:
        ratings = get_ratings_for_products(conn, [_row_at(row, 0) for row in products])
    finally:
        conn.close()
    return render_template(
        "category_shoes.html",
        products=products,
        ratings=ratings,
        brands=brands,
        colors=colors,
        filters=filters,
    )


@app.route("/category/electricals")
def category_electricals():
    filters = {
        "brand": request.args.get("brand", "").strip(),
        "color": request.args.get("color", "").strip(),
        "min_price": request.args.get("min_price", "").strip(),
        "max_price": request.args.get("max_price", "").strip(),
        "availability": request.args.get("availability", "").strip(),
    }
    products, brands, colors = get_products_by_category("Electricals", filters)
    conn = get_db_connection()
    try:
        ratings = get_ratings_for_products(conn, [_row_at(row, 0) for row in products])
    finally:
        conn.close()
    return render_template(
        "category_electricals.html",
        products=products,
        ratings=ratings,
        brands=brands,
        colors=colors,
        filters=filters,
    )

@app.route("/category/<slug>")
def category_dynamic(slug):
    categories_list = get_category_overview()
    category = next((c for c in categories_list if c.get("slug") == slug), None)
    if not category:
        set_site_message("Category not found.", "warning")
        return redirect(url_for("categories"))

    filters = {
        "brand": request.args.get("brand", "").strip(),
        "color": request.args.get("color", "").strip(),
        "min_price": request.args.get("min_price", "").strip(),
        "max_price": request.args.get("max_price", "").strip(),
        "availability": request.args.get("availability", "").strip(),
    }
    category_db_name = category.get("db_name") or category["name"]
    products, brands, colors = get_products_by_category(category_db_name, filters)
    conn = get_db_connection()
    try:
        ratings = get_ratings_for_products(conn, [_row_at(row, 0) for row in products])
    finally:
        conn.close()
    return render_template(
        "category_generic.html",
        category_name=category["name"],
        category_slug=slug,
        category_image=category.get("image") or "images/hero.jpg",
        products=products,
        ratings=ratings,
        brands=brands,
        colors=colors,
        filters=filters,
    )


@app.route("/flash-sales")
def flash_sales_page():
    conn = get_db_connection()
    try:
        flash_state = get_flash_sale_state(conn)
        ratings = get_ratings_for_products(
            conn, [_row_at(row, 0) for row in flash_state["items"]]
        )
    finally:
        conn.close()

    flash_sales = flash_state["items"] if flash_state["active"] else []
    flash_sale_active = flash_state["active"]
    flash_sale_seconds = flash_state["seconds_left"]
    flash_sale_time_label = format_duration(
        flash_sale_seconds if flash_sale_active else 0
    )

    return render_template(
        "flash_sales.html",
        flash_sales=flash_sales,
        flash_sale_active=flash_sale_active,
        flash_sale_seconds=flash_sale_seconds,
        flash_sale_time_label=flash_sale_time_label,
        ratings=ratings,
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/returns")
def returns_refunds():
    return render_template("returns_refunds.html")


@app.route("/shipping")
def shipping_policy():
    return render_template("shipping_policy.html")


@app.route("/privacy")
def privacy_policy():
    return render_template("privacy_policy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


ORDER_STATUS_LABELS = {
    "PENDING": "Pending",
    "PROCESSING": "Processing",
    "DELIVERED": "Delivered",
    "COMPLETED": "Completed",
    "CANCELLED": "Cancelled",
}
ORDER_STATUS_SEQUENCE = tuple(ORDER_STATUS_LABELS.keys())
ORDER_STATUS_TABS = [("ALL", "All")] + [
    (code, ORDER_STATUS_LABELS[code]) for code in ORDER_STATUS_SEQUENCE
]


@app.route("/my-orders")
def my_orders():
    if not session.get("username"):
        session["next_url"] = url_for("my_orders")
        return redirect(url_for("signin"))

    user_id = session.get("username")
    status_filter = (request.args.get("status") or "ALL").strip().upper()
    allowed = {code for code, _ in ORDER_STATUS_TABS}
    if status_filter not in allowed:
        status_filter = "ALL"

    has_ref = orders_has_reference()
    has_discount = orders_has_discount()
    status_counts = {code: 0 for code, _ in ORDER_STATUS_TABS}
    orders_view = []

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, COUNT(*)
                FROM orders
                WHERE user_id = %s
                GROUP BY status
                """,
                (user_id,),
            )
            for row in cur.fetchall() or []:
                status_key = str(_row_at(row, 0, "") or "").upper()
                status_counts[status_key] = int(_row_at(row, 1, 0) or 0)

            cols = ["order_id", "status", "subtotal", "created_at", "location", "payment_method"]
            if has_ref:
                cols.append("order_reference")
            if has_discount:
                cols.extend(["discount", "discount_reason"])

            sql = f"SELECT {', '.join(cols)} FROM orders WHERE user_id = %s"
            params = [user_id]
            if status_filter != "ALL":
                sql += " AND status = %s"
                params.append(status_filter)
            sql += " ORDER BY order_id DESC LIMIT 120"
            cur.execute(sql, tuple(params))
            orders = cur.fetchall() or []

            order_ids = [int(_row_at(row, 0, 0) or 0) for row in orders if _row_at(row, 0, None)]
            items_by_order = {}
            first_product_id = {}
            image_map = {}

            if order_ids:
                placeholders = ", ".join(["%s"] * len(order_ids))
                cur.execute(
                    f"""
                    SELECT order_id, product_id, product_name, unit_price, quantity, line_total
                    FROM order_items
                    WHERE order_id IN ({placeholders})
                    ORDER BY id ASC
                    """,
                    tuple(order_ids),
                )
                for row in cur.fetchall() or []:
                    oid = int(_row_at(row, 0, 0) or 0)
                    if not oid:
                        continue
                    items_by_order.setdefault(oid, []).append(row)
                    if oid not in first_product_id:
                        first_product_id[oid] = int(_row_at(row, 1, 0) or 0)

                product_ids = sorted({pid for pid in first_product_id.values() if pid})
                if product_ids:
                    p_ph = ", ".join(["%s"] * len(product_ids))
                    cur.execute(
                        f"SELECT product_id, image_url FROM products WHERE product_id IN ({p_ph})",
                        tuple(product_ids),
                    )
                    for row in cur.fetchall() or []:
                        image_map[int(_row_at(row, 0, 0) or 0)] = _row_at(row, 1, "")

            ref_idx = 6 if has_ref else None
            discount_idx = (7 if has_ref else 6) if has_discount else None
            for row in orders:
                order_id = int(_row_at(row, 0, 0) or 0)
                status = str(_row_at(row, 1, "PENDING") or "PENDING").upper()
                amount = float(_row_at(row, 2, 0) or 0)
                created_at = _row_at(row, 3, None)
                location = str(_row_at(row, 4, "") or "")
                payment_method = str(_row_at(row, 5, "") or "")

                reference = f"ZC-{order_id:06d}"
                if has_ref:
                    custom_ref = _row_at(row, ref_idx, "")
                    if custom_ref:
                        reference = custom_ref

                discount = float(_row_at(row, discount_idx, 0) or 0) if has_discount else 0.0
                discount_reason = (
                    str(_row_at(row, discount_idx + 1, "") or "") if has_discount else ""
                )

                item_rows = items_by_order.get(order_id, [])
                first_item = item_rows[0] if item_rows else None
                first_name = str(_row_at(first_item, 2, "No items") or "No items")
                first_qty = int(_row_at(first_item, 4, 0) or 0)
                first_total = float(_row_at(first_item, 5, 0) or 0)
                extra_lines = max(len(item_rows) - 1, 0)
                total_units = sum(int(_row_at(it, 4, 0) or 0) for it in item_rows)
                pid = first_product_id.get(order_id)
                image_path = image_map.get(pid, "images/hero.jpg")

                orders_view.append(
                    {
                        "order_id": order_id,
                        "reference": reference,
                        "status": status,
                        "amount": amount,
                        "created_at": created_at,
                        "location": location,
                        "payment_method": payment_method,
                        "discount": discount,
                        "discount_reason": discount_reason,
                        "first_item_name": first_name,
                        "first_item_qty": first_qty,
                        "first_item_total": first_total,
                        "extra_lines": extra_lines,
                        "total_units": total_units,
                        "image_path": image_path,
                    }
                )
    finally:
        conn.close()

    status_counts["ALL"] = sum(v for k, v in status_counts.items() if k != "ALL")
    return render_template(
        "my_orders.html",
        orders=orders_view,
        status_filter=status_filter,
        status_tabs=ORDER_STATUS_TABS,
        status_counts=status_counts,
    )


@app.route("/my-referrals")
def my_referrals():
    if not session.get("username"):
        session["next_url"] = url_for("my_referrals")
        return redirect(url_for("signin"))

    user_id = session.get("username")
    user_name = session.get("key", "")
    referral_code = session.get("referral_code", "")
    referral_link = ""
    referrals = []
    total_referrals = 0
    active_coupons = 0
    used_coupons = 0
    total_coupons_earned = 0
    error = ""

    conn = get_db_connection()
    try:
        if not ensure_referral_schema(conn):
            error = "Referral program is currently unavailable."
        else:
            referral_code = ensure_user_referral_code(conn, user_id, user_name)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        u.id,
                        u.username,
                        u.email,
                        u.phone,
                        r.created_at
                    FROM user_referrals r
                    JOIN users u ON u.id = r.referred_user_id
                    WHERE r.referrer_user_id = %s
                    ORDER BY r.created_at DESC
                    """,
                    (user_id,),
                )
                referrals = cur.fetchall() or []
                total_referrals = len(referrals)

                now = _now_utc()
                cur.execute(
                    """
                    SELECT
                        SUM(CASE WHEN status='ACTIVE' AND (expires_at IS NULL OR expires_at >= %s) THEN 1 ELSE 0 END),
                        SUM(CASE WHEN status='USED' THEN 1 ELSE 0 END),
                        COUNT(*)
                    FROM user_coupons
                    WHERE user_id = %s
                    """,
                    (now, user_id),
                )
                stats = cur.fetchone()
                active_coupons = int(_row_at(stats, 0, 0) or 0)
                used_coupons = int(_row_at(stats, 1, 0) or 0)
                total_coupons_earned = int(_row_at(stats, 2, 0) or 0)
            conn.commit()
    finally:
        conn.close()

    if referral_code:
        session["referral_code"] = referral_code
        referral_link = f"{request.host_url.rstrip('/')}{url_for('signup')}?ref={quote(referral_code)}"

    return render_template(
        "my_referrals.html",
        referral_code=referral_code,
        referral_link=referral_link,
        referrals=referrals,
        total_referrals=total_referrals,
        active_coupons=active_coupons,
        used_coupons=used_coupons,
        total_coupons_earned=total_coupons_earned,
        error=error,
    )


@app.route("/admin/review-photos")
@admin_required
def admin_review_photos():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if not ensure_reviews_table(cur):
                reviews = []
            else:
                cur.execute(
                    """
                    SELECT r.id, r.product_id, p.product_name, r.user_name, r.rating,
                           r.comment, r.review_photo, r.created_at
                    FROM product_reviews r
                    JOIN products p ON p.product_id = r.product_id
                    WHERE r.review_photo IS NOT NULL
                      AND r.review_photo <> ''
                      AND (r.review_photo_approved IS NULL OR r.review_photo_approved = 0)
                    ORDER BY r.created_at DESC
                    """
                )
                reviews = cur.fetchall() or []
    finally:
        conn.close()
    return render_template("admin_review_photos.html", reviews=reviews)


@app.route("/admin/reviews/<int:review_id>/approve", methods=["POST"])
@admin_required
def admin_review_photo_approve(review_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE product_reviews SET review_photo_approved = 1 WHERE id = %s",
                (review_id,),
            )
        conn.commit()
    finally:
        conn.close()
    return redirect(url_for("admin_review_photos"))


@app.route("/admin/reviews/<int:review_id>/reject", methods=["POST"])
@admin_required
def admin_review_photo_reject(review_id):
    photo_path = None
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT review_photo FROM product_reviews WHERE id = %s", (review_id,))
            row = cur.fetchone()
            photo_path = _row_at(row, 0, None) if row else None
            cur.execute(
                "UPDATE product_reviews SET review_photo = NULL, review_photo_approved = 0 WHERE id = %s",
                (review_id,),
            )
        conn.commit()
    finally:
        conn.close()

    if photo_path and not str(photo_path).startswith(("http://", "https://")):
        try:
            base_root = app.config.get("UPLOAD_ROOT", UPLOAD_ROOT)
            full_path = os.path.join(base_root, photo_path)
            if os.path.isfile(full_path):
                os.remove(full_path)
        except Exception:
            pass
    return redirect(url_for("admin_review_photos"))


#Add to cart route
def get_product(product_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM products WHERE product_id=%s", (product_id,))
    product = cursor.fetchone()
    connection.close()
    return product

@app.route("/add_to_cart/<int:product_id>", methods=["POST"])
@rate_limit("add_to_cart")
def add_to_cart(product_id):
    wants_json = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
    )
    try:
        qty = int(request.form.get("qty", 1))
    except ValueError:
        qty = 1
    if qty <= 0:
        qty = 1

    product = get_product(product_id)
    if not product:
        message = "Product not found."
        if wants_json:
            return jsonify({"ok": False, "message": message, "level": "danger"}), 404
        set_site_message(message, "danger")
        return redirect(request.referrer or url_for("home"))

    try:
        stock = int(product[5]) if product[5] is not None else 0
    except (ValueError, TypeError):
        stock = 0

    if stock <= 0:
        message = "This item is currently out of stock."
        if wants_json:
            return jsonify({"ok": False, "message": message, "level": "danger"}), 409
        set_site_message(message, "danger")
        return redirect(request.referrer or url_for("home"))

    cart = session.get("cart", {})  # {"12": 2, "15": 1}
    pid = str(product_id)
    current = int(cart.get(pid, 0))
    desired = current + qty

    if desired > stock:
        cart[pid] = stock
        message = f"Only {stock} left in stock. Cart updated."
        level = "warning"
    else:
        cart[pid] = desired
        message = "Added to cart."
        level = "success"

    session["cart"] = cart
    total_items = sum(cart.values())
    user_id = session.get("username")
    if user_id:
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                if ensure_abandoned_carts_table(cur):
                    cur.execute(
                        """
                        INSERT INTO abandoned_carts (user_id, cart_json, notified_at)
                        VALUES (%s, %s, NULL)
                        ON DUPLICATE KEY UPDATE cart_json=VALUES(cart_json), notified_at=NULL
                        """,
                        (user_id, json.dumps(cart)),
                    )
                    conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    if wants_json:
        return jsonify(
            {
                "ok": True,
                "cart_count": total_items,
                "message": message,
                "level": level,
            }
        )
    set_site_message(message, level)
    return redirect(request.referrer or url_for("home"))


#Cart page route
@app.route("/cart")
def cart():
    cart = session.get("cart", {})
    items = []
    grand_total = 0

    for pid, qty in cart.items():
        product = get_product(int(pid))
        if not product:
            continue

        price = float(product[4])  # adjust index if your price is different
        total = price * int(qty)
        grand_total += total

        items.append({
            "product": product,
            "qty": int(qty),
            "total": total
        })

    discount = 0.0
    discount_reason = ""
    repeat_count = 0
    if session.get("username"):
        conn = get_db_connection()
        try:
            discount, discount_reason, repeat_count, _ = calculate_checkout_discount(
                conn, session.get("username"), grand_total
            )
        finally:
            conn.close()
    total_after_discount = max(0.0, grand_total - discount)

    return render_template(
        "cart.html",
        items=items,
        grand_total=grand_total,
        discount=discount,
        discount_reason=discount_reason,
        repeat_count=repeat_count,
        total_after_discount=total_after_discount,
        loyalty_repeat_min=LOYALTY_REPEAT_ORDERS_MIN,
        loyalty_discount_pct=LOYALTY_REPEAT_DISCOUNT_PCT,
    )


@app.route("/update_cart/<int:product_id>", methods=["POST"])
def update_cart(product_id):
    action = request.form.get("action", "set")  # inc | dec | set
    try:
        qty = int(request.form.get("qty", 1))
    except ValueError:
        qty = 1

    cart = session.get("cart", {})
    pid = str(product_id)
    current = int(cart.get(pid, 0))

    product = get_product(product_id)
    if not product:
        cart.pop(pid, None)
        session["cart"] = cart
        set_site_message("Product no longer exists and was removed.", "warning")
        return redirect(url_for("cart"))

    try:
        stock = int(product[5]) if product[5] is not None else 0
    except (ValueError, TypeError):
        stock = 0

    if action == "inc":
        new_qty = current + 1
    elif action == "dec":
        new_qty = current - 1
    else:
        new_qty = qty

    if stock <= 0:
        cart.pop(pid, None)
        session["cart"] = cart
        set_site_message("This item is out of stock and was removed.", "danger")
        return redirect(url_for("cart"))

    if new_qty > stock:
        new_qty = stock
        set_site_message(f"Only {stock} left in stock. Quantity adjusted.", "warning")

    if new_qty <= 0:
        cart.pop(pid, None)
    else:
        cart[pid] = new_qty

    session["cart"] = cart
    user_id = session.get("username")
    if user_id:
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                if ensure_abandoned_carts_table(cur):
                    cur.execute(
                        """
                        INSERT INTO abandoned_carts (user_id, cart_json, notified_at)
                        VALUES (%s, %s, NULL)
                        ON DUPLICATE KEY UPDATE cart_json=VALUES(cart_json), notified_at=NULL
                        """,
                        (user_id, json.dumps(cart)),
                    )
                    conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    return redirect(url_for("cart"))



@app.route("/remove_from_cart/<int:product_id>")
def remove_from_cart(product_id):
    cart = session.get("cart", {})
    cart.pop(str(product_id), None)
    session["cart"] = cart
    user_id = session.get("username")
    if user_id:
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                if ensure_abandoned_carts_table(cur):
                    cur.execute(
                        """
                        INSERT INTO abandoned_carts (user_id, cart_json, notified_at)
                        VALUES (%s, %s, NULL)
                        ON DUPLICATE KEY UPDATE cart_json=VALUES(cart_json), notified_at=NULL
                        """,
                        (user_id, json.dumps(cart)),
                    )
                    conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    return redirect(url_for("cart"))


@app.route("/clear_cart")
def clear_cart():
    session.pop("cart", None)
    user_id = session.get("username")
    if user_id:
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                if ensure_abandoned_carts_table(cur):
                    cur.execute(
                        "UPDATE abandoned_carts SET cart_json=%s, notified_at=NULL WHERE user_id=%s",
                        (json.dumps({}), user_id),
                    )
                    conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    return redirect(url_for("cart"))

@app.route("/checkout")
def checkout():
    return "Checkout coming soon"


def get_db():
    return get_db_connection()



#Whatsapp route
@app.route("/pay_on_delivery", methods=["POST"])
@rate_limit("pay_on_delivery")
def pay_on_delivery():
    # 1) Require login
    if not session.get("username"):
        session["next_url"] = request.referrer or url_for("home")
        return redirect(url_for("signin"))

    user_id = session["username"]

    # 2) Location required
    location = request.form.get("location", "").strip()
    if not location:
        # return back with a query flag (simple)
        return redirect((request.referrer or url_for("cart")) + "?err=location")

    # 3) Start from cart in session
    cart = session.get("cart", {}).copy()

    # If coming from single.html, include that product too
    product_id = request.form.get("product_id")
    quantity = int(request.form.get("quantity", 1))
    if product_id:
        pid = str(product_id)
        cart[pid] = cart.get(pid, 0) + quantity

    if not cart:
        return redirect(request.referrer or url_for("home"))

    # 4) Fetch user info + products from DB
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT username, email, phone FROM users WHERE id=%s", (user_id,))
    u = cur.fetchone()
    username, email, phone = u if u else ("Customer", "", "")
    email = email or ""
    phone = phone or ""

    ids = list(cart.keys())
    placeholders = ",".join(["%s"] * len(ids))

    cur.execute(f"SELECT * FROM products WHERE product_id IN ({placeholders})", ids)
    products = cur.fetchall()

    product_map = {str(p[0]): p for p in products}  # p[0] = product_id

    # Validate stock before creating order
    updated_cart = cart.copy()
    stock_issue = False
    for pid, qty in cart.items():
        p = product_map.get(pid)
        if not p:
            updated_cart.pop(pid, None)
            stock_issue = True
            continue
        try:
            stock = int(p[5]) if p[5] is not None else 0
        except (ValueError, TypeError):
            stock = 0
        if stock <= 0:
            updated_cart.pop(pid, None)
            stock_issue = True
        elif int(qty) > stock:
            updated_cart[pid] = stock
            stock_issue = True

    if stock_issue:
        session["cart"] = updated_cart
        set_site_message("Some items are out of stock or limited. Please review your cart.", "warning")
        cur.close()
        conn.close()
        return redirect(url_for("cart"))

    # 5) Compute totals and prepare order items
    order_items = []
    subtotal = 0.0

    for pid, qty in cart.items():
        p = product_map.get(pid)
        if not p:
            continue

        name = p[1]                 # product name index
        unit_price = float(p[4])    # price index
        qty = int(qty)
        line_total = unit_price * qty
        subtotal += line_total

        order_items.append({
            "product_id": int(pid),
            "product_name": name,
            "unit_price": unit_price,
            "quantity": qty,
            "line_total": line_total
        })

    if not order_items:
        cur.close()
        conn.close()
        return redirect(request.referrer or url_for("home"))

    ensure_orders_schema(conn)
    discount = 0.0
    discount_reason = ""
    applied_coupon = None
    discount, discount_reason, _, applied_coupon = calculate_checkout_discount(conn, user_id, subtotal)
    total_after_discount = max(0.0, subtotal - discount)

    # 6) Store order in DB
    if orders_has_discount():
        cur.execute(
            """
            INSERT INTO orders (user_id, location, payment_method, status, subtotal, discount, discount_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id, location, "PAY_ON_DELIVERY", "PENDING", total_after_discount, discount, discount_reason),
        )
    else:
        cur.execute(
            "INSERT INTO orders (user_id, location, payment_method, status, subtotal) VALUES (%s, %s, %s, %s, %s)",
            (user_id, location, "PAY_ON_DELIVERY", "PENDING", total_after_discount),
        )
    order_id = cur.lastrowid
    order_reference = None
    if orders_has_reference():
        order_reference = f"ZC-{order_id:06d}"
        cur.execute(
            "UPDATE orders SET order_reference=%s WHERE order_id=%s",
            (order_reference, order_id),
        )

    for it in order_items:
        cur.execute(
            """INSERT INTO order_items
               (order_id, product_id, product_name, unit_price, quantity, line_total)
               VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (order_id, it["product_id"], it["product_name"], it["unit_price"], it["quantity"], it["line_total"])
        )
    if applied_coupon and discount > 0:
        coupon_consumed = consume_user_coupon(cur, applied_coupon.get("id"), order_id)
        if not coupon_consumed:
            # Fallback if coupon was concurrently used elsewhere.
            discount = 0.0
            discount_reason = ""
            total_after_discount = subtotal
            if orders_has_discount():
                cur.execute(
                    """
                    UPDATE orders
                    SET subtotal=%s, discount=%s, discount_reason=%s
                    WHERE order_id=%s
                    """,
                    (total_after_discount, discount, discount_reason, order_id),
                )
            else:
                cur.execute(
                    "UPDATE orders SET subtotal=%s WHERE order_id=%s",
                    (total_after_discount, order_id),
                )
    conn.commit()
    cur.close()
    conn.close()

    # 7) Build best WhatsApp message
    lines = []
    lines.append("Bigoh ORDER (PAY ON DELIVERY)")
    lines.append("--------------------------------")
    if order_reference:
        lines.append(f"Order Ref: {order_reference}")
    lines.append(f"Order ID: #{order_id}")
    lines.append(f"Customer: {username}")
    if phone: lines.append(f"Phone: {phone}")
    if email: lines.append(f"Email: {email}")
    lines.append("")
    lines.append(f"Delivery Location: {location}")
    lines.append("")
    lines.append("ITEMS:")

    n = 1
    for it in order_items:
        link = url_for("single", product_id=it["product_id"], _external=True)
        lines.append(f"{n}. {it['product_name']}")
        lines.append(f"   Qty: {it['quantity']} | Unit: KES {it['unit_price']:,.2f} | Line: KES {it['line_total']:,.2f}")
        lines.append(f"   Link: {link}")
        n += 1

    lines.append("")
    lines.append(f"SUBTOTAL: KES {subtotal:,.2f}")
    if discount > 0:
        lines.append(f"COUPON DISCOUNT: -KES {discount:,.2f}")
        if discount_reason:
            lines.append(f"COUPON: {discount_reason}")
    lines.append(f"TOTAL: KES {total_after_discount:,.2f}")
    lines.append("")
    lines.append("Kindly send me payment details for my orders.")
    lines.append("Thank you.")

    text = "\n".join(lines)

    # 8) Clear cart (cannot detect WhatsApp send success; clearing happens at redirect time)
    session.pop("cart", None)

    wa_url = f"https://wa.me/{WHATSAPP_NUMBER}?text={quote(text)}"
    session["last_order_id"] = order_id
    session["last_wa_url"] = wa_url
    return redirect(url_for("order_confirmation", order_id=order_id))




#Upload route
@app.route("/upload", methods=["GET", "POST"])
@admin_required
def upload():
    categories_list = get_managed_categories()
    if request.method == "POST":
        product_name = request.form.get("product_name", "").strip()
        selected_category = request.form.get("category", "").strip()
        category = coerce_allowed_category(selected_category, categories_list)
        brand = request.form.get("brand", "").strip()
        seller = request.form.get("seller", "").strip()
        color = request.form.get("color", "").strip()
        price = request.form.get("price", "").strip()
        stock = request.form.get("stock", "0").strip()
        description = request.form.get("description", "").strip()

        file = request.files.get("image")

        if not categories_list:
            return render_template(
                "upload.html",
                error="No categories are available. Category is locked to existing categories only.",
                categories=categories_list,
                submitted_category=selected_category,
            )

        # Basic validation
        if not product_name or not selected_category or not price:
            return render_template(
                "upload.html",
                error="Product name, category, and price are required.",
                categories=categories_list,
                submitted_category=selected_category,
            )

        if not category:
            return render_template(
                "upload.html",
                error="Invalid category. Choose one of your existing categories.",
                categories=categories_list,
                submitted_category=selected_category,
            )

        if not file or file.filename == "":
            return render_template(
                "upload.html",
                error="Please choose an image.",
                categories=categories_list,
                submitted_category=selected_category,
            )

        if not allowed_file(file.filename):
            return render_template(
                "upload.html",
                error="Invalid image type. Use jpg, jpeg, png, or webp.",
                categories=categories_list,
                submitted_category=selected_category,
            )

        image_url = None
        if USE_CLOUDINARY:
            image_url = _cloudinary_upload(file, "bigoh/products")

        if not image_url:
            try:
                file.stream.seek(0)
            except Exception:
                pass
            # Ensure upload folder exists
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

            # Save image with safe filename
            filename = secure_filename(file.filename)
            # avoid overwriting
            base, ext = os.path.splitext(filename)
            i = 1
            final_name = filename
            while os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], final_name)):
                final_name = f"{base}_{i}{ext}"
                i += 1

            save_path = os.path.join(app.config["UPLOAD_FOLDER"], final_name)
            file.save(save_path)
            compress_product_image(save_path)

            # Store relative path in DB (matches how you render: /static/....)
            image_url = f"images/{final_name}"  # because it's inside static/images/

        # Insert into DB
        connection = get_db_connection()
        try:
            ensure_products_schema(connection)
            with connection.cursor() as cursor:
                if products_has_seller(connection):
                    sql = """
                        INSERT INTO products
                        (product_name, category, brand, seller, color, price, stock, description, image_url)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(
                        sql,
                        (product_name, category, brand, seller, color, price, stock, description, image_url),
                    )
                else:
                    sql = """
                        INSERT INTO products
                        (product_name, category, brand, color, price, stock, description, image_url)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(
                        sql,
                        (product_name, category, brand, color, price, stock, description, image_url),
                    )
                connection.commit()
        finally:
            connection.close()

        return render_template(
            "upload.html",
            success="Product uploaded successfully.",
            categories=categories_list,
            submitted_category="",
        )

    return render_template("upload.html", categories=categories_list, submitted_category="")


@app.route("/order/<int:order_id>/tracking")
def order_tracking(order_id):
    wants_json = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or "application/json" in request.headers.get("Accept", "")
    )
    if not session.get("username") and not session.get("is_admin"):
        if wants_json:
            return jsonify(ok=False, message="Please sign in first."), 401
        return redirect(url_for("signin"))

    conn = get_db_connection()
    try:
        ensure_orders_delivery_columns(conn)
        has_ref = orders_has_reference()
        has_status_updated = table_has_column(conn, "orders", "status_updated_at")
        has_delivered = table_has_column(conn, "orders", "delivered_at")
        with conn.cursor() as cur:
            cols = ["order_id", "user_id", "status", "created_at", "location"]
            if has_ref:
                cols.append("order_reference")
            if has_status_updated:
                cols.append("status_updated_at")
            if has_delivered:
                cols.append("delivered_at")
            cur.execute(
                f"""
                SELECT {", ".join(cols)}
                FROM orders
                WHERE order_id = %s
                """,
                (order_id,),
            )
            row = cur.fetchone()
            if not row:
                if wants_json:
                    return jsonify(ok=False, message="Order not found."), 404
                return redirect(url_for("home"))

            user_id = int(_row_at(row, 1, 0) or 0)
            if not session.get("is_admin") and user_id != session.get("username"):
                if wants_json:
                    return jsonify(ok=False, message="Not allowed."), 403
                return redirect(url_for("home"))

            status = str(_row_at(row, 2, "PENDING") or "PENDING").upper()
            created_at = _row_at(row, 3)
            location = str(_row_at(row, 4, "") or "")

            offset = 5
            reference = f"ZC-{order_id:06d}"
            if has_ref:
                custom_ref = _row_at(row, offset, "")
                if custom_ref:
                    reference = custom_ref
                offset += 1

            status_updated_at = _row_at(row, offset) if has_status_updated else None
            if has_status_updated:
                offset += 1
            delivered_at = _row_at(row, offset) if has_delivered else None
    finally:
        conn.close()

    def _fmt(dt):
        if not dt:
            return ""
        try:
            return dt.strftime("%b %d, %Y at %H:%M")
        except Exception:
            return str(dt)

    status_to_step = {
        "PENDING": 1,
        "PROCESSING": 2,
        "DELIVERED": 3,
        "COMPLETED": 4,
        "CANCELLED": 1,
    }
    current_step = status_to_step.get(status, 1)
    step_defs = [
        ("ORDERED", "Ordered", "fa-regular fa-clipboard"),
        ("TRANSIT", "In Transit", "fa-solid fa-truck"),
        ("DELIVERED", "Delivered", "fa-solid fa-box-open"),
        ("PAID", "Payment Confirmed", "fa-solid fa-money-bill-wave"),
    ]
    steps = []
    for idx, (code, label, icon) in enumerate(step_defs, start=1):
        if status == "CANCELLED":
            done = idx == 1
            current = False
        else:
            done = idx <= current_step
            current = idx == current_step
        steps.append(
            {
                "code": code,
                "label": label,
                "icon": icon,
                "done": done,
                "current": current,
            }
        )

    events = []
    events.append(
        {
            "title": "Order Placed",
            "description": f"Your order {reference} was received and queued for processing.",
            "time": _fmt(created_at),
            "location": location,
            "highlight": True,
        }
    )

    if status in {"PROCESSING", "DELIVERED", "COMPLETED"}:
        t = status_updated_at or created_at
        events.append(
            {
                "title": "In Transit",
                "description": "Your package has left our fulfillment point and is moving through delivery routes.",
                "time": _fmt(t),
                "location": location,
                "highlight": status in {"PROCESSING"},
            }
        )
    if status in {"DELIVERED", "COMPLETED"}:
        t = delivered_at or status_updated_at or created_at
        events.append(
            {
                "title": "Delivered",
                "description": "Order delivered successfully. Payment is collected after delivery.",
                "time": _fmt(t),
                "location": location,
                "highlight": status == "DELIVERED",
            }
        )
    if status == "COMPLETED":
        t = status_updated_at or delivered_at or created_at
        events.append(
            {
                "title": "Payment Confirmed",
                "description": "Cash-on-delivery payment was received and the order is now complete.",
                "time": _fmt(t),
                "location": location,
                "highlight": True,
            }
        )
    if status == "CANCELLED":
        t = status_updated_at or created_at
        events.append(
            {
                "title": "Order Cancelled",
                "description": "This order was cancelled. Contact support if you need help.",
                "time": _fmt(t),
                "location": location,
                "highlight": True,
            }
        )

    tracking_number = f"KKE{order_id:010d}"
    status_labels = {
        "PENDING": "Pending",
        "PROCESSING": "In Transit",
        "DELIVERED": "Delivered - Payment Pending",
        "COMPLETED": "Completed - Paid",
        "CANCELLED": "Cancelled",
    }
    payload = {
        "ok": True,
        "order_id": order_id,
        "order_reference": reference,
        "tracking_number": tracking_number,
        "status": status,
        "status_label": status_labels.get(status, status.title()),
        "steps": steps,
        "events": events,
    }
    return jsonify(payload)


@app.route("/order/confirmation/<int:order_id>")
def order_confirmation(order_id):
    if not session.get("username"):
        return redirect(url_for("signin"))

    conn = get_db_connection()
    has_ref = orders_has_reference()
    has_discount = orders_has_discount()
    try:
        with conn.cursor() as cur:
            cols = ["order_id", "user_id", "location", "payment_method", "status", "subtotal"]
            if has_discount:
                cols.extend(["discount", "discount_reason"])
            if has_ref:
                cols.append("order_reference")

            cur.execute(
                f"""
                SELECT {", ".join(cols)}
                FROM orders
                WHERE order_id = %s
                """,
                (order_id,),
            )
            order = cur.fetchone()

            if not order or order[1] != session.get("username"):
                return redirect(url_for("home"))

            cur.execute(
                """
                SELECT product_id, product_name, unit_price, quantity, line_total
                FROM order_items
                WHERE order_id = %s
                """,
                (order_id,),
            )
            items = cur.fetchall() or []
    finally:
        conn.close()

    reference = f"ZC-{order_id:06d}"
    if order and has_ref:
        ref_idx = len(order) - 1
        if order[ref_idx]:
            reference = order[ref_idx]

    discount = 0.0
    discount_reason = ""
    if order and has_discount:
        discount_idx = 6
        discount = float(_row_at(order, discount_idx, 0) or 0)
        discount_reason = _row_at(order, discount_idx + 1, "")
    wa_url = session.get("last_wa_url")
    customer_name = session.get("username") or (order[1] if order else "Customer")
    issued_at = datetime.now()
    qr_payload = f"INVOICE|{reference}|ORDER:{order_id}|TOTAL:{float(order[5] or 0):.2f}|DATE:{issued_at.strftime('%Y-%m-%d')}"
    qr_data_uri = _make_qr_data_uri(qr_payload)
    scu_info = _build_scu_info(order_id, reference, issued_at, items, float(order[5] or 0))
    return render_template(
        "order_confirmation.html",
        order=order,
        items=items,
        reference=reference,
        wa_url=wa_url,
        discount=discount,
        discount_reason=discount_reason,
        issued_at=issued_at,
        qr_data_uri=qr_data_uri,
        scu_info=scu_info,
        customer_name=customer_name,
    )


@app.route("/order/receipt/<int:order_id>")
def order_receipt(order_id):
    if not session.get("username") and not session.get("is_admin"):
        return redirect(url_for("signin"))

    conn = get_db_connection()
    has_ref = orders_has_reference()
    has_discount = orders_has_discount()
    try:
        with conn.cursor() as cur:
            cols = ["order_id", "user_id", "location", "payment_method", "status", "subtotal"]
            if has_ref:
                cols.append("order_reference")
            if has_discount:
                cols.extend(["discount", "discount_reason"])

            cur.execute(
                f"""
                SELECT {", ".join(cols)}
                FROM orders
                WHERE order_id = %s
                """,
                (order_id,),
            )
            order = cur.fetchone()
            if not order:
                return redirect(url_for("home"))

            user_id = _row_at(order, 1, None)
            if not session.get("is_admin") and user_id != session.get("username"):
                return redirect(url_for("home"))

            status = str(_row_at(order, 4, "") or "").upper()
            if status not in ("DELIVERED", "COMPLETED"):
                set_site_message(
                    "Receipt is available after delivery. Final paid receipt is issued after payment confirmation.",
                    "warning",
                )
                return redirect(url_for("order_confirmation", order_id=order_id))

            cur.execute(
                """
                SELECT product_id, product_name, unit_price, quantity, line_total
                FROM order_items
                WHERE order_id = %s
                """,
                (order_id,),
            )
            items = cur.fetchall() or []

            cur.execute(
                "SELECT username, email, phone FROM users WHERE id=%s",
                (user_id,),
            )
            user = cur.fetchone() or ("Customer", "", "")
    finally:
        conn.close()

    reference = f"ZC-{order_id:06d}"
    if order and has_ref:
        ref_idx = 6
        if _row_at(order, ref_idx, ""):
            reference = _row_at(order, ref_idx, reference)

    discount = 0.0
    discount_reason = ""
    if order and has_discount:
        discount_idx = 6 + (1 if has_ref else 0)
        discount = float(_row_at(order, discount_idx, 0) or 0)
        discount_reason = _row_at(order, discount_idx + 1, "")

    items_total = sum(float(_row_at(it, 4, 0) or 0) for it in items)
    order_total = float(_row_at(order, 5, 0) or 0)
    receipt_date = datetime.now()

    is_paid = str(_row_at(order, 4, "") or "").upper() == "COMPLETED"
    payment_method = str(_row_at(order, 3, "") or "")
    if is_paid:
        payment_details_lines = [line.strip() for line in PAYMENT_DETAILS_LINES.splitlines() if line.strip()]
    else:
        payment_details_lines = [
            f"Payment method: {payment_method or 'Cash on Delivery'}",
            "Status: Payment pending",
            "Payment is collected after delivery.",
        ]
    payment_details_title = PAYMENT_DETAILS_TITLE if is_paid else "Payment Status"

    transactions = []
    if is_paid:
        transactions = [
            {
                "date": receipt_date.strftime("%A, %B %d, %Y"),
                "gateway": payment_method,
                "transaction_id": reference,
                "amount": order_total,
            }
        ]

    receipt_state_label = "PAID" if is_paid else "PAYMENT PENDING"
    document_title = "Receipt" if is_paid else "Invoice"
    due_date_label = receipt_date.strftime("%A, %B %d, %Y") if is_paid else "Pay on delivery"
    return render_template(
        "receipt.html",
        order=order,
        items=items,
        reference=reference,
        discount=discount,
        discount_reason=discount_reason,
        items_total=items_total,
        order_total=order_total,
        customer=user,
        business_logo=image_url(BUSINESS_LOGO),
        business_name=BUSINESS_NAME,
        business_address=BUSINESS_ADDRESS,
        business_reg_no=BUSINESS_REG_NO,
        business_reg_body=BUSINESS_REG_BODY,
        support_email=SUPPORT_EMAIL,
        support_phone=SUPPORT_PHONE,
        support_whatsapp=SUPPORT_WHATSAPP,
        receipt_date=receipt_date,
        is_paid=is_paid,
        receipt_state_label=receipt_state_label,
        document_title=document_title,
        due_date_label=due_date_label,
        payment_details_title=payment_details_title,
        payment_details_lines=payment_details_lines,
        transactions=transactions,
    )



# Admin dashboard route
@app.route("/admin")
@admin_required
def admin_dashboard():
    conn = None
    error = None
    search = request.args.get("q", "").strip()
    metrics = {
        "users": 0,
        "products": 0,
        "orders": 0,
        "revenue": 0.0,
        "quantity": 0,
        "pending": 0,
        "processing": 0,
        "delivered": 0,
        "completed": 0,
        "cancelled": 0,
    }
    orders = []
    top_products = []
    status_rows = []
    category_rows = []
    recent_products = []
    low_stock = []
    flash_sale_active = False
    flash_sale_duration_seconds = 0
    flash_sale_time_label = format_duration(0)
    flash_selected_ids = []
    flash_products = []
    flash_duration_hours = 0
    flash_duration_minutes = 0
    repeat_customers = 0
    conversion_rate = 0.0
    avg_order_value = 0.0
    active_count = 0
    active_sessions = []

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        metrics["users"] = _scalar(cur, "SELECT COUNT(*) FROM users", default=0)
        metrics["products"] = _scalar(cur, "SELECT COUNT(*) FROM products", default=0)
        metrics["orders"] = _scalar(cur, "SELECT COUNT(*) FROM orders", default=0)
        metrics["revenue"] = _scalar(cur, "SELECT COALESCE(SUM(subtotal), 0) FROM orders", default=0.0)
        metrics["quantity"] = _scalar(cur, "SELECT COALESCE(SUM(quantity), 0) FROM order_items", default=0)
        metrics["pending"] = _scalar(cur, "SELECT COUNT(*) FROM orders WHERE status = %s", ("PENDING",), default=0)
        metrics["processing"] = _scalar(cur, "SELECT COUNT(*) FROM orders WHERE status = %s", ("PROCESSING",), default=0)
        metrics["delivered"] = _scalar(cur, "SELECT COUNT(*) FROM orders WHERE status = %s", ("DELIVERED",), default=0)
        metrics["completed"] = _scalar(cur, "SELECT COUNT(*) FROM orders WHERE status = %s", ("COMPLETED",), default=0)
        metrics["cancelled"] = _scalar(cur, "SELECT COUNT(*) FROM orders WHERE status = %s", ("CANCELLED",), default=0)
        repeat_customers = _scalar(
            cur,
            """
            SELECT COUNT(*)
            FROM (
                SELECT user_id
                FROM orders
                WHERE user_id IS NOT NULL
                GROUP BY user_id
                HAVING COUNT(*) >= 2
            ) t
            """,
            default=0,
        )

        try:
            has_ref = orders_has_reference()
            if has_ref:
                base_query = """
                    SELECT o.order_id, o.order_reference, u.username, u.phone, o.location, o.payment_method, o.status, o.subtotal
                    FROM orders o
                    LEFT JOIN users u ON o.user_id = u.id
                """
            else:
                base_query = """
                    SELECT o.order_id, u.username, u.phone, o.location, o.payment_method, o.status, o.subtotal
                    FROM orders o
                    LEFT JOIN users u ON o.user_id = u.id
                """
            params = []
            if search:
                like = f"%{search}%"
                try:
                    order_id = int(search)
                except ValueError:
                    order_id = -1
                base_query += " WHERE o.order_id = %s OR u.username LIKE %s OR u.phone LIKE %s OR o.location LIKE %s "
                params = [order_id, like, like, like]
            base_query += " ORDER BY o.order_id DESC LIMIT 20 "
            cur.execute(base_query, params)
            orders = cur.fetchall() or []
        except Exception:
            orders = []

        try:
            cur.execute(
                """
                SELECT oi.product_name, SUM(oi.quantity) AS qty, SUM(oi.line_total) AS revenue
                FROM order_items oi
                GROUP BY oi.product_name
                ORDER BY qty DESC
                LIMIT 6
                """
            )
            top_products = cur.fetchall() or []
        except Exception:
            top_products = []

        try:
            cur.execute(
                """
                SELECT status, COUNT(*) AS total
                FROM orders
                GROUP BY status
                """
            )
            status_rows = cur.fetchall() or []
        except Exception:
            status_rows = []

        try:
            cur.execute(
                """
                SELECT p.category, SUM(oi.quantity) AS total_qty
                FROM order_items oi
                JOIN products p ON oi.product_id = p.product_id
                GROUP BY p.category
                ORDER BY total_qty DESC
                """
            )
            category_rows = cur.fetchall() or []
        except Exception:
            category_rows = []

        try:
            cur.execute(
                """
                SELECT product_id, product_name, category, price, stock, image_url
                FROM products
                ORDER BY product_id DESC
                LIMIT 8
                """
            )
            recent_products = cur.fetchall() or []
        except Exception:
            recent_products = []

        try:
            cur.execute(
                """
                SELECT product_id, product_name, category, price, stock
                FROM products
                WHERE stock <= %s
                ORDER BY stock ASC, product_id DESC
                LIMIT 6
                """
                ,
                (LOW_STOCK_THRESHOLD,),
            )
            low_stock = cur.fetchall() or []
        except Exception:
            low_stock = []

        try:
            send_low_stock_alerts(conn, low_stock)
        except Exception:
            pass

        try:
            if ensure_active_sessions_table(cur):
                conn.commit()
                cutoff = datetime.now() - timedelta(minutes=ACTIVE_SESSION_WINDOW_MINUTES)
                active_count = _scalar(
                    cur,
                    "SELECT COUNT(*) FROM active_sessions WHERE last_seen >= %s",
                    (cutoff,),
                    default=0,
                )
                cur.execute(
                    """
                    SELECT username, user_id, ip_address, last_seen, current_path
                    FROM active_sessions
                    WHERE last_seen >= %s
                    ORDER BY last_seen DESC
                    LIMIT 12
                    """,
                    (cutoff,),
                )
                active_sessions = []
                for row in cur.fetchall() or []:
                    last_seen = _row_at(row, 3, None)
                    active_sessions.append(
                        {
                            "username": _row_at(row, 0, "") or "Guest",
                            "user_id": _row_at(row, 1, None),
                            "ip": _row_at(row, 2, "-"),
                            "last_seen": last_seen.strftime("%H:%M") if last_seen else "-",
                            "path": _row_at(row, 4, "-"),
                        }
                    )
        except Exception:
            active_count = 0
            active_sessions = []

        try:
            if ensure_flash_sale_tables(cur):
                conn.commit()
                cur.execute(
                    "SELECT is_active, duration_seconds, ends_at FROM flash_sale_settings WHERE id = 1"
                )
                row = cur.fetchone()
                flash_sale_active = bool(_row_at(row, 0, 0)) if row else False
                flash_sale_duration_seconds = int(_row_at(row, 1, 0) or 0) if row else 0
                ends_at = _row_at(row, 2, None) if row else None

                now = datetime.now()
                if flash_sale_active and flash_sale_duration_seconds <= 0:
                    flash_sale_active = False
                if flash_sale_active and flash_sale_duration_seconds > 0 and ends_at is None:
                    ends_at = now + timedelta(seconds=flash_sale_duration_seconds)
                    cur.execute(
                        "UPDATE flash_sale_settings SET ends_at=%s WHERE id=1",
                        (ends_at,),
                    )
                    conn.commit()

                seconds_left = 0
                if flash_sale_active and ends_at:
                    seconds_left = int((ends_at - now).total_seconds())
                    if seconds_left <= 0:
                        flash_sale_active = False
                        seconds_left = 0
                        cur.execute(
                            "UPDATE flash_sale_settings SET is_active=0, ends_at=NULL WHERE id=1"
                        )
                        conn.commit()

                flash_sale_time_label = format_duration(
                    seconds_left if flash_sale_active else 0
                )
                flash_duration_hours = flash_sale_duration_seconds // 3600
                flash_duration_minutes = (flash_sale_duration_seconds % 3600) // 60

                cur.execute(
                    "SELECT product_id FROM flash_sale_items WHERE is_active = 1"
                )
                flash_selected_ids = [int(_row_at(row, 0, 0)) for row in cur.fetchall() or []]

                cur.execute(
                    """
                    SELECT product_id, product_name, category, price, stock, image_url
                    FROM products
                    ORDER BY product_id DESC
                    LIMIT 60
                    """
                )
                flash_products = cur.fetchall() or []
        except Exception:
            flash_sale_active = False
            flash_selected_ids = []
            flash_products = []

    except Exception as exc:
        error = str(exc)
    finally:
        if conn:
            conn.close()

    status_totals = {
        str(_row_at(row, 0, "") or "").upper(): int(_row_at(row, 1, 0) or 0)
        for row in status_rows
    }
    status_labels = [ORDER_STATUS_LABELS[code] for code in ORDER_STATUS_SEQUENCE]
    status_values = [int(status_totals.get(code, 0)) for code in ORDER_STATUS_SEQUENCE]

    category_labels = [_row_at(row, 0) for row in category_rows]
    category_values = [int(_row_at(row, 1, 0)) for row in category_rows]

    completion_rate = 0.0
    if metrics["orders"]:
        completion_rate = round((metrics["completed"] / metrics["orders"]) * 100, 2)
    if metrics["users"]:
        conversion_rate = round((metrics["orders"] / metrics["users"]) * 100, 2)
    if metrics["orders"]:
        avg_order_value = round(metrics["revenue"] / metrics["orders"], 2)

    return render_template(
        "admin_dashboard.html",
        metrics=metrics,
        completion_rate=completion_rate,
        conversion_rate=conversion_rate,
        repeat_customers=repeat_customers,
        avg_order_value=avg_order_value,
        orders=orders,
        recent_products=recent_products,
        low_stock=low_stock,
        top_products=top_products,
        active_count=active_count,
        active_sessions=active_sessions,
        active_window=ACTIVE_SESSION_WINDOW_MINUTES,
        status_labels=json.dumps(status_labels),
        status_values=json.dumps(status_values),
        category_labels=json.dumps(category_labels),
        category_values=json.dumps(category_values),
        category_rows=category_rows,
        has_reference=orders_has_reference(),
        search=search,
        error=error,
        flash_sale_active=flash_sale_active,
        flash_sale_duration_seconds=flash_sale_duration_seconds,
        flash_sale_time_label=flash_sale_time_label,
        flash_selected_ids=flash_selected_ids,
        flash_products=flash_products,
        flash_duration_hours=flash_duration_hours,
        flash_duration_minutes=flash_duration_minutes,
    )


@app.route("/admin/migrate-uploads", methods=["POST"])
@admin_required
def admin_migrate_uploads():
    result = _migrate_static_uploads()
    if "error" in result:
        set_site_message(result["error"], "warning")
    else:
        copied = result.get("copied", 0)
        skipped = result.get("skipped", 0)
        errors = result.get("errors", 0)
        level = "success" if errors == 0 else "warning"
        set_site_message(
            f"Upload migration complete: copied {copied}, skipped {skipped}, errors {errors}.",
            level,
        )
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/migrate-cloudinary", methods=["POST"])
@admin_required
def admin_migrate_cloudinary():
    try:
        limit = int(request.form.get("limit", "25") or 25)
    except ValueError:
        limit = 25
    limit = max(1, min(200, limit))
    result = _migrate_cloudinary_assets(limit=limit)
    if "error" in result:
        set_site_message(result["error"], "warning")
    else:
        copied = result.get("copied", 0)
        skipped = result.get("skipped", 0)
        errors = result.get("errors", 0)
        processed = result.get("processed", 0)
        limit_used = result.get("limit", limit)
        level = "success" if errors == 0 else "warning"
        set_site_message(
            f"Cloudinary migration batch complete: processed {processed}/{limit_used}, copied {copied}, skipped {skipped}, errors {errors}. Run again to continue.",
            level,
        )
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/cloudinary-duplicates", methods=["POST"])
@admin_required
def admin_cloudinary_duplicates():
    try:
        limit = int(request.form.get("limit", "200") or 200)
    except ValueError:
        limit = 200
    limit = max(1, min(500, limit))
    result = _cloudinary_find_duplicates(max_results=limit)
    if "error" in result:
        set_site_message(result["error"], "warning")
    else:
        dupes = result.get("duplicates", [])
        scanned = result.get("scanned", 0)
        set_site_message(
            f"Cloudinary scan complete: scanned {scanned}, duplicates found {len(dupes)} (limit {limit}).",
            "info",
        )
        session["cloudinary_duplicates"] = dupes
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/cloudinary-cleanup", methods=["POST"])
@admin_required
def admin_cloudinary_cleanup():
    try:
        limit = int(request.form.get("limit", "500") or 500)
    except ValueError:
        limit = 500
    limit = max(10, min(1000, limit))
    result = _cloudinary_delete_duplicate_content(max_results=limit)
    if "error" in result:
        set_site_message(result["error"], "warning")
    else:
        deleted = result.get("deleted", 0)
        errors = result.get("errors", 0)
        kept = result.get("kept", 0)
        scanned = result.get("scanned", 0)
        level = "success" if errors == 0 else "warning"
        set_site_message(
            f"Cloudinary cleanup complete: scanned {scanned}, kept {kept}, deleted {deleted}, errors {errors}.",
            level,
        )
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/flash-sale", methods=["GET", "POST"])
@admin_required
def admin_flash_sale():
    if request.method == "POST":
        is_active = request.form.get("is_active") == "on"

        try:
            duration_hours = int(request.form.get("duration_hours", "0"))
        except ValueError:
            duration_hours = 0
        try:
            duration_minutes = int(request.form.get("duration_minutes", "0"))
        except ValueError:
            duration_minutes = 0

        duration_hours = max(duration_hours, 0)
        duration_minutes = max(duration_minutes, 0)
        duration_seconds = (duration_hours * 3600) + (duration_minutes * 60)
        if is_active and duration_seconds <= 0:
            is_active = False

        selected_ids = []
        for raw in request.form.getlist("flash_products"):
            try:
                selected_ids.append(int(raw))
            except (TypeError, ValueError):
                continue
        if selected_ids:
            selected_ids = sorted(set(selected_ids))

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                if not ensure_flash_sale_tables(cur):
                    return redirect(url_for("admin_dashboard"))

                cur.execute("DELETE FROM flash_sale_items")
                if selected_ids:
                    cur.executemany(
                        "INSERT INTO flash_sale_items (product_id, is_active) VALUES (%s, 1)",
                        [(pid,) for pid in selected_ids],
                    )

                ends_at = None
                if is_active and duration_seconds > 0:
                    ends_at = datetime.now() + timedelta(seconds=duration_seconds)

                cur.execute(
                    """
                    UPDATE flash_sale_settings
                    SET is_active=%s, duration_seconds=%s, ends_at=%s
                    WHERE id=1
                    """,
                    (1 if is_active else 0, duration_seconds, ends_at),
                )
            conn.commit()
        finally:
            conn.close()

        return redirect(url_for("admin_flash_sale"))

    conn = get_db_connection()
    flash_sale_active = False
    flash_sale_time_label = format_duration(0)
    flash_duration_hours = 0
    flash_duration_minutes = 0
    flash_selected_ids = []
    flash_products = []

    try:
        with conn.cursor() as cur:
            if ensure_flash_sale_tables(cur):
                conn.commit()
                cur.execute(
                    "SELECT is_active, duration_seconds, ends_at FROM flash_sale_settings WHERE id = 1"
                )
                row = cur.fetchone()
                flash_sale_active = bool(_row_at(row, 0, 0)) if row else False
                flash_sale_duration_seconds = int(_row_at(row, 1, 0) or 0) if row else 0
                ends_at = _row_at(row, 2, None) if row else None

                now = datetime.now()
                if flash_sale_active and flash_sale_duration_seconds <= 0:
                    flash_sale_active = False
                if flash_sale_active and flash_sale_duration_seconds > 0 and ends_at is None:
                    ends_at = now + timedelta(seconds=flash_sale_duration_seconds)
                    cur.execute(
                        "UPDATE flash_sale_settings SET ends_at=%s WHERE id=1",
                        (ends_at,),
                    )
                    conn.commit()

                seconds_left = 0
                if flash_sale_active and ends_at:
                    seconds_left = int((ends_at - now).total_seconds())
                    if seconds_left <= 0:
                        flash_sale_active = False
                        seconds_left = 0
                        cur.execute(
                            "UPDATE flash_sale_settings SET is_active=0, ends_at=NULL WHERE id=1"
                        )
                        conn.commit()

                flash_sale_time_label = format_duration(
                    seconds_left if flash_sale_active else 0
                )
                flash_duration_hours = flash_sale_duration_seconds // 3600
                flash_duration_minutes = (flash_sale_duration_seconds % 3600) // 60

                cur.execute(
                    "SELECT product_id FROM flash_sale_items WHERE is_active = 1"
                )
                flash_selected_ids = [int(_row_at(row, 0, 0)) for row in cur.fetchall() or []]

                cur.execute(
                    """
                    SELECT product_id, product_name, category, price, stock, image_url
                    FROM products
                    ORDER BY product_id DESC
                    LIMIT 80
                    """
                )
                flash_products = cur.fetchall() or []
    finally:
        conn.close()

    return render_template(
        "admin_flash_sale.html",
        flash_sale_active=flash_sale_active,
        flash_sale_time_label=flash_sale_time_label,
        flash_duration_hours=flash_duration_hours,
        flash_duration_minutes=flash_duration_minutes,
        flash_selected_ids=flash_selected_ids,
        flash_products=flash_products,
    )


@app.route("/admin/sponsored-products", methods=["GET", "POST"])
@admin_required
def admin_sponsored_products():
    if request.method == "POST":
        selected_ids = []
        for raw in request.form.getlist("sponsored_products"):
            try:
                selected_ids.append(int(raw))
            except (TypeError, ValueError):
                continue
        if selected_ids:
            selected_ids = sorted(set(selected_ids))

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                if not ensure_sponsored_products_table(cur):
                    return redirect(url_for("admin_dashboard"))
                cur.execute("DELETE FROM sponsored_products")
                if selected_ids:
                    cur.executemany(
                        "INSERT INTO sponsored_products (product_id, is_active) VALUES (%s, 1)",
                        [(pid,) for pid in selected_ids],
                    )
            conn.commit()
        finally:
            conn.close()

        return redirect(url_for("admin_sponsored_products"))

    conn = get_db_connection()
    selected_ids = []
    products = []
    try:
        with conn.cursor() as cur:
            if ensure_sponsored_products_table(cur):
                conn.commit()
                cur.execute(
                    "SELECT product_id FROM sponsored_products WHERE is_active = 1"
                )
                selected_ids = [int(_row_at(row, 0, 0)) for row in cur.fetchall() or []]
                cur.execute(
                    """
                    SELECT product_id, product_name, category, price, stock, image_url
                    FROM products
                    ORDER BY product_id DESC
                    LIMIT 80
                    """
                )
                products = cur.fetchall() or []
    finally:
        conn.close()

    return render_template(
        "admin_sponsored_products.html",
        sponsored_products=products,
        sponsored_selected_ids=selected_ids,
    )


@app.route("/admin/order/<int:order_id>/status", methods=["POST"])
@admin_required
def admin_update_order_status(order_id):
    status = request.form.get("status", "").strip().upper()
    allowed = set(ORDER_STATUS_SEQUENCE)
    if status not in allowed:
        return redirect(request.referrer or url_for("admin_dashboard"))

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            ensure_orders_delivery_columns(conn)
            cur.execute("SELECT status, user_id FROM orders WHERE order_id=%s", (order_id,))
            row = cur.fetchone()
            prev_status = str(_row_at(row, 0, "") or "").upper()
            user_id = _row_at(row, 1, None)
            if status == "COMPLETED" and prev_status not in {"DELIVERED", "COMPLETED"}:
                set_site_message(
                    "For pay-after-delivery orders, mark this order as DELIVERED before COMPLETED.",
                    "warning",
                )
                return redirect(request.referrer or url_for("admin_dashboard"))
            cur.execute(
                "UPDATE orders SET status=%s, status_updated_at=NOW() WHERE order_id=%s",
                (status, order_id),
            )
            if user_id and prev_status != status and WHATSAPP_STATUS_UPDATES_ENABLED:
                cur.execute("SELECT phone, username FROM users WHERE id=%s", (user_id,))
                phone_row = cur.fetchone()
                user_phone = _row_at(phone_row, 0, "")
                user_name = _row_at(phone_row, 1, "") or "Customer"
                receipt_link = url_for("order_receipt", order_id=order_id, _external=True)
                message = _build_status_message(status, order_id, user_name, receipt_link)
                _send_whatsapp_message(user_phone, message)
            if status == "DELIVERED" and prev_status != "DELIVERED":
                cur.execute(
                    "UPDATE orders SET delivered_at=NOW() WHERE order_id=%s",
                    (order_id,),
                )
        conn.commit()
    finally:
        conn.close()

    return redirect(request.referrer or url_for("admin_dashboard"))


@app.route("/admin/orders/<int:order_id>/send-receipt", methods=["POST"])
@admin_required
def admin_send_receipt(order_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, user_id FROM orders WHERE order_id=%s", (order_id,))
            row = cur.fetchone()
            status = str(_row_at(row, 0, "") or "").upper()
            user_id = _row_at(row, 1, None)
            if status != "COMPLETED":
                set_site_message(
                    "Final receipt can be sent only after payment is confirmed (COMPLETED).",
                    "warning",
                )
                return redirect(request.referrer or url_for("admin_orders"))
            if not user_id:
                set_site_message("Customer not found for this order.", "warning")
                return redirect(request.referrer or url_for("admin_orders"))
            cur.execute("SELECT phone FROM users WHERE id=%s", (user_id,))
            phone_row = cur.fetchone()
            user_phone = _row_at(phone_row, 0, "")
    finally:
        conn.close()

    receipt_link = url_for("order_receipt", order_id=order_id, _external=True)
    message = (
        f"Hello,\n"
        f"Your order #{order_id} has been completed and payment has been confirmed.\n"
        f"Invoice/Receipt: {receipt_link}\n"
        f"Regards,\n"
        f"{BUSINESS_NAME}"
    )
    ok = _send_whatsapp_message(user_phone, message)
    if ok:
        set_site_message("Receipt sent to customer via WhatsApp.", "success")
    else:
        set_site_message("Receipt not sent. Check WhatsApp settings or phone number.", "warning")
    return redirect(request.referrer or url_for("admin_orders"))


@app.route("/admin/products")
@admin_required
def admin_products():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            ensure_products_schema(conn)
            cur.execute("SELECT * FROM products ORDER BY product_id DESC")
            products = cur.fetchall() or []
        has_seller = products_has_seller(conn)
    finally:
        conn.close()
    return render_template(
        "admin_products.html",
        products=products,
        products_has_seller=has_seller,
        seller_index=9,
    )


@app.route("/admin/products/<int:product_id>/edit", methods=["GET", "POST"])
@admin_required
def admin_edit_product(product_id):
    product = None
    categories_list = []
    category_warning = ""
    product_seller = ""
    product_color = ""
    conn = get_db_connection()
    try:
        has_seller = ensure_products_schema(conn)
        categories_list = get_managed_categories()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM products WHERE product_id=%s", (product_id,))
            product = cur.fetchone()
            if not product:
                return redirect(url_for("admin_products"))
            product_seller = _row_at(product, 9, "") if has_seller else ""
            product_color = _row_at(product, 10, "") if products_has_color(conn) else ""
            existing_category = str(_row_at(product, 2, "") or "").strip()
            if existing_category and not coerce_allowed_category(existing_category, categories_list):
                category_warning = (
                    f'Current category "{existing_category}" is not in the allowed list. '
                    "Select a valid category and save."
                )

            if request.method == "POST":
                product_name = request.form.get("product_name", "").strip()
                selected_category = request.form.get("category", "").strip()
                category = coerce_allowed_category(selected_category, categories_list)
                brand = request.form.get("brand", "").strip()
                seller = request.form.get("seller", "").strip()
                color = request.form.get("color", "").strip()
                price = request.form.get("price", "").strip()
                stock = request.form.get("stock", "0").strip()
                description = request.form.get("description", "").strip()

                if not categories_list:
                    return render_template(
                        "admin_product_edit.html",
                        error="No categories are available. Category is locked to existing categories only.",
                        product=product,
                        product_seller=product_seller,
                        product_color=product_color,
                        products_has_seller=has_seller,
                        category_warning=category_warning,
                        categories=categories_list,
                    )

                if not category:
                    return render_template(
                        "admin_product_edit.html",
                        error="Invalid category. Choose one of your existing categories.",
                        product=product,
                        product_seller=product_seller,
                        product_color=product_color,
                        products_has_seller=has_seller,
                        category_warning=category_warning,
                        categories=categories_list,
                    )

                image_url = product[7]
                old_stock = int(_row_at(product, 5, 0) or 0)
                file = request.files.get("image")
                if file and file.filename:
                    if allowed_file(file.filename):
                        image_url = _cloudinary_upload(file, "bigoh/products") if USE_CLOUDINARY else None
                        if not image_url:
                            try:
                                file.stream.seek(0)
                            except Exception:
                                pass
                            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
                            filename = secure_filename(file.filename)
                            base, ext = os.path.splitext(filename)
                            i = 1
                            final_name = filename
                            while os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], final_name)):
                                final_name = f"{base}_{i}{ext}"
                                i += 1
                            save_path = os.path.join(app.config["UPLOAD_FOLDER"], final_name)
                            file.save(save_path)
                            compress_product_image(save_path)
                            image_url = f"images/{final_name}"

                if has_seller and products_has_seller(conn):
                    cur.execute(
                        """
                        UPDATE products
                        SET product_name=%s, category=%s, brand=%s, seller=%s, color=%s, price=%s, stock=%s, description=%s, image_url=%s
                        WHERE product_id=%s
                        """,
                        (product_name, category, brand, seller, color, price, stock, description, image_url, product_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE products
                        SET product_name=%s, category=%s, brand=%s, color=%s, price=%s, stock=%s, description=%s, image_url=%s
                        WHERE product_id=%s
                        """,
                        (product_name, category, brand, color, price, stock, description, image_url, product_id),
                    )
                conn.commit()
                try:
                    new_stock = int(stock or 0)
                except ValueError:
                    new_stock = 0
                if old_stock > 0 and new_stock <= 0:
                    try:
                        with conn.cursor() as alert_cur:
                            alert_cur.execute(
                                "UPDATE back_in_stock_alerts SET notified_at = NULL WHERE product_id = %s",
                                (product_id,),
                            )
                        conn.commit()
                    except Exception:
                        pass
                if old_stock <= 0 and new_stock > 0:
                    send_back_in_stock_alerts(conn, product_id, product_name)
                return redirect(url_for("admin_products"))
    finally:
        conn.close()

    return render_template(
        "admin_product_edit.html",
        product=product,
        product_seller=product_seller,
        product_color=product_color,
        products_has_seller=has_seller,
        category_warning=category_warning,
        categories=categories_list,
    )


@app.route("/admin/products/<int:product_id>/delete", methods=["POST"])
@admin_required
def admin_delete_product(product_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM products WHERE product_id=%s", (product_id,))
        conn.commit()
    finally:
        conn.close()
    return redirect(url_for("admin_products"))


@app.route("/admin/orders")
@admin_required
def admin_orders():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if orders_has_reference():
                cur.execute(
                    """
                    SELECT o.order_id, o.order_reference, u.username, u.phone, o.location, o.payment_method, o.status, o.subtotal
                    FROM orders o
                    LEFT JOIN users u ON o.user_id = u.id
                    ORDER BY o.order_id DESC
                    """
                )
            else:
                cur.execute(
                    """
                    SELECT o.order_id, u.username, u.phone, o.location, o.payment_method, o.status, o.subtotal
                    FROM orders o
                    LEFT JOIN users u ON o.user_id = u.id
                    ORDER BY o.order_id DESC
                    """
                )
            orders = cur.fetchall() or []
    finally:
        conn.close()
    return render_template("admin_orders.html", orders=orders, has_reference=orders_has_reference())


@app.route("/admin/orders/<int:order_id>")
@admin_required
def admin_order_detail(order_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if orders_has_reference():
                cur.execute(
                    """
                    SELECT o.order_id, o.location, o.payment_method, o.status, o.subtotal, o.order_reference,
                           u.username, u.email, u.phone
                    FROM orders o
                    LEFT JOIN users u ON o.user_id = u.id
                    WHERE o.order_id = %s
                    """,
                    (order_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT o.order_id, o.location, o.payment_method, o.status, o.subtotal,
                           u.username, u.email, u.phone
                    FROM orders o
                    LEFT JOIN users u ON o.user_id = u.id
                    WHERE o.order_id = %s
                    """,
                    (order_id,),
                )
            order = cur.fetchone()

            cur.execute(
                """
                SELECT product_name, unit_price, quantity, line_total
                FROM order_items
                WHERE order_id = %s
                """,
                (order_id,),
            )
            items = cur.fetchall() or []
    finally:
        conn.close()

    if not order:
        return redirect(url_for("admin_orders"))
    reference = f"ZC-{order_id:06d}"
    if order and orders_has_reference() and len(order) > 5 and order[5]:
        reference = order[5]
    return render_template("admin_order_detail.html", order=order, items=items, reference=reference, has_reference=orders_has_reference())


@app.route("/admin/orders/<int:order_id>/items")
@admin_required
def admin_order_items(order_id):
    conn = get_db_connection()
    items = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT product_name, unit_price, quantity, line_total
                FROM order_items
                WHERE order_id = %s
                """,
                (order_id,),
            )
            items = cur.fetchall() or []
    except Exception:
        items = []
    finally:
        conn.close()

    payload = []
    for it in items:
        try:
            payload.append(
                {
                    "name": it[0],
                    "unit_price": float(it[1] or 0),
                    "quantity": int(it[2] or 0),
                    "line_total": float(it[3] or 0),
                }
            )
        except Exception:
            continue
    return jsonify(payload)


@app.route("/whoami")
def whoami():
    return str(dict(session))



 



if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
