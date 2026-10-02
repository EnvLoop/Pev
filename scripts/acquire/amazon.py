"""Products from Amazon Reviews 2023 (McAuley Lab raw per-category files) -> staging JSONL.

Pass 1 (DuckDB) scans the narrow review columns of every category, dedupes (user, item) to the latest review,
and picks users with >= 8 distinct reviewed items (>= 8 with text >= 20 chars), a >= 30-day span and <= 200
items. Sampling is deterministic: sha256("20260930:" + user_id) order, MIXED_TARGET users with both a <= 3 and
a >= 4 rating, then OTHER_TARGET users without. Pass 2 pulls the full reviews of the selected users; items are
their reviewed products plus the POPULAR_PER_CATEGORY most-reviewed products per category (candidate pool).
"""
import json
import math
import time

import duckdb

from common import DL, MIN_INTERACTIONS, SEED, STAGING, TEXT_MAX, anon_user, clip, iso_from_ms, write_jsonl

SRC = "amazon_reviews_2023"
CATEGORIES = ["Grocery_and_Gourmet_Food", "Pet_Supplies", "Baby_Products", "Appliances", "All_Beauty"]
MIXED_TARGET, OTHER_TARGET = 6000, 2000
MAX_ITEMS_PER_USER = 200
POPULAR_PER_CATEGORY = 5000
REVIEW_COLS = ("{rating: 'DOUBLE', title: 'VARCHAR', text: 'VARCHAR', parent_asin: 'VARCHAR', "
               "user_id: 'VARCHAR', timestamp: 'BIGINT'}")
META_COLS = ("{parent_asin: 'VARCHAR', title: 'VARCHAR', main_category: 'VARCHAR', average_rating: 'DOUBLE', "
             "rating_number: 'BIGINT', features: 'VARCHAR[]', description: 'VARCHAR[]', price: 'VARCHAR', "
             "store: 'VARCHAR', categories: 'VARCHAR[]', details: 'JSON'}")


def reviews_src(cat: str) -> str:
    path = DL / "amazon" / f"{cat}.jsonl.gz"
    return f"read_json('{path}', format='newline_delimited', compression='gzip', columns={REVIEW_COLS})"


def pass1(con) -> None:
    con.execute("CREATE OR REPLACE TABLE r (cat VARCHAR, user_id VARCHAR, item VARCHAR, rating INT, ts BIGINT, "
                "tlen INT)")
    for cat in CATEGORIES:
        con.execute(f"INSERT INTO r SELECT '{cat}', user_id, parent_asin, CAST(round(rating) AS INT), timestamp, "
                    f"length(trim(coalesce(text, ''))) FROM {reviews_src(cat)} "
                    f"WHERE user_id IS NOT NULL AND parent_asin IS NOT NULL AND rating BETWEEN 1 AND 5")
        print("pass1", cat, con.execute("SELECT count(*) FROM r").fetchone()[0], flush=True)
    con.execute("CREATE OR REPLACE TABLE item_pop AS SELECT cat, item, count(*) n FROM r GROUP BY ALL")
    con.execute("""CREATE OR REPLACE TABLE ru AS SELECT * FROM r
                   QUALIFY row_number() OVER (PARTITION BY user_id, item ORDER BY ts DESC) = 1""")
    con.execute(f"""CREATE OR REPLACE TABLE users AS
        SELECT user_id, count(*) n, sum((tlen >= 20)::INT) n_text, min(rating) rmin, max(rating) rmax,
               (max(ts) - min(ts)) / 86400000.0 span_days, sha256('{SEED}:' || user_id) k
        FROM ru GROUP BY user_id
        HAVING n BETWEEN {MIN_INTERACTIONS} AND {MAX_ITEMS_PER_USER} AND n_text >= {MIN_INTERACTIONS}
           AND span_days >= 30""")
    con.execute(f"""CREATE OR REPLACE TABLE selected AS
        (SELECT user_id, true mixed FROM users WHERE rmin <= 3 AND rmax >= 4 ORDER BY k LIMIT {MIXED_TARGET})
        UNION ALL
        (SELECT user_id, false FROM users WHERE NOT (rmin <= 3 AND rmax >= 4) ORDER BY k LIMIT {OTHER_TARGET})""")
    print("eligible users", con.execute("SELECT count(*), sum((rmin <= 3 AND rmax >= 4)::INT) FROM users").fetchone(),
          "selected", con.execute("SELECT count(*) FROM selected").fetchone()[0], flush=True)


def pass2(con) -> list[dict]:
    rows = []
    for cat in CATEGORIES:
        q = f"""SELECT s.user_id, x.parent_asin, CAST(round(x.rating) AS INT), x.timestamp, x.title, x.text
                FROM {reviews_src(cat)} x JOIN selected s USING (user_id)
                WHERE x.parent_asin IS NOT NULL AND x.rating BETWEEN 1 AND 5"""
        for uid, item, rating, ts, title, text in con.execute(q).fetchall():
            title, text = (title or "").strip(), (text or "").strip()
            body = f"{title}\n{text}" if title and title.lower() not in text.lower()[: len(title) + 5] else text
            rows.append({"user_id": anon_user(SRC, uid), "item_id": f"amz:{item}", "rating": rating,
                         "timestamp": iso_from_ms(ts), "text": clip(body, TEXT_MAX), "source": SRC, "_ts": ts})
        print("pass2", cat, len(rows), flush=True)
    latest = {}
    for r in rows:  # same (user, item) can recur within and across categories: keep the latest review
        key = (r["user_id"], r["item_id"])
        if key not in latest or r["_ts"] > latest[key]["_ts"]:
            latest[key] = r
    out = sorted(latest.values(), key=lambda r: (r["user_id"], r["_ts"], r["item_id"]))
    for r in out:
        del r["_ts"]
    return out


def parse_price(p):
    try:
        v = float(str(p).replace("$", "").replace(",", ""))
        return v if math.isfinite(v) and v > 0 else None
    except (TypeError, ValueError):
        return None


def items(con, used: set[str]) -> list[dict]:
    out = {}
    for cat in CATEGORIES:
        popular = {i for (i,) in con.execute(f"SELECT item FROM item_pop WHERE cat = '{cat}' ORDER BY n DESC, item "
                                             f"LIMIT {POPULAR_PER_CATEGORY}").fetchall()}
        path = DL / "amazon" / f"meta_{cat}.jsonl.gz"
        while f'"file": "{path.name}"' not in (DL / "MANIFEST.jsonl").read_text():
            time.sleep(10)  # the downloader appends to the manifest only after a file is complete
        q = f"SELECT * FROM read_json('{path}', format='newline_delimited', compression='gzip', columns={META_COLS})"
        for (asin, title, main_cat, avg, n, feats, desc, price, store, cats, details) in con.execute(q).fetchall():
            iid = f"amz:{asin}"
            if iid in out or (iid not in used and asin not in popular):
                continue
            det = json.loads(details) if details else {}
            det = {k: v for k, v in det.items() if isinstance(v, (str, int, float)) and len(str(v)) <= 200}
            out[iid] = {
                "item_id": iid, "domain": "product", "name": clip(title, 400), "category": cat,
                "price": parse_price(price), "price_level": None,
                "attributes": {
                    "main_category": main_cat, "subcategories": cats or [], "store": store,
                    "average_rating": avg, "rating_number": n,
                    "features": [clip(f, 200) for f in (feats or [])[:5]],
                    "description": clip(" ".join(desc or []), 500) or None, "details": det,
                    "popular_pool": asin in popular,
                },
                "city": None, "source": SRC,
            }
        print("items", cat, len(out), flush=True)
    return [out[k] for k in sorted(out)]


def main() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(STAGING / "amazon.duckdb"))
    con.execute("SET memory_limit = '14GB'; SET preserve_insertion_order = false; SET enable_progress_bar = false;")
    con.execute(f"SET temp_directory = '{STAGING / 'duckdb_tmp'}'")
    pass1(con)
    rows = pass2(con)
    used = {r["item_id"] for r in rows}
    item_rows = items(con, used)
    known = {i["item_id"] for i in item_rows}
    missing = used - known
    rows = [r for r in rows if r["item_id"] in known]  # drop reviews whose product has no metadata row
    print("interactions", write_jsonl(STAGING / "interactions_product.jsonl", rows),
          "items", write_jsonl(STAGING / "items_product.jsonl", item_rows), "items without metadata", len(missing))


if __name__ == "__main__":
    main()
