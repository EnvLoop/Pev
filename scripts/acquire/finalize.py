"""Merge staging tables into items.jsonl / interactions.jsonl, validate them, write STATS.json and LICENSES.json."""
import json
import re
import statistics
from collections import Counter, defaultdict

from common import DL, EMAIL_BODY_MAX, MIN_INTERACTIONS, RAW, SEED, STAGING, TEXT_MAX

ITEM_KEYS = ["item_id", "domain", "name", "category", "price", "price_level", "attributes", "city", "source"]
INTER_KEYS = ["user_id", "item_id", "rating", "timestamp", "text", "source"]
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
HEX16 = re.compile(r"^[0-9a-f]{16}$")

SOURCES = {
    "amazon": {
        "name": "Amazon Reviews 2023 (McAuley Lab, UCSD)", "source_tag": "amazon_reviews_2023",
        "landing": "https://amazon-reviews-2023.github.io/ ; https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023",
        "hf_revision": "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e (card revision; files fetched from the raw links it "
                       "points to at mcauleylab.ucsd.edu)",
        "license": "No explicit license published (HF card has no license field); released by McAuley Lab for research.",
        "selection": "Categories Grocery_and_Gourmet_Food, Pet_Supplies, Baby_Products, Appliances, All_Beauty. Users "
                     "with 8-200 distinct items (>=8 reviews with >=20 chars text), >=30-day span; 6000 with mixed "
                     "ratings (<=3 and >=4) + 2000 others, ordered by sha256('20260930:'+user_id). Items = reviewed "
                     "products + top-5000 most-reviewed per category (attributes.popular_pool).",
        "usage_restriction": "Academic/non-commercial research use; cite Hou et al. 2024 'Bridging Language and Items "
                             "for Retrieval and Recommendation' (arXiv:2403.03952). Do not redistribute raw data.",
    },
    "google": {
        "name": "Google Local Data 2021 (UCSD; Li, Shang, McAuley; Yan et al.)", "source_tag": "google_local_2021",
        "landing": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/",
        "license": "No explicit license published; released by UCSD for research.",
        "selection": "States District_of_Columbia, Delaware, Rhode_Island (10-core files). Businesses with a food "
                     "category (regex in scripts/google_local.py); all users with >=8 distinct food businesses, "
                     ">=30-day span and >=5 texted reviews (no sampling). Reviews dated before 2005 dropped.",
        "usage_restriction": "Academic/non-commercial research use; cite UCTopic (ACL 2022) and Personalized Showcases "
                             "(SIGIR 2023). Reviewer names, photos and owner responses were dropped; user ids hashed.",
    },
    "openflights": {
        "name": "OpenFlights airports/airlines/routes", "source_tag": "openflights",
        "landing": "https://github.com/jpatokal/openflights (data/), https://openflights.org/data.php",
        "revision": "7d1a611e070295dba776d6afb86e57d0d1aa1cef (master, committed 2026-09-21)",
        "license": "Open Database License (ODbL) 1.0; contents under Database Contents License (DbCL) 1.0",
        "usage_restriction": "Attribute OpenFlights; derived databases made public must be ODbL (share-alike). "
                             "Route data is historical (last major update 2014), not a live schedule.",
    },
    "enron": {
        "name": "Enron email corpus (CMU release), Hugging Face parquet mirror corbt/enron-emails",
        "source_tag": "enron",
        "landing": "https://huggingface.co/datasets/corbt/enron-emails ; https://www.cs.cmu.edu/~enron/",
        "revision": "cfc06c758093d90993abce1a43668fb7357258a6",
        "license": "No explicit license; emails were made public by the US FERC during its investigation and "
                   "redistributed by CMU for research.",
        "usage_restriction": "Distractor text only. 20,000 emails sampled (seeded); addresses, phone numbers and URLs "
                             "replaced by [EMAIL]/[PHONE]/[URL]; sender/recipient fields dropped.",
    },
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


MIN_TIMESTAMP = "2005-01-01"  # Google Local carries a few placeholder dates (1990-12-31); drop anything earlier


def merge_items():
    if not (STAGING / "items_product.jsonl").exists():  # staging already merged and deleted: re-read final table
        return list(read_jsonl(RAW / "items.jsonl"))
    rows = [*read_jsonl(STAGING / "items_product.jsonl"), *read_jsonl(STAGING / "items_restaurant.jsonl")]
    for r in rows:
        assert list(r) == ITEM_KEYS and r["domain"] in ("product", "restaurant")
        assert isinstance(r["attributes"], dict)
    with (RAW / "items.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def merge_interactions(item_domain):
    names = ("interactions_product.jsonl", "interactions_restaurant.jsonl")
    if (STAGING / names[0]).exists():
        rows = [r for name in names for r in read_jsonl(STAGING / name)]
    else:
        rows = list(read_jsonl(RAW / "interactions.jsonl"))
    per_user = defaultdict(list)
    for r in rows:
        assert list(r) == INTER_KEYS and HEX16.match(r["user_id"]) and ISO.match(r["timestamp"])
        assert isinstance(r["rating"], int) and 1 <= r["rating"] <= 5 and len(r["text"]) <= TEXT_MAX
        assert r["item_id"] in item_domain, r["item_id"]
        if r["timestamp"] >= MIN_TIMESTAMP:
            per_user[(item_domain[r["item_id"]][0], r["user_id"])].append(r)
    per_user = {k: v for k, v in per_user.items() if len(v) >= MIN_INTERACTIONS}
    with (RAW / "interactions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            if r["timestamp"] >= MIN_TIMESTAMP and (item_domain[r["item_id"]][0], r["user_id"]) in per_user:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return per_user


def domain_stats(domain, per_user, items, item_domain):
    users = {u: rs for (d, u), rs in per_user.items() if d == domain}
    inter = [r for rs in users.values() for r in rs]
    counts = [len(rs) for rs in users.values()]
    ts = [r["timestamp"] for r in inter]
    return {
        "items": sum(1 for i in items if i["domain"] == domain),
        "items_with_interactions": len({r["item_id"] for r in inter}),
        "interactions": len(inter),
        "users": len(users),
        f"users_with_ge_{MIN_INTERACTIONS}_interactions": sum(c >= MIN_INTERACTIONS for c in counts),
        "users_with_mixed_ratings_(<=3_and_>=4)": sum(
            1 for rs in users.values() if min(r["rating"] for r in rs) <= 3 and max(r["rating"] for r in rs) >= 4),
        "interactions_per_user": {"min": min(counts), "median": statistics.median(counts), "max": max(counts)},
        "time_range": [min(ts), max(ts)],
        "rating_distribution": dict(sorted(Counter(r["rating"] for r in inter).items())),
        "interactions_with_text": sum(1 for r in inter if r["text"]),
        "items_per_category": dict(Counter(i["category"] for i in items if i["domain"] == domain).most_common()),
        "interactions_per_category": dict(Counter(item_domain[r["item_id"]][1] for r in inter).most_common()),
        "items_per_city": dict(Counter(i["city"] for i in items if i["domain"] == domain).most_common(30))
        if domain == "restaurant" else None,
    }


def licenses():
    manifest = [json.loads(line) for line in (DL / "MANIFEST.jsonl").read_text().splitlines()]
    latest = {}
    for m in manifest:
        latest[m["url"]] = m
    out = {"seed": SEED, "sources": {}, "skipped": [
        {"source": "Yelp Open Dataset", "reason": "requires a web form and accepting terms; not attempted."},
        {"source": "UCSD Google Local full per-state review files", "reason": "10-core files used instead (smaller, "
         "same schema); full files not needed."},
    ]}
    for key, info in SOURCES.items():
        files = [{k: m[k] for k in ("file", "url", "bytes", "sha256", "server_last_modified", "downloaded_at")}
                 for m in latest.values() if m["group"] == key]
        out["sources"][key] = {**info, "files": files, "total_bytes": sum(f["bytes"] for f in files),
                               "raw_files_deleted_after_normalizing": not any((DL / key / f["file"]).exists()
                                                                              for f in files)}
    return out


def main() -> None:
    items = merge_items()
    item_domain = {i["item_id"]: (i["domain"], i["category"]) for i in items}
    per_user = merge_interactions(item_domain)
    emails = list(read_jsonl(RAW / "emails.jsonl"))
    assert all(len(e["body"]) <= EMAIL_BODY_MAX for e in emails)
    dates = sorted(e["date"] for e in emails if e["date"])
    stats = {
        "seed": SEED,
        "row_counts": {name: sum(1 for _ in (RAW / f"{name}.jsonl").open()) for name in
                       ("items", "interactions", "airports", "airlines", "routes", "emails")},
        "domains": {d: domain_stats(d, per_user, items, item_domain) for d in ("product", "restaurant")},
        "users_with_ge_8_interactions_in_one_domain": len({u for (d, u), rs in per_user.items() if len(rs) >= 8}),
        "emails": {"date_range": [dates[0], dates[-1]] if dates else None},
    }
    (RAW / "STATS.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    (RAW / "LICENSES.json").write_text(json.dumps(licenses(), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats["row_counts"]), stats["users_with_ge_8_interactions_in_one_domain"])


if __name__ == "__main__":
    main()
