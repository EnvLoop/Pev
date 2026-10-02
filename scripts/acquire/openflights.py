"""Normalize OpenFlights airports/airlines/routes .dat files into JSONL with named fields."""
import csv

from common import DL, RAW, write_jsonl

SRC = "openflights"


def val(x: str):
    return None if x in ("\\N", "") else x


def num(x: str, cast):
    x = val(x)
    try:
        return None if x is None else cast(x)
    except ValueError:
        return None


def rows(name: str):
    with (DL / "openflights" / f"{name}.dat").open(encoding="utf-8", newline="") as f:
        yield from csv.reader(f)


def airports():
    for r in rows("airports"):
        yield {
            "airport_id": num(r[0], int), "name": val(r[1]), "city": val(r[2]), "country": val(r[3]),
            "iata": val(r[4]), "icao": val(r[5]), "latitude": num(r[6], float), "longitude": num(r[7], float),
            "altitude_ft": num(r[8], int), "utc_offset": num(r[9], float), "dst": val(r[10]),
            "tz_database": val(r[11]), "type": val(r[12]), "data_source": val(r[13]), "source": SRC,
        }


def airlines():
    for r in rows("airlines"):
        yield {
            "airline_id": num(r[0], int), "name": val(r[1]), "alias": val(r[2]), "iata": val(r[3]),
            "icao": val(r[4]), "callsign": val(r[5]), "country": val(r[6]), "active": r[7] == "Y", "source": SRC,
        }


def routes():
    for r in rows("routes"):
        yield {
            "airline": val(r[0]), "airline_id": num(r[1], int), "src_airport": val(r[2]),
            "src_airport_id": num(r[3], int), "dst_airport": val(r[4]), "dst_airport_id": num(r[5], int),
            "codeshare": r[6] == "Y", "stops": num(r[7], int), "equipment": r[8].split() if r[8] else [],
            "source": SRC,
        }


if __name__ == "__main__":
    for name, gen in (("airports", airports), ("airlines", airlines), ("routes", routes)):
        print(name, write_jsonl(RAW / f"{name}.jsonl", gen()))
