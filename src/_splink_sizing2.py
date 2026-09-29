import time
import duckdb

DATA = "data"


def sizes(split, src):
    return f"read_parquet('{DATA}/norm_{split}_s{src}.parquet')"


def count_rules(con, split, src):
    t = time.time()
    res = {}
    sql_a = (f"SELECT COUNT(*) FROM {sizes(split,1)} a JOIN {sizes(split,src)} b "
             "ON a.country_norm=b.country_norm AND substr(a.name_norm,1,5)=substr(b.name_norm,1,5)")
    res["country+substr5"] = con.execute(sql_a).fetchone()[0]

    sql_b = (f"SELECT COUNT(*) FROM {sizes(split,1)} a JOIN {sizes(split,src)} b "
             "ON a.country_norm=b.country_norm AND substr(a.name_norm,1,6)=substr(b.name_norm,1,6)")
    res["country+substr6"] = con.execute(sql_b).fetchone()[0]

    # first token overlap (regexp_extract of first word of name_norm with country)
    sql_c = (f"SELECT COUNT(*) FROM "
             f"(SELECT entity_id, country_norm, regexp_extract(name_norm, '[^ ]+', 0) AS ft FROM {sizes(split,1)}) a "
             f"JOIN (SELECT entity_id, country_norm, regexp_extract(name_norm, '[^ ]+', 0) AS ft FROM {sizes(split,src)}) b "
             "ON a.country_norm=b.country_norm AND a.ft=b.ft")
    res["country+firsttoken"] = con.execute(sql_c).fetchone()[0]

    # first two tokens (sorted) as full string
    sql_d = (f"SELECT COUNT(*) FROM "
             f"(SELECT entity_id, country_norm, list_sort(name_tokens[1:2]) AS f2 FROM {sizes(split,1)}) a "
             f"JOIN (SELECT entity_id, country_norm, list_sort(name_tokens[1:2]) AS f2 FROM {sizes(split,src)}) b "
             "ON a.country_norm=b.country_norm AND a.f2=b.f2")
    res["country+first2sorted"] = con.execute(sql_d).fetchone()[0]

    print(f"  {split} s1xs{src}: " + "  ".join(f"{k}={v:,}" for k, v in res.items())
          + f"  ({time.time()-t:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    con = duckdb.connect()
    for split in ("train", "test"):
        for src in (2, 3):
            count_rules(con, split, src)