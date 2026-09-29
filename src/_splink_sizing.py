import time
import duckdb

DATA = "data"


def sizes(split, src):
    return f"read_parquet('{DATA}/norm_{split}_s{src}.parquet')"


def rule_a_count(con, split, src):
    sql = (
        "SELECT COUNT(*) FROM "
        f"{sizes(split, 1)} a JOIN {sizes(split, src)} b "
        "ON a.country_norm=b.country_norm "
        "AND substr(a.name_norm,1,4)=substr(b.name_norm,1,4)"
    )
    return con.execute(sql).fetchone()[0]


def rule_b_count(con, split, src):
    # token-overlap: unnest name_tokens on each side, join on token
    l = (f"SELECT entity_id, unnest(name_tokens) AS tok FROM {sizes(split, 1)}")
    r = (f"SELECT entity_id, unnest(name_tokens) AS tok FROM {sizes(split, src)}")
    sql = (f"SELECT COUNT(*) FROM ({l}) a JOIN ({r}) b ON a.tok=b.tok "
           "AND a.entity_id <> b.entity_id")
    return con.execute(sql).fetchone()[0]


if __name__ == "__main__":
    con = duckdb.connect()
    for split in ("train", "test"):
        for src in (2, 3):
            t = time.time()
            na = rule_a_count(con, split, src)
            tb = time.time()
            nb = rule_b_count(con, split, src)
            print(f"{split} s1xs{src}: ruleA={na:,} "
                  f"({(tb-t):.0f}s)  ruleB(tok-overlap)={nb:,} "
                  f"({(time.time()-tb):.0f}s)", flush=True)