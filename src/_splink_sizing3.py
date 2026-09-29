import time
import duckdb
import polars as pl
from src.blocking import _soundex

DATA = "data"


def sizes(split, src):
    return f"read_parquet('{DATA}/norm_{split}_s{src}.parquet')"


def soundex_col_sql(frame):
    # lazily map soundex in python on collected first tokens (only needed for count variant)
    pass


def gt_pairs(con, split, src):
    df = pl.read_csv(f"dataset/{split}/{split}_ground_truth.tsv", separator="\t")
    pairs = (
        df.filter(pl.col("matched_entity_ids").is_not_null())
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
        .explode("ids")
        .select(
            pl.col("source1_entity_id").alias("e1"),
            pl.col("ids").str.strip_chars().alias("e2"),
        )
        .filter(pl.col("e2").str.starts_with(f"S{src}-"))
        .unique()
    )
    return pairs


def measure(split, src, con):
    print(f"== {split} s1xs{src} ==", flush=True)
    t = time.time()
    # load small cols into polars for soundex (count + recall computed via duckdb temp tables)
    s1 = pl.read_parquet(f"{DATA}/norm_{split}_s1.parquet", columns=["entity_id", "country_norm", "name_tokens"])
    s2 = pl.read_parquet(f"{DATA}/norm_{split}_s{src}.parquet", columns=["entity_id", "country_norm", "name_tokens"])
    # soundex of first token
    s1 = s1.with_columns(
        pl.col("name_tokens").list.get(0, null_on_oob=True).map_elements(
            _soundex, return_dtype=pl.Utf8, skip_nulls=False).alias("sx"))
    s2 = s2.with_columns(
        pl.col("name_tokens").list.get(0, null_on_oob=True).map_elements(
            _soundex, return_dtype=pl.Utf8, skip_nulls=False).alias("sx"))
    s1 = s1.filter(pl.col("sx") != "").select(["entity_id", "country_norm", "name_tokens", "sx"])
    s2 = s2.filter(pl.col("sx") != "").select(["entity_id", "country_norm", "name_tokens", "sx"])
    con.register("sdx1", s1.to_arrow())
    con.register("sdx2", s2.to_arrow())
    t1 = time.time()
    n = con.execute(
        "SELECT COUNT(*) FROM sdx1 a JOIN sdx2 b "
        "ON a.country_norm=b.country_norm AND a.sx=b.sx").fetchone()[0]
    print(f"  country+sdx1 (uncapped): {n:,}  ({(time.time()-t1):.0f}s)", flush=True)

    # rare-token overlap: drop tokens with doc freq > cutoff on either side, explode, join
    pairs = gt_pairs(con, split, src)
    con.register("gtp", pairs.to_arrow())
    gt_total = pairs.height
    for cutoff in (0.005, 0.02):
        t1 = time.time()
        # build table of per-record rare tokens on each side
        tokens_sql = (
            f"SELECT entity_id, tok FROM "
            f"(SELECT entity_id, unnest(name_tokens) AS tok FROM {sizes(split,1)})")
        for tag, srcx in (("a", 1), ("b", src)):
            tot = pl.scan_parquet(f"{DATA}/norm_{split}_s{srcx}.parquet").select(pl.len()).collect().item()
            fr = con.execute(
                f"SELECT tok FROM (SELECT tok, COUNT(*) AS n FROM ({tokens_sql}) GROUP BY tok) "
                f"WHERE n > {int(tot*cutoff)}").fetchdf()["tok"].tolist()
            keep = con.execute(
                f"SELECT entity_id, tok FROM (SELECT entity_id, UNNEST(name_tokens) AS tok FROM {sizes(split,srcx)}) "
                f"WHERE tok NOT IN ({",".join("'" + w.replace("'", "''") + "'" for w in fr)})"
            ).fetch_arrow_table()
            con.register(f"rt_{tag}", keep)
        nums = con.execute(
            "SELECT COUNT(*) FROM rt_a x JOIN rt_b y ON x.tok=y.tok "
            "AND x.entity_id<>y.entity_id").fetchone()[0]
        # recall of this rule's candidate set
        cand = con.execute(
            "SELECT DISTINCT x.entity_id AS e1, y.entity_id AS e2 FROM rt_a x JOIN rt_b y "
            "ON x.tok=y.tok AND x.entity_id<>y.entity_id").fetch_arrow_table()
        con.register("cand", cand)
        cap = con.execute(
            "SELECT COUNT(*) FROM gtp g INNER JOIN cand c ON g.e1=c.e1 AND g.e2=c.e2").fetchone()[0]
        print(f"  rare-token-overlap cutoff={cutoff}: {nums:,} candidates, "
              f"recall={cap}/{gt_total}={100*cap/max(gt_total,1):.2f}%  ({(time.time()-t1):.0f}s)", flush=True)
        con.execute("DROP TABLE IF EXISTS cand")


if __name__ == "__main__":
    con = duckdb.connect()
    con.execute("SET threads=8")
    for split in ("train",):
        for src in (2, 3):
            measure(split, src, con)