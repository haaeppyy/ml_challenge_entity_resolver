import time
import duckdb
import polars as pl

DATA = "data"


def sizes(split, src):
    return f"read_parquet('{DATA}/norm_{split}_s{src}.parquet')"


def gt_pairs(split, src):
    df = pl.read_csv(f"dataset/{split}/{split}_ground_truth.tsv", separator="\t")
    return (
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


def measure(con, split, src):
    print(f"== {split} s1xs{src} ==", flush=True)
    pairs = gt_pairs(split, src)
    con.register("gtp", pairs.to_arrow())
    gt_total = pairs.height
    totals = {1: pl.scan_parquet(f"{DATA}/norm_{split}_s1.parquet").select(pl.len()).collect().item(),
              src: pl.scan_parquet(f"{DATA}/norm_{split}_s{src}.parquet").select(pl.len()).collect().item()}

    for cutoff in (0.01, 0.03, 0.08):
        t1 = time.time()
        freq = {}
        for tag, srcx in (("a", 1), ("b", src)):
            fr = con.execute(
                f"SELECT tok FROM "
                f"(SELECT tok, COUNT(*) n FROM (SELECT UNNEST(name_tokens) tok FROM {sizes(split,srcx)}) "
                f"GROUP BY tok) WHERE n > {int(totals[srcx]*cutoff)}").fetchdf()["tok"].tolist()
            con.execute(f"DROP VIEW IF EXISTS f_{tag}")
            quoted = ", ".join("'" + w.replace("'", "''") + "'" for w in fr)
            con.execute(
                f"CREATE VIEW f_{tag} AS "
                f"SELECT entity_id, tok FROM "
                f"(SELECT entity_id, UNNEST(name_tokens) tok FROM {sizes(split,srcx)}) "
                f"WHERE tok NOT IN ({quoted})"
            )
        nums = con.execute(
            "SELECT COUNT(*) FROM f_a x JOIN f_b y ON x.tok=y.tok AND x.entity_id<>y.entity_id"
        ).fetchone()[0]
        # recall of this single rule: count of gt pairs that share a rare token (streaming count of distinct gt pairs via join+count-distinct)
        recap = con.execute(
            "SELECT COUNT(DISTINCT g.e1 || '~' || g.e2) FROM gtp g "
            "WHERE EXISTS (SELECT 1 FROM f_a x JOIN f_b y ON x.tok=y.tok "
            "WHERE x.entity_id=g.e1 AND y.entity_id=g.e2)"
        ).fetchone()[0]
        print(f"  rare-tok overlap cutoff={cutoff}: {nums:,} pairs, "
              f"recall={recap:,}/{gt_total:,} = {100*recap/max(gt_total,1):.2f}% "
              f"({time.time()-t1:.0f}s)", flush=True)


if __name__ == "__main__":
    con = duckdb.connect()
    con.execute("SET memory_limit='9GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    for split in ("train",):
        for src in (2, 3):
            measure(con, split, src)