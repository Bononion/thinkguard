"""thinkguard.data.token_lengths.report_rows: nearest-rank percentiles and the share above each candidate."""

from thinkguard.data.token_lengths import HEADER, report_rows


def test_report_rows_uses_nearest_rank_and_counts_rows_above_each_candidate():
    lengths = [1000, 1100, 1600, 2100, *range(1, 7)]  # 10 rows, given unsorted
    (row,) = report_rows("train", {"total": lengths})
    got = dict(zip(HEADER, row, strict=True))
    # sorted: 1 2 3 4 5 6 1000 1100 1600 2100; p50 = sorted[ceil(5) - 1], p90 = sorted[8], p95 = p99 = sorted[9]
    assert (got["rows"], got["p50"], got["p90"], got["p95"], got["p99"], got["max"]) == (10, 5, 1600, 2100, 2100, 2100)
    assert (got["above_1024"], got["above_1536"], got["above_2048"]) == ("0.3000", "0.2000", "0.1000")
