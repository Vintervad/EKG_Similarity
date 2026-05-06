from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_NUMERIC_COLUMNS = [
    "VentricularRate",
    "AtrialRate",
    "AvgRRInterval",
    "PR_Interval",
    "QRSDuration",
    "QT_Interval",
    "QTcBazett",
    "QTcFridericia",
    "QTcFramingham",
    "P_Frontal_Axis",
    "R_Frontal_Axis",
    "T_Frontal_Axis",
]
DEFAULT_CATEGORICAL_COLUMNS = ["Gender", "Location"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FAISS top-k ECG results against a large ECG metadata file."
    )
    parser.add_argument(
        "--topk-results",
        type=str,
        required=True,
        help="CSV from retrieve.py --output-csv containing query_id and neighbor_id columns.",
    )
    parser.add_argument("--metadata-file", type=str, required=True)
    parser.add_argument("--metadata-sep", type=str, default=";")
    parser.add_argument("--metadata-id-column", type=str, default="TestID")
    parser.add_argument("--statement-column", type=str, default="Statements")
    parser.add_argument("--patient-column", type=str, default="PatientID")
    parser.add_argument("--numeric-columns", type=str, default=",".join(DEFAULT_NUMERIC_COLUMNS))
    parser.add_argument("--categorical-columns", type=str, default=",".join(DEFAULT_CATEGORICAL_COLUMNS))
    parser.add_argument("--metadata-chunksize", type=int, default=500000)
    parser.add_argument("--output-dir", type=str, default="embedding_metadata_eval")
    return parser.parse_args()


def _split_columns(value: str) -> list[str]:
    return [column.strip() for column in value.split(",") if column.strip()]


def _normalize_test_id(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("TestID_"):
        return text
    if text.startswith("Test_ID_"):
        return "TestID_" + text.removeprefix("Test_ID_")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return f"TestID_{text}"
    return text


def _statement_codes(value: Any) -> set[str]:
    return set(re.findall(r"\d+", str(value)))


def _primary_statement(value: Any) -> str:
    codes = re.findall(r"\d+", str(value))
    return codes[0] if codes else "no_statement"


def _jaccard(left: set[str], right: set[str]) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _load_topk_results(path: Path) -> pd.DataFrame:
    topk = pd.read_csv(path, dtype=str)
    if "query_id" not in topk.columns:
        raise KeyError(f"{path} must contain a query_id column.")
    if "neighbor_id" not in topk.columns:
        if "id" in topk.columns:
            topk = topk.rename(columns={"id": "neighbor_id"})
        else:
            raise KeyError(f"{path} must contain a neighbor_id column.")
    topk["query_id"] = topk["query_id"].map(_normalize_test_id)
    topk["neighbor_id"] = topk["neighbor_id"].map(_normalize_test_id)
    if "rank" in topk.columns:
        topk["rank"] = pd.to_numeric(topk["rank"], errors="coerce").astype("Int64")
    return topk


def _load_needed_metadata(
    *,
    metadata_file: Path,
    sep: str,
    id_column: str,
    statement_column: str,
    patient_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    needed_ids: set[str],
    chunksize: int,
) -> dict[str, dict[str, Any]]:
    header = pd.read_csv(metadata_file, sep=sep, nrows=0)
    available_columns = set(header.columns)
    if id_column not in available_columns:
        raise KeyError(f"Metadata file is missing ID column {id_column!r}. Found: {list(header.columns)}")

    requested_columns = [id_column, statement_column, patient_column, *numeric_columns, *categorical_columns]
    usecols = [column for column in requested_columns if column in available_columns]
    missing = sorted(set(requested_columns) - set(usecols))
    if missing:
        print(f"metadata_missing_columns={missing}")

    metadata: dict[str, dict[str, Any]] = {}
    for chunk in pd.read_csv(
        metadata_file,
        sep=sep,
        usecols=usecols,
        dtype=str,
        chunksize=chunksize,
    ):
        normalized_ids = chunk[id_column].map(_normalize_test_id)
        mask = normalized_ids.isin(needed_ids)
        if not mask.any():
            continue
        filtered = chunk.loc[mask].copy()
        filtered["_normalized_id"] = normalized_ids.loc[mask]
        filtered = filtered.drop_duplicates("_normalized_id", keep="first")
        for row in filtered.to_dict("records"):
            normalized_id = row.pop("_normalized_id")
            metadata.setdefault(normalized_id, row)
        if len(metadata) >= len(needed_ids):
            break
    return metadata


def _float_value(metadata: dict[str, Any] | None, column: str) -> float | None:
    if metadata is None or column not in metadata:
        return None
    value = metadata[column]
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _same_value(left: dict[str, Any] | None, right: dict[str, Any] | None, column: str) -> float | None:
    if left is None or right is None or column not in left or column not in right:
        return None
    left_value = str(left[column]).strip()
    right_value = str(right[column]).strip()
    if left_value == "" or right_value == "" or left_value.lower() == "nan" or right_value.lower() == "nan":
        return None
    return float(left_value == right_value)


def _build_eval_row(
    pair: dict[str, Any],
    *,
    query_meta: dict[str, Any] | None,
    neighbor_meta: dict[str, Any] | None,
    statement_column: str,
    patient_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    if query_meta is not None and neighbor_meta is not None:
        query_statement = query_meta.get(statement_column, "")
        neighbor_statement = neighbor_meta.get(statement_column, "")
        query_codes = _statement_codes(query_statement)
        neighbor_codes = _statement_codes(neighbor_statement)
        query_primary = _primary_statement(query_statement)
        neighbor_primary = _primary_statement(neighbor_statement)
        same_primary_statement = float(query_primary == neighbor_primary)
        statement_jaccard = _jaccard(query_codes, neighbor_codes)
        statement_overlap_count = len(query_codes & neighbor_codes)
        query_statement_count = len(query_codes)
        neighbor_statement_count = len(neighbor_codes)
    else:
        query_primary = None
        neighbor_primary = None
        same_primary_statement = None
        statement_jaccard = None
        statement_overlap_count = None
        query_statement_count = None
        neighbor_statement_count = None

    row = dict(pair)
    row.update(
        {
            "query_metadata_found": query_meta is not None,
            "neighbor_metadata_found": neighbor_meta is not None,
            "query_primary_statement": query_primary,
            "neighbor_primary_statement": neighbor_primary,
            "same_primary_statement": same_primary_statement,
            "statement_jaccard": statement_jaccard,
            "statement_overlap_count": statement_overlap_count,
            "query_statement_count": query_statement_count,
            "neighbor_statement_count": neighbor_statement_count,
            "same_patient": _same_value(query_meta, neighbor_meta, patient_column),
        }
    )

    for column in categorical_columns:
        row[f"same_{column}"] = _same_value(query_meta, neighbor_meta, column)

    for column in numeric_columns:
        query_value = _float_value(query_meta, column)
        neighbor_value = _float_value(neighbor_meta, column)
        row[f"query_{column}"] = query_value
        row[f"neighbor_{column}"] = neighbor_value
        row[f"abs_{column}_diff"] = (
            abs(query_value - neighbor_value)
            if query_value is not None and neighbor_value is not None
            else None
        )
    return row


def _write_summaries(detail: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = [
        column
        for column in detail.columns
        if column.startswith("same_")
        or column.startswith("abs_")
        or column in {"statement_jaccard", "statement_overlap_count"}
    ]
    summary = detail[metric_columns].mean(numeric_only=True).to_frame(name="mean")
    summary.insert(0, "pairs", len(detail))
    summary.to_csv(output_dir / "summary_overall.csv")

    if "rank" in detail.columns:
        by_rank = detail.groupby("rank", dropna=False)[metric_columns].mean(numeric_only=True)
        by_rank.insert(0, "pairs", detail.groupby("rank", dropna=False).size())
        by_rank.to_csv(output_dir / "summary_by_rank.csv")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_columns = _split_columns(args.numeric_columns)
    categorical_columns = _split_columns(args.categorical_columns)

    topk = _load_topk_results(Path(args.topk_results))
    needed_ids = set(topk["query_id"]) | set(topk["neighbor_id"])
    metadata = _load_needed_metadata(
        metadata_file=Path(args.metadata_file),
        sep=args.metadata_sep,
        id_column=args.metadata_id_column,
        statement_column=args.statement_column,
        patient_column=args.patient_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        needed_ids=needed_ids,
        chunksize=args.metadata_chunksize,
    )

    rows: list[dict[str, Any]] = []
    for pair in topk.to_dict("records"):
        query_id = pair["query_id"]
        neighbor_id = pair["neighbor_id"]
        rows.append(
            _build_eval_row(
                pair,
                query_meta=metadata.get(query_id),
                neighbor_meta=metadata.get(neighbor_id),
                statement_column=args.statement_column,
                patient_column=args.patient_column,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
            )
        )

    detail = pd.DataFrame(rows)
    detail_path = output_dir / "topk_metadata_pairs.csv"
    detail.to_csv(detail_path, index=False)
    _write_summaries(detail, output_dir)

    print(f"topk_results={args.topk_results}")
    print(f"metadata_rows_loaded={len(metadata)}, needed_ids={len(needed_ids)}")
    print(f"saved_detail={detail_path}")
    print(f"saved_summary={output_dir / 'summary_overall.csv'}")
    if (output_dir / "summary_by_rank.csv").exists():
        print(f"saved_summary_by_rank={output_dir / 'summary_by_rank.csv'}")


if __name__ == "__main__":
    main()
