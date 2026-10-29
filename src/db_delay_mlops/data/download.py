"""Download monthly processed Deutsche Bahn data from Hugging Face.

Usage:
    uv run python -m db_delay_mlops.data.download --start 2026-09
    uv run python -m db_delay_mlops.data.download --start 2024-07 --end 2026-09
"""

import argparse
import logging
from datetime import date
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)

REPO_ID = "piebro/deutsche-bahn-data"
REPO_TYPE = "dataset"
REMOTE_DIR = "monthly_processed_data"
DEFAULT_DATA_DIR = Path("data/raw/hf")


def parse_month(value: str) -> date:
    """Parse 'YYYY-MM' into the first day of that month."""
    year, month = value.split("-")
    return date(int(year), int(month), 1)


def month_range(start: date, end: date) -> list[date]:
    """All months from start to end, both inclusive."""
    months = []
    current = start
    while current <= end:
        months.append(current)
        current = date(current.year + current.month // 12, current.month % 12 + 1, 1)
    return months


def resolve_revision(revision: str) -> str:
    """Turn a branch name like 'main' into a fixed commit hash.

    Pinning the commit makes a download reproducible: 'main' can change at any
    time, a commit hash never does.
    """
    info = HfApi().dataset_info(REPO_ID, revision=revision)
    return info.sha


def download_month(month: date, data_dir: Path, revision: str) -> Path:
    """Download one monthly file. Skips the download if the file is already there."""
    filename = f"{REMOTE_DIR}/data-{month:%Y-%m}.parquet"
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
        revision=revision,
        local_dir=data_dir,
    )
    logger.info("Ready: %s", path)
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, type=parse_month, help="First month, YYYY-MM")
    parser.add_argument("--end", type=parse_month, help="Last month, YYYY-MM (default: same as --start)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--revision", default="main", help="Branch, tag or commit of the dataset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one log line per HTTP request is noise

    revision = resolve_revision(args.revision)
    logger.info("Dataset %s at commit %s", REPO_ID, revision)

    for month in month_range(args.start, args.end or args.start):
        download_month(month, args.data_dir, revision)


if __name__ == "__main__":
    main()
