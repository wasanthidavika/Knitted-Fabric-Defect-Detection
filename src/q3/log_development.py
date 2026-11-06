"""Append a development or verification record to logs/development_test_log.csv.

Example (run from the project root):
python src/q3/log_development.py --activity "Q3 script import check" --status PASSED --details "All Q3 modules imported successfully" --evidence "SS17"
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs" / "development_test_log.csv"
FIELDS = ["date", "activity", "status", "details", "evidence"]


def append_record(activity: str, status: str, details: str, evidence: str = "") -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists_and_has_data = LOG_PATH.exists() and LOG_PATH.stat().st_size > 0

    with LOG_PATH.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        if not file_exists_and_has_data:
            writer.writeheader()

        writer.writerow(
            {
                "date": datetime.now().astimezone().isoformat(timespec="seconds"),
                "activity": activity,
                "status": status.upper(),
                "details": details,
                "evidence": evidence,
            }
        )

    print(f"Development record saved: {LOG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append one Q3 development or verification record."
    )
    parser.add_argument("--activity", required=True, help="Short name of the check")
    parser.add_argument(
        "--status", required=True, choices=["PASSED", "FAILED", "WARNING", "INFO"]
    )
    parser.add_argument("--details", required=True, help="What was checked and the outcome")
    parser.add_argument("--evidence", default="", help="Screenshot ID or evidence filename")
    args = parser.parse_args()

    append_record(args.activity, args.status, args.details, args.evidence)


if __name__ == "__main__":
    main()
