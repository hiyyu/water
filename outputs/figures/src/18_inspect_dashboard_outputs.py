from pathlib import Path
import json
import pandas as pd

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_DIR / "outputs" / "evaluation"

REPORT_FILE = EVALUATION_DIR / "dashboard_source_inventory.txt"


def main():
    if not EVALUATION_DIR.exists():
        print(f"ERROR: Folder not found: {EVALUATION_DIR}")
        return

    lines = []
    lines.append("DASHBOARD SOURCE FILE INVENTORY")
    lines.append("=" * 70)
    lines.append(f"Evaluation folder: {EVALUATION_DIR}")
    lines.append("")

    files = sorted(EVALUATION_DIR.iterdir())

    if not files:
        print("No evaluation files were found.")
        return

    for file_path in files:
        if file_path.suffix.lower() not in {".csv", ".json", ".txt"}:
            continue

        lines.append("")
        lines.append("=" * 70)
        lines.append(f"FILE: {file_path.name}")
        lines.append("=" * 70)

        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)

                lines.append(f"Rows: {len(df)}")
                lines.append(f"Columns: {len(df.columns)}")
                lines.append("")
                lines.append("Column names and data types:")

                for column in df.columns:
                    lines.append(
                        f"  - {column} ({df[column].dtype})"
                    )

                lines.append("")
                lines.append("First 3 rows:")
                lines.append(
                    df.head(3).to_string(index=False)
                )

            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                lines.append(
                    json.dumps(data, indent=2, ensure_ascii=False)
                )

            elif file_path.suffix.lower() == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                lines.append(content[:5000])

        except Exception as error:
            lines.append(f"ERROR reading file: {error}")

    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")

    print(report)
    print("\n" + "=" * 70)
    print(f"Inventory saved to:\n{REPORT_FILE}")


if __name__ == "__main__":
    main()