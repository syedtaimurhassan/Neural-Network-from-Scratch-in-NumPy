import csv
from pathlib import Path


class CSVLogger:
    """Minimal CSV logger for metrics."""

    def __init__(self, path: Path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        write_header = not self.path.exists()
        self.file = open(self.path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if write_header:
            self.writer.writeheader()

    def log(self, row: dict):
        filtered = {k: row.get(k, None) for k in self.fieldnames}
        self.writer.writerow(filtered)
        self.file.flush()

    def close(self):
        self.file.close()
