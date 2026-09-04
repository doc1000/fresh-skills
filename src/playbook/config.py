"""Filesystem layout for the playbook demo."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_DEMO = ROOT / "data" / "demo"
DATA_REFERENCE = ROOT / "data" / "reference"

ABCD_JSON = DATA_RAW / "abcd_v1.1.json"
ABCD_GZ = DATA_RAW / "abcd_v1.1.json.gz"
ONTOLOGY_JSON = DATA_RAW / "ontology.json"
GUIDELINES_JSON = DATA_RAW / "guidelines.json"
KB_JSON = DATA_RAW / "kb.json"

WEEKS_PARQUET = DATA_DEMO / "weeks.parquet"
FIXTURE_LABELS_PARQUET = DATA_REFERENCE / "fixture_labels.parquet"
DEMO_MANIFEST_JSON = DATA_REFERENCE / "demo_manifest.json"

WEEK_IDS = ("week_1", "week_2", "week_3")
