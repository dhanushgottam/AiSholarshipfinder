"""Fictional demo profiles. Swap the profile and the same agents adapt — no code changes."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
LEGACY_SAMPLES_DIR = PROJECT_ROOT / "sample_data"

PRESETS = {
    "Rahul Sharma (B.Tech CS → AI)": {
        "name": "Rahul Sharma", "degree": "B.Tech Computer Science", "cgpa": "8.2 / 10", "grad": "2027",
        "nationality": "Indian", "field": "Computer Science / Artificial Intelligence", "level": "Master's",
        "countries": ["India", "Canada", "Germany", "Singapore"], "github": "https://github.com/octocat",
        "ielts": "Not taken / unknown", "funding": "Full funding needed", "intake": "Fall 2027",
        "files": ["rahul_sample_resume.txt", "rahul_sample_marksheet.txt"],
    },
    "Ananya Verma (B.Sc Physics → Data Science)": {
        "name": "Ananya Verma", "degree": "B.Sc (Hons) Physics", "cgpa": "9.1 / 10", "grad": "2027",
        "nationality": "Indian", "field": "Data Science / Computational Neuroscience", "level": "Master's",
        "countries": ["Netherlands", "Sweden", "Ireland"], "github": "",
        "ielts": "IELTS completed", "funding": "Tuition scholarship needed", "intake": "Fall 2027",
        "files": ["ananya_sample_resume.txt"],
    },
}


def sample_text(preset: dict) -> str:
    """Load bundled fictional sample documents; tolerate missing files gracefully."""
    documents = []
    for name in preset.get("files", []):
        candidates = (SAMPLES_DIR / name, LEGACY_SAMPLES_DIR / name)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path:
            content = path.read_text(encoding="utf-8")
            documents.append(f"=== FILE: {name} ===\n{content}")
        else:
            documents.append(
                f"=== FILE: {name} ===\n"
                "Sample document is missing. This is a fictional profile; no unsupported details should be inferred."
            )
    return "\n\n".join(documents)
