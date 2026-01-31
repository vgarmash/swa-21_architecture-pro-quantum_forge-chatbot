import pathlib
import re


BASE_PATH = pathlib.Path(__file__).resolve().parents[1] / "knowledge_base"


def clean_knowledge_base(folder: pathlib.Path | str | None = None) -> None:
    base_path = pathlib.Path(folder) if folder else BASE_PATH
    for file_path in base_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        if text.startswith("---"):
            match = re.search(r"^---\s*\n.*?\n---\s*\n?", text, flags=re.S)
            if match:
                text = text[match.end():]

        markers = (
            "**Reference documents:**",
            "**Related",
            "## Related Documents",
        )
        marker_index = min((text.find(marker) for marker in markers if text.find(marker) != -1), default=-1)
        if marker_index != -1:
            text = text[:marker_index].rstrip() + "\n"

        file_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    clean_knowledge_base()
