import hashlib
import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "download_models", Path(__file__).parents[2] / "scripts" / "download_models.py"
)
download_models = importlib.util.module_from_spec(_spec)
sys.modules["download_models"] = download_models  # dataclasses look the module up here
_spec.loader.exec_module(download_models)


def _pin_for(content: bytes) -> "download_models.PinnedModel":
    return download_models.PinnedModel(
        purpose="ci",
        repo_id="example/repo",
        filename="model.gguf",
        revision="0" * 40,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def test_file_matching_pin_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"pinned bytes")
    assert download_models.is_valid(_pin_for(b"pinned bytes"), path)


def test_same_size_different_bytes_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"tampered!!!!")  # same length as the pinned content
    assert not download_models.is_valid(_pin_for(b"pinned bytes"), path)


def test_missing_file_is_invalid(tmp_path: Path) -> None:
    assert not download_models.is_valid(_pin_for(b"x"), tmp_path / "absent.gguf")


def test_every_pin_is_well_formed() -> None:
    for m in download_models.MODELS:
        assert m.purpose in {"ci", "dev", "bench"}
        assert len(m.revision) == 40 and len(m.sha256) == 64
        assert m.filename.endswith(".gguf") and m.size_bytes > 0
