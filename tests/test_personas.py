from pathlib import Path

import pytest
import yaml

from gspaenli.config import ROOT
from gspaenli.personas import load_personas

PERSONAS_DIR = ROOT / "personas"


def test_loads_the_three_personas():
    personas = load_personas(PERSONAS_DIR)
    assert set(personas) == {"newcomer", "fedlex_clerk", "buenzli"}
    for persona in personas.values():
        assert len(persona.example_questions) == 3
        assert persona.preferred_tools
        assert "Never invent figures" in persona.system_prompt


def _write(directory: Path, name: str, **overrides) -> None:
    data = {
        "id": "test",
        "name": "Test",
        "tagline": "t",
        "system_prompt": "s",
        "preferred_tools": ["search_knowledge"],
        "example_questions": ["a", "b", "c"],
        "language_hint": "h",
    }
    data.update(overrides)
    (directory / name).write_text(yaml.safe_dump(data), encoding="utf-8")


def test_adding_a_file_adds_a_persona(tmp_path):
    _write(tmp_path, "a.yaml", id="alpha")
    _write(tmp_path, "b.yaml", id="beta")
    assert list(load_personas(tmp_path)) == ["alpha", "beta"]


def test_rejects_wrong_number_of_examples(tmp_path):
    _write(tmp_path, "a.yaml", example_questions=["only one"])
    with pytest.raises(ValueError):
        load_personas(tmp_path)


def test_rejects_missing_field(tmp_path):
    (tmp_path / "a.yaml").write_text("id: x\nname: X\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_personas(tmp_path)


def test_rejects_duplicate_ids(tmp_path):
    _write(tmp_path, "a.yaml", id="same")
    _write(tmp_path, "b.yaml", id="same")
    with pytest.raises(ValueError, match="duplicate"):
        load_personas(tmp_path)


def test_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no personas"):
        load_personas(tmp_path)
