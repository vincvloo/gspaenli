"""Persona loading. One YAML file per persona in personas/; adding a file adds a persona."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class Persona(BaseModel):
    id: str
    name: str
    tagline: str
    system_prompt: str
    preferred_tools: list[str] = Field(min_length=1)
    example_questions: list[str] = Field(min_length=3, max_length=3)
    language_hint: str

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not value.replace("_", "").isalnum() or value != value.lower():
            raise ValueError(f"persona id must be a lowercase slug, got {value!r}")
        return value


def load_personas(directory: Path) -> dict[str, Persona]:
    """Load every *.yaml in directory, keyed by id, in file-name order."""
    personas: dict[str, Persona] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        persona = Persona.model_validate(data)
        if persona.id in personas:
            raise ValueError(f"duplicate persona id {persona.id!r} in {path.name}")
        personas[persona.id] = persona
    if not personas:
        raise ValueError(f"no personas found in {directory}")
    return personas
