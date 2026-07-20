import os

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY tidak diset — skip smoke test panggilan LLM nyata.",
)


def test_generate_returns_valid_json():
    from app.reasoning.llm_client import generate

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    result = generate(
        prompt="Balas dengan JSON {\"ok\": true} saja, tanpa penjelasan apapun.",
        json_schema=schema,
        schema_name="ok_check",
        max_tokens=50,
    )

    assert isinstance(result, dict)
    assert "ok" in result
    assert isinstance(result["ok"], bool)
