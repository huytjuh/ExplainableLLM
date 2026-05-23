import json

import pytest

from azure_devops import AzurePipelineArtifact
from gemini_flash_lite import GeminiFlashLiteClient
from judge import HeuristicJudge, build_judge_prompt
from metrics import exact_match, token_f1
from recorder import TraceRecorder
from reports import write_evaluation_report
from vertex_ai import VertexAIConfig


def test_metrics_and_heuristic_judge():
    assert exact_match("A token.", "a token") == 1.0
    assert token_f1("retrieves context", "retrieves useful context") > 0.5
    result = HeuristicJudge().score("What is RAG?", "RAG retrieves context.", "RAG retrieves context for answers.")
    assert result.score >= 3.0
    assert "Return JSON" in build_judge_prompt("q", "a", "r")


def test_report_writer(tmp_path):
    output = write_evaluation_report([{"score": 4.0}, {"score": 2.0}], tmp_path / "report.json")
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["average_score"] == 3.0


def test_trace_recorder_writes_json(tmp_path):
    recorder = TraceRecorder("test-run")
    recorder.add_event("prompt", text="hello")
    path = recorder.write(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["name"] == "test-run"
    assert data["events"][0]["name"] == "prompt"


def test_integration_helpers(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    assert VertexAIConfig.from_env().resource_prefix == "projects/demo-project/locations/us-central1"
    assert "PublishPipelineArtifact" in AzurePipelineArtifact("reports", "artifacts/reports").publish_yaml()


def test_gemini_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        GeminiFlashLiteClient().generate("hello")
