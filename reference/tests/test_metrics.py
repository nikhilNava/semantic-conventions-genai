"""Verify the committed data files survive into the reports.

How a run is reduced into ``data.json`` is the conformance runner's business
and is tested there; what is left here is this repo's own view of it -- the
specs the reports are built from, and the mapping from the registry names a
data file uses onto the shorter keys the reports address signals by.

Runnable directly (``python tests/test_metrics.py``) or under pytest.
"""

from __future__ import annotations

import json
from pathlib import Path

from semconv_genai.data_files import _normalize_scenario_data_entry, load_scenario_data_files
from semconv_genai.semconv_model import metric_specs, span_specs

_TOOL_CALLS = "gen_ai.invoke_agent.tool_calls"
_INFERENCE_CALLS = "gen_ai.invoke_agent.inference_calls"
_TRANSFER_ATTRIBUTES = {
    "gen_ai.transfer.mode",
    "gen_ai.transfer.target.name",
    "gen_ai.transfer.target.type",
}


def test_metric_specs_expose_recommended_agent_name():
    specs = metric_specs()
    assert specs, "expected at least one tracked metric"
    # Only the per-invocation agent metrics are agent-scoped; gen_ai.client.*
    # metrics (token usage, operation duration) are not dimensioned by
    # gen_ai.agent.name, so this checks the invoke_agent metrics specifically
    # rather than every tracked metric.
    for name in (_INFERENCE_CALLS, _TOOL_CALLS):
        assert "gen_ai.agent.name" in specs[name].recommended, name


def test_metric_specs_are_named_as_the_registry_names_them():
    for name, spec in metric_specs().items():
        assert spec.registry_id == name, name


def test_committed_google_adk_metrics_round_trip():
    entries = {e.library: e for e in load_scenario_data_files()}
    adk = entries["google-adk"]
    for name in (_INFERENCE_CALLS, _TOOL_CALLS):
        assert adk.metrics[name]["gen_ai.agent.name"] == "present", name


def test_registry_span_names_map_onto_report_keys():
    """A data file names spans as the registry does; reports use short keys."""
    entry = _normalize_scenario_data_entry(
        {"spans": {"gen_ai.inference.client": ["gen_ai.operation.name"]}},
        "fake",
    )
    assert set(entry.spans) == {"inference"}
    assert entry.spans["inference"]["gen_ai.operation.name"] == "present"
    # Everything else the span type declares is reported as absent, not missing.
    assert entry.spans["inference"]["gen_ai.provider.name"] == "absent"


def test_events_keep_their_registry_names():
    entry = _normalize_scenario_data_entry(
        {"events": {"gen_ai.evaluation.result": ["gen_ai.evaluation.name"]}},
        "fake",
    )
    assert entry.events["gen_ai.evaluation.result"]["gen_ai.evaluation.name"] == "present"


def test_span_types_absent_from_a_data_file_are_not_reported():
    entry = _normalize_scenario_data_entry({"spans": {}}, "fake")
    assert entry.spans == {}


def test_span_specs_are_named_as_the_registry_names_them():
    for key, spec in span_specs().items():
        assert spec.registry_id.startswith("gen_ai."), key


def test_committed_langchain_transfer_coverage():
    entries = {entry.library: entry for entry in load_scenario_data_files()}
    langchain = entries["langchain"]
    execute_tool = langchain.spans["execute_tool"]

    for attribute in _TRANSFER_ATTRIBUTES:
        assert execute_tool[attribute] == "present", attribute
    assert "gen_ai.transfer.target.id" not in execute_tool
    assert not any(
        attribute.startswith("gen_ai.transfer.")
        for attribute in langchain.spans["invoke_agent_internal"]
    )


def test_interaction_type_is_removed_from_committed_scenarios():
    for path in (Path(__file__).parents[1] / "scenarios").glob("*/data.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "gen_ai.agent.interaction.type" not in json.dumps(data), path


if __name__ == "__main__":
    test_metric_specs_expose_recommended_agent_name()
    test_metric_specs_are_named_as_the_registry_names_them()
    test_committed_google_adk_metrics_round_trip()
    test_registry_span_names_map_onto_report_keys()
    test_events_keep_their_registry_names()
    test_span_types_absent_from_a_data_file_are_not_reported()
    test_span_specs_are_named_as_the_registry_names_them()
    test_committed_langchain_transfer_coverage()
    test_interaction_type_is_removed_from_committed_scenarios()
    print("ok")
