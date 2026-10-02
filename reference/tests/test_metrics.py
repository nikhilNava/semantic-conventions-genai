"""Verify the committed data files survive into the reports.

How a run is reduced into ``data.json`` is the conformance runner's business
and is tested there; what is left here is this repo's own view of it -- the
specs the reports are built from, and the mapping from the registry names a
data file uses onto the shorter keys the reports address signals by.

Runnable directly (``python tests/test_metrics.py``) or under pytest.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from semconv_genai.data_files import _normalize_scenario_data_entry, load_scenario_data_files
from semconv_genai.semconv_model import entity_specs, metric_specs, span_specs

_TOOL_CALLS = "gen_ai.invoke_agent.tool_calls"
_INFERENCE_CALLS = "gen_ai.invoke_agent.inference_calls"
_SEMCONV_ROOT = Path(__file__).resolve().parents[2]


def _load_semconv_yaml(relative_path: str) -> dict:
    return yaml.safe_load((_SEMCONV_ROOT / relative_path).read_text(encoding="utf-8"))


def _required_level(attribute: dict) -> tuple[str, str | None]:
    level = attribute["requirement_level"]
    if isinstance(level, str):
        return level, None
    name, condition = next(iter(level.items()))
    return name, condition


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


def test_entity_specs_expose_required_id():
    specs = entity_specs()
    assert "gen_ai.main_agent" in specs
    assert "gen_ai.main_agent.id" in specs["gen_ai.main_agent"].required


def test_entities_keep_their_registry_names():
    entry = _normalize_scenario_data_entry(
        {"entities": {"gen_ai.main_agent": ["gen_ai.main_agent.id"]}},
        "fake",
    )
    assert entry.entities["gen_ai.main_agent"]["gen_ai.main_agent.id"] == "present"


def test_entities_with_sections_round_trip():
    entry = _normalize_scenario_data_entry(
        {
            "entities": {
                "gen_ai.main_agent": {
                    "identity": ["gen_ai.main_agent.id"],
                    "description": ["gen_ai.main_agent.description", "gen_ai.main_agent.name"],
                }
            }
        },
        "fake",
    )
    assert entry.entities["gen_ai.main_agent"]["gen_ai.main_agent.id"] == "present"
    assert entry.entities["gen_ai.main_agent"]["gen_ai.main_agent.description"] == "present"
    assert entry.entities["gen_ai.main_agent"]["gen_ai.main_agent.name"] == "present"


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


def test_foundry_invoke_agent_refinement_contract():
    registry = _load_semconv_yaml("model/gen-ai/registry.yaml")
    provider = next(
        member
        for attribute in registry["attributes"]
        if attribute["key"] == "gen_ai.provider.name"
        for member in attribute["type"]["members"]
        if member["id"] == "azure.ai.foundry"
    )
    assert provider["value"] == "azure.ai.foundry"
    provider_doc = "https://learn.microsoft.com/azure/foundry/agents/overview"
    assert provider["brief"] == f"[Microsoft Foundry Agent Service]({provider_doc})"

    spans = _load_semconv_yaml("model/gen-ai/spans.yaml")
    refinement = next(
        item for item in spans["span_refinements"] if item["id"] == "azure.ai.foundry.invoke_agent.client"
    )
    assert refinement["ref"] == "gen_ai.invoke_agent.client"
    assert refinement["brief"] == (
        f"Semantic Conventions for [Microsoft Foundry Agent Service]({provider_doc}) client spans extend and "
        "override the semantic conventions for [Gen AI Spans](gen-ai-spans.md).\n"
    )
    assert refinement["note"] == (
        "This refinement applies when invoking a remotely hosted Foundry agent through the "
        "agent-scoped OpenAI Responses API.\n\n"
        '`gen_ai.provider.name` MUST be set to `"azure.ai.foundry"` and SHOULD be provided '
        "**at span creation time**.\n"
    )

    assert {attribute["ref"] for attribute in refinement["attributes"]} == {
        "gen_ai.agent.name",
        "gen_ai.conversation.id",
        "gen_ai.request.model",
        "server.port",
    }
    attributes = {attribute["ref"]: attribute for attribute in refinement["attributes"]}

    assert _required_level(attributes["gen_ai.agent.name"]) == ("required", None)
    assert _required_level(attributes["gen_ai.conversation.id"]) == (
        "conditionally_required",
        "When the request references a Foundry conversation.",
    )
    assert _required_level(attributes["gen_ai.request.model"]) == (
        "recommended",
        "When the invoked agent has one configured model and it is readily available to instrumentation.",
    )
    assert _required_level(attributes["server.port"]) == (
        "conditionally_required",
        "When the endpoint port is not the default port 443.",
    )


if __name__ == "__main__":
    test_metric_specs_expose_recommended_agent_name()
    test_metric_specs_are_named_as_the_registry_names_them()
    test_committed_google_adk_metrics_round_trip()
    test_entity_specs_expose_required_id()
    test_entities_keep_their_registry_names()
    test_registry_span_names_map_onto_report_keys()
    test_events_keep_their_registry_names()
    test_span_types_absent_from_a_data_file_are_not_reported()
    test_span_specs_are_named_as_the_registry_names_them()
    test_foundry_invoke_agent_refinement_contract()
    print("ok")
