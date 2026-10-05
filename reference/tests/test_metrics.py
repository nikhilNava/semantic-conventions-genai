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

import semconv_genai.run_scenario as run_scenario
from semconv_genai.data_files import (
    _normalize_scenario_data_entry,
    load_scenario_data_files,
)
from semconv_genai.refinement_coverage import (
    collect_span_refinement_coverage,
    update_span_refinement_coverage,
)
from semconv_genai.report import _render_signal_section
from semconv_genai.semconv_model import (
    entity_specs,
    metric_specs,
    span_refinement_specs,
    span_specs,
)

_TOOL_CALLS = "gen_ai.invoke_agent.tool_calls"
_INFERENCE_CALLS = "gen_ai.invoke_agent.inference_calls"
_TRANSFER_ATTRIBUTES = {
    "gen_ai.transfer.mode",
    "gen_ai.transfer.target.name",
}
_CALLER_ATTRIBUTES = {
    "gen_ai.caller.type",
    "gen_ai.caller.name",
}


def _attribute_block(model_block: str, attribute: str) -> str:
    return model_block.split(f"- ref: {attribute}", 1)[1].split("\n      - ref:", 1)[0]


def _generated_section(document: str, selector: str) -> str:
    start = document.index(f"<!-- weaver {selector} -->")
    end = document.index("<!-- endweaver -->", start)
    return document[start:end]


def _raw_span(kind: str, attributes: dict[str, object]) -> dict[str, object]:
    return {
        "span": {
            "kind": kind,
            "attributes": [{"name": name, "value": value} for name, value in attributes.items()],
        }
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


def test_execute_tool_duration_does_not_include_transfer_attributes():
    metrics_model = (Path(__file__).parents[2] / "model" / "gen-ai" / "metrics.yaml").read_text(encoding="utf-8")
    execute_tool_duration = metrics_model.split("- name: gen_ai.execute_tool.duration", 1)[1].split(
        "metric_refinements:", 1
    )[0]
    assert "gen_ai.transfer." not in execute_tool_duration


def test_execute_tool_transfer_is_a_span_refinement():
    spans_model = (Path(__file__).parents[2] / "model" / "gen-ai" / "spans.yaml").read_text(encoding="utf-8")
    spans, refinements = spans_model.split("span_refinements:", 1)
    execute_tool = spans.split("- type: gen_ai.execute_tool.internal", 1)[1].split(
        "- type: gen_ai.invoke_workflow.internal", 1
    )[0]

    transfer = refinements.split("- id: gen_ai.execute_tool.transfer.internal", 1)[1]

    assert "instrumented library API or state" in execute_tool
    assert "[refinement](gen-ai-agent-spans.md#execute-tool-span)" in execute_tool
    assert "SHOULD NOT record two" in execute_tool
    assert "different spans for one call" in execute_tool

    assert "ref: gen_ai.execute_tool.internal" in transfer
    assert "execute_tool {gen_ai.tool.name} {gen_ai.transfer.target.name}" in transfer
    assert "including when the transfer attempt fails" in transfer
    assert "MUST NOT be" in transfer
    assert "used for transfers that are not tool executions" in transfer
    assert "does not produce an additional span" in transfer

    for attribute in (
        "gen_ai.transfer.mode",
        "gen_ai.transfer.target.name",
    ):
        assert f"- ref: {attribute}" not in execute_tool
        assert f"- ref: {attribute}" in transfer
        assert "sampling_relevant" not in _attribute_block(transfer, attribute)


def test_invoke_agent_client_owns_caller_attributes_without_refinement():
    repository_root = Path(__file__).parents[2]
    spans_model = (repository_root / "model" / "gen-ai" / "spans.yaml").read_text(encoding="utf-8")
    spans, refinements = spans_model.split("span_refinements:", 1)
    invoke_agent_client = spans.split("- type: gen_ai.invoke_agent.client", 1)[1].split(
        "- type: gen_ai.invoke_agent.internal", 1
    )[0]
    google_adk = (repository_root / "reference" / "scenarios" / "google-adk" / "scenario.py").read_text(
        encoding="utf-8"
    )

    assert "- ref: gen_ai.caller.type" in invoke_agent_client
    assert "- ref: gen_ai.caller.name" in invoke_agent_client
    assert "immediate logical caller" in invoke_agent_client
    assert "gen_ai.invoke_agent.caller.client" not in refinements
    caller_type = _attribute_block(invoke_agent_client, "gen_ai.caller.type")
    caller_name = _attribute_block(invoke_agent_client, "gen_ai.caller.name")
    assert "explicitly exposed as an agent or workflow" in caller_type
    assert "including when the invocation fails" in caller_type
    assert "sampling_relevant: true" in caller_type
    assert "When `gen_ai.caller.type` is recorded and the caller name is available" in caller_name
    assert "including when the invocation fails" in caller_name
    assert "sampling_relevant: true" in caller_name
    assert "gen_ai.caller." not in invoke_agent_client.split("brief:", 1)[0]
    assert '"gen_ai.caller.type": call_context["caller_type"]' in google_adk
    assert '"gen_ai.caller.name": call_context["caller_name"]' in google_adk


def test_transfer_examples_use_target_qualified_names_when_available():
    repository_root = Path(__file__).parents[2]
    scenarios_dir = Path(__file__).parents[1] / "scenarios"

    google_adk = (scenarios_dir / "google-adk" / "scenario.py").read_text(encoding="utf-8")
    openai_agents = (scenarios_dir / "openai-agents" / "scenario.py").read_text(encoding="utf-8")
    langchain = (scenarios_dir / "langchain" / "scenario.py").read_text(encoding="utf-8")
    base_spans = (repository_root / "docs" / "gen-ai" / "gen-ai-spans.md").read_text(encoding="utf-8")
    agent_spans = (repository_root / "docs" / "gen-ai" / "gen-ai-agent-spans.md").read_text(encoding="utf-8")
    interaction_examples = (
        repository_root / "docs" / "gen-ai" / "non-normative" / "examples-agent-interactions.md"
    ).read_text(encoding="utf-8")

    assert 'f"execute_tool {agent_tool.name} {specialist.name}"' in google_adk
    assert 'f"execute_tool {weather_tool.name} {specialist.name}"' in openai_agents
    assert '"execute_tool transfer_to_weather_agent"' in langchain
    assert "gen_ai.transfer.target.type" not in langchain
    assert "[agents exposed as tools](gen-ai-agent-spans.md#agent-as-a-tool)" in base_spans
    assert "### Agent as a tool" in agent_spans
    assert "`execute_tool transfer_to_weather_agent weather_agent`" in interaction_examples
    assert "`execute_tool transfer_to_weather_agent`" in interaction_examples
    assert "[agent-as-a-tool refinement](../gen-ai-agent-spans.md#agent-as-a-tool)" in interaction_examples


def test_caller_attributes_are_documented_with_workflow_example():
    repository_root = Path(__file__).parents[2]
    agent_spans = (repository_root / "docs" / "gen-ai" / "gen-ai-agent-spans.md").read_text(encoding="utf-8")
    interaction_examples = (
        repository_root / "docs" / "gen-ai" / "non-normative" / "examples-agent-interactions.md"
    ).read_text(encoding="utf-8")

    invoke_agent_client = _generated_section(
        agent_spans,
        '.registry.spans[] | select(.type == "gen_ai.invoke_agent.client")',
    )
    assert "gen_ai.caller.type" in invoke_agent_client
    assert "gen_ai.caller.name" in invoke_agent_client
    assert "Caller-aware remote invocation" not in agent_spans
    assert "`gen_ai.caller.type`" in interaction_examples
    assert "`gen_ai.caller.name`" in interaction_examples
    assert "gen_ai.transfer.*` is not recorded because remote invocation" not in interaction_examples
    assert "weather_workflow" in interaction_examples
    assert "caller attributes on that span" in interaction_examples


def test_agent_interaction_refinements_are_grouped_by_base_span():
    repository_root = Path(__file__).parents[2]
    agent_spans = (repository_root / "docs" / "gen-ai" / "gen-ai-agent-spans.md").read_text(encoding="utf-8")

    invoke_agent = agent_spans.index("## Invoke agent client span")
    execute_tool = agent_spans.index("## Execute tool span")
    transfer = agent_spans.index("### Agent as a tool")
    skills = agent_spans.index("### Agent skills")

    assert invoke_agent < execute_tool < transfer < skills
    assert "Caller-aware remote invocation" not in agent_spans
    assert "    - [Agent as a tool](#agent-as-a-tool)" in agent_spans
    assert "    - [Agent skills](#agent-skills)" in agent_spans


def test_generated_base_span_docs_include_caller_but_exclude_transfer_attributes():
    repository_root = Path(__file__).parents[2]
    agent_spans = (repository_root / "docs" / "gen-ai" / "gen-ai-agent-spans.md").read_text(encoding="utf-8")
    gen_ai_spans = (repository_root / "docs" / "gen-ai" / "gen-ai-spans.md").read_text(encoding="utf-8")

    invoke_agent_base = _generated_section(
        agent_spans,
        '.registry.spans[] | select(.type == "gen_ai.invoke_agent.client")',
    )
    execute_tool_base = _generated_section(
        gen_ai_spans,
        '.registry.spans[] | select(.type == "gen_ai.execute_tool.internal")',
    )
    transfer_refinement = _generated_section(
        agent_spans,
        '.refinements.spans[] | select(.id == "gen_ai.execute_tool.transfer.internal")',
    )

    assert "gen_ai.caller.type" in invoke_agent_base
    assert "gen_ai.caller.name" in invoke_agent_base

    assert "gen_ai.transfer.mode" not in execute_tool_base
    assert "gen_ai.transfer.target.name" not in execute_tool_base
    assert "gen_ai.transfer.target.type" not in execute_tool_base
    assert "gen_ai.transfer.mode" in transfer_refinement
    assert "gen_ai.transfer.target.name" in transfer_refinement
    assert "gen_ai.transfer.target.type" not in transfer_refinement


def test_transfer_refinement_names_agent_target_without_redundant_type():
    repository_root = Path(__file__).parents[2]
    registry = (repository_root / "model" / "gen-ai" / "registry.yaml").read_text(encoding="utf-8")
    spans = (repository_root / "model" / "gen-ai" / "spans.yaml").read_text(encoding="utf-8")
    examples = (repository_root / "docs" / "gen-ai" / "non-normative" / "examples-agent-interactions.md").read_text(
        encoding="utf-8"
    )

    assert "gen_ai.transfer.target.type" not in registry
    assert "gen_ai.transfer.target.type" not in spans
    assert "`gen_ai.transfer.target.name` identifies the target agent." in examples
    assert "gen_ai.transfer.target.type" not in examples
    assert "transfer.target.type" not in examples


def test_committed_refinement_reports_preserve_reference_coverage():
    reports = Path(__file__).parents[1] / "reports"
    invoke_agent = (reports / "invoke-agent-client-span.md").read_text(encoding="utf-8")
    transfer = (reports / "execute-tool-transfer-span-refinement.md").read_text(encoding="utf-8")
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

    assert "| gen_ai.caller.type | [google-adk] |" in invoke_agent
    assert "| gen_ai.caller.name | [google-adk] |" in invoke_agent
    assert "| gen_ai.transfer.mode | [google-adk], [langchain], [openai-agents] |" in transfer
    assert "| gen_ai.transfer.target.name | [google-adk], [langchain], [openai-agents] |" in transfer
    assert "gen_ai.transfer.target.type" not in transfer
    assert "[Execute Tool Transfer](reports/execute-tool-transfer-span-refinement.md)" in readme
    assert "invoke-agent-caller-client-span-refinement.md" not in readme
    assert not (reports / "invoke-agent-caller-client-span-refinement.md").exists()


def test_committed_metrics_do_not_include_transfer_attributes():
    for path in (Path(__file__).parents[1] / "scenarios").glob("*/data.json"):
        metrics = json.loads(path.read_text(encoding="utf-8")).get("metrics", {})
        assert "gen_ai.transfer." not in json.dumps(metrics), path


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


def test_refinement_data_normalizes_against_transfer_refinement_spec():
    entry = _normalize_scenario_data_entry(
        {
            "span_refinements": {
                "gen_ai.execute_tool.transfer.internal": [
                    "gen_ai.transfer.mode",
                    "gen_ai.transfer.target.name",
                ]
            }
        },
        "openai-agents",
    )

    assert entry.span_refinements["execute_tool_transfer"] == {
        "gen_ai.transfer.mode": "present",
        "gen_ai.transfer.target.name": "present",
    }


def test_generated_transfer_refinement_report_keeps_base_coverage_separate(tmp_path):
    entry = _normalize_scenario_data_entry(
        {
            "spans": {
                "gen_ai.execute_tool.internal": [
                    "gen_ai.operation.name",
                    "gen_ai.tool.name",
                ]
            },
            "span_refinements": {
                "gen_ai.execute_tool.transfer.internal": [
                    "gen_ai.transfer.mode",
                    "gen_ai.transfer.target.name",
                ]
            },
        },
        "openai-agents",
    )

    page = _render_signal_section(
        [entry],
        "execute_tool_transfer",
        span_refinement_specs()["execute_tool_transfer"],
        tmp_path,
        "Span Refinement",
        lambda item: item.span_refinements,
    )

    rendered = "\n".join(page)
    assert "# Execute Tool Transfer Span Refinement" in rendered
    assert "| gen_ai.transfer.mode | [openai-agents] |" in rendered
    assert "gen_ai.operation.name" not in rendered


def test_span_specs_are_named_as_the_registry_names_them():
    for key, spec in span_specs().items():
        assert spec.registry_id.startswith("gen_ai."), key


def test_span_refinement_specs_describe_the_same_physical_base_spans():
    transfer = span_refinement_specs()["execute_tool_transfer"]

    assert set(span_refinement_specs()) == {"execute_tool_transfer"}
    assert transfer.registry_id == "gen_ai.execute_tool.transfer.internal"
    assert transfer.base_registry_id == "gen_ai.execute_tool.internal"
    assert transfer.operation_name == "execute_tool"
    assert transfer.span_kind == "internal"
    assert transfer.discriminator == "gen_ai.transfer.mode"


def test_collect_span_refinement_coverage_uses_discriminators(tmp_path):
    report_dir = tmp_path / "weaver-reports"
    report_dir.mkdir()
    report = {
        "samples": [
            _raw_span(
                "client",
                {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.caller.type": "workflow",
                    "gen_ai.caller.name": "weather_workflow",
                },
            ),
            _raw_span(
                "internal",
                {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.transfer.mode": "return_to_caller",
                    "gen_ai.transfer.target.name": "weather_agent",
                },
            ),
            _raw_span(
                "client",
                {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": "no_explicit_caller",
                },
            ),
        ]
    }
    (report_dir / "reference.json").write_text(json.dumps(report), encoding="utf-8")

    assert collect_span_refinement_coverage(report_dir) == {
        "gen_ai.execute_tool.transfer.internal": [
            "gen_ai.transfer.mode",
            "gen_ai.transfer.target.name",
        ],
    }


def test_collect_span_refinement_coverage_excludes_type_mismatches(tmp_path):
    report_dir = tmp_path / "weaver-reports-mismatch"
    report_dir.mkdir()
    sample = _raw_span(
        "internal",
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.transfer.mode": "return_to_caller",
            "gen_ai.transfer.target.name": "weather_agent",
        },
    )
    target_name = next(
        attribute for attribute in sample["span"]["attributes"] if attribute["name"] == "gen_ai.transfer.target.name"
    )
    target_name["live_check_result"] = {"all_advice": [{"id": "type_mismatch"}]}
    (report_dir / "reference.json").write_text(
        json.dumps({"samples": [sample]}),
        encoding="utf-8",
    )

    assert collect_span_refinement_coverage(report_dir) == {
        "gen_ai.execute_tool.transfer.internal": ["gen_ai.transfer.mode"]
    }


def test_update_span_refinement_coverage_preserves_runner_data(tmp_path):
    scenario_dir = tmp_path / "scenario"
    report_dir = scenario_dir / "output" / "weaver-reports"
    report_dir.mkdir(parents=True)
    original = {
        "spans": {"gen_ai.invoke_agent.client": ["gen_ai.operation.name"]},
        "events": {},
        "metrics": {},
        "findings": [],
        "entities": {},
    }
    (scenario_dir / "data.json").write_text(
        json.dumps(original, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "samples": [
            _raw_span(
                "client",
                {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.caller.type": "workflow",
                    "gen_ai.caller.name": "weather_workflow",
                },
            )
        ]
    }
    (report_dir / "reference.json").write_text(json.dumps(report), encoding="utf-8")

    update_span_refinement_coverage(scenario_dir)

    updated = json.loads((scenario_dir / "data.json").read_text(encoding="utf-8"))
    assert updated["spans"] == original["spans"]
    assert updated["span_refinements"] == {}


def test_runner_output_paths_follow_conformance_flags(tmp_path):
    scenario_dir = tmp_path / "scenario"

    report_dir, data_file = run_scenario._runner_output_paths(
        scenario_dir,
        [
            "--report-dir",
            "custom-reports",
            "--data-file=custom-data.json",
        ],
    )

    repository_root = Path(__file__).parents[2]
    assert report_dir == repository_root / "custom-reports"
    assert data_file == repository_root / "custom-data.json"


def test_partial_runner_output_does_not_update_refinement_coverage(tmp_path):
    scenario_dir = tmp_path / "partial-scenario"
    report_dir = tmp_path / "partial-reports"
    data_file = tmp_path / "partial-data.json"
    report_dir.mkdir()
    data_file.write_text(
        json.dumps({"spans": {}, "events": {}, "metrics": {}}),
        encoding="utf-8",
    )
    before = run_scenario._file_state(data_file)

    assert not run_scenario._update_refinement_coverage_after_run(
        scenario_dir,
        report_dir,
        data_file,
        before,
    )
    assert "span_refinements" not in json.loads(data_file.read_text(encoding="utf-8"))


def test_complete_runner_output_updates_selected_paths(tmp_path):
    scenario_dir = tmp_path / "complete-scenario"
    report_dir = tmp_path / "complete-reports"
    data_file = tmp_path / "complete-data.json"
    report_dir.mkdir()
    data_file.write_text(
        json.dumps({"spans": {}, "events": {}, "metrics": {}}),
        encoding="utf-8",
    )
    before = run_scenario._file_state(data_file)
    data_file.write_text(
        json.dumps(
            {
                "spans": {"gen_ai.invoke_agent.client": ["gen_ai.operation.name"]},
                "events": {},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "reference.json").write_text(
        json.dumps(
            {
                "samples": [
                    _raw_span(
                        "client",
                        {
                            "gen_ai.operation.name": "invoke_agent",
                            "gen_ai.caller.type": "workflow",
                            "gen_ai.caller.name": "weather_workflow",
                        },
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    assert run_scenario._update_refinement_coverage_after_run(
        scenario_dir,
        report_dir,
        data_file,
        before,
    )
    updated = json.loads(data_file.read_text(encoding="utf-8"))
    assert updated["span_refinements"] == {}


def test_invoke_agent_client_does_not_duplicate_transfer_target():
    spec = span_specs()["invoke_agent_client"]
    attributes = spec.required + spec.conditionally_required + spec.recommended + spec.opt_in
    assert not any(attribute.startswith("gen_ai.transfer.") for attribute in attributes)


def test_committed_google_adk_remote_agent_covers_internal_and_client_spans():
    path = Path(__file__).parents[1] / "scenarios" / "google-adk" / "data.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    invoke_agent_internal = data["spans"]["gen_ai.invoke_agent.internal"]
    invoke_agent_client = data["spans"]["gen_ai.invoke_agent.client"]
    assert not any(attribute.startswith("gen_ai.transfer.") for attribute in invoke_agent_internal)
    assert "gen_ai.agent.name" in invoke_agent_client
    assert "gen_ai.provider.name" in invoke_agent_client
    assert set(invoke_agent_client) >= _CALLER_ATTRIBUTES
    assert "gen_ai.invoke_agent.caller.client" not in data["span_refinements"]
    assert not any(attribute.startswith("gen_ai.transfer.") for attribute in invoke_agent_client)


def test_google_adk_remote_agent_uses_public_per_invocation_caller_state():
    path = Path(__file__).parents[1] / "scenarios" / "google-adk" / "scenario.py"
    scenario = path.read_text(encoding="utf-8")

    assert "Workflow(" in scenario
    assert "AgentTool(agent=agent_remote_agent)" in scenario
    assert "class _AgentToolModel(BaseLlm)" in scenario
    assert 'routing_model = _AgentToolModel(model="reference-agent-tool-model")' in scenario
    assert "model=routing_model" in scenario
    assert 'name="weather_workflow"' in scenario
    assert "FunctionNode(" in scenario
    assert "await ctx.run_node(" in scenario
    assert "caller = self.node" in scenario
    assert "tool_context.agent_name" in scenario
    assert '"remote_agent": self.agent' in scenario
    assert "class _TracingClientFactory(ClientFactory)" in scenario
    assert "client.send_message" in scenario
    assert "_A2A_CALL_CONTEXT_KEY" in scenario
    assert "_otel_context.attach(" in scenario
    assert "_otel_context.detach(" in scenario
    assert "_otel_context.get_value(_A2A_CALL_CONTEXT_KEY)" in scenario
    assert 'contextvars.ContextVar("workflow_a2a_call_context"' in scenario
    assert "workflow_call_context.set(call_context)" in scenario
    assert "workflow_call_context.reset(token)" in scenario
    assert "request_barrier = asyncio.Barrier(3)" in scenario
    assert "await asyncio.wait_for(request_barrier.wait(), timeout=10)" in scenario
    assert "asyncio.gather(" in scenario
    assert "remote_agent.parent_agent" not in scenario
    assert '"gen_ai.caller.type": call_context["caller_type"]' in scenario
    assert '"gen_ai.caller.name": call_context["caller_name"]' in scenario
    assert 'attributes.get("gen_ai.caller.type")' in scenario
    assert 'attributes.get("gen_ai.caller.name")' in scenario
    assert '("workflow", "invoke_remote_weather_agent")' in scenario
    assert '("workflow", "invoke_remote_forecast_agent")' in scenario
    assert '("agent", "routing_agent")' in scenario
    assert "set(self.client_calls) != expected_callers" in scenario
    assert 'attribute.startswith("gen_ai.caller.")' not in scenario
    assert 'f"invoke_workflow {workflow.name}"' in scenario
    assert "ctx.parent_ctx.node" not in scenario
    assert "tool_context._invocation_context" not in scenario
    assert "._custom_metadata" not in scenario
    assert "._telemetry_context" not in scenario
    assert "remote_agent._agent_card" not in scenario
    assert "adk_a2a_compat" not in scenario
    assert '"_run_impl"' not in scenario
    assert '"_run_async_impl"' not in scenario


def test_execute_tool_transfer_report_links_to_agent_as_tool():
    path = Path(__file__).parents[1] / "src" / "semconv_genai" / "report.py"
    report_source = path.read_text(encoding="utf-8")

    assert '"execute_tool_transfer": "../../docs/gen-ai/gen-ai-agent-spans.md#agent-as-a-tool"' in report_source


def test_caller_and_transfer_guidance_is_positive_and_scoped():
    model_dir = Path(__file__).parents[2] / "model" / "gen-ai"
    registry = (model_dir / "registry.yaml").read_text(encoding="utf-8")
    spans = (model_dir / "spans.yaml").read_text(encoding="utf-8")

    caller_registry = registry.split("- key: gen_ai.caller.type", 1)[1].split("- key: gen_ai.transfer.mode", 1)[0]
    assert "readily available from the instrumented library API or state" in caller_registry
    assert "MUST NOT infer caller identity" not in caller_registry

    transfer_registry = registry.split("- key: gen_ai.transfer.mode", 1)[1].split("- key: gen_ai.main_agent.id", 1)[0]
    assert "The initiator does not wait for the target to finish." in transfer_registry
    assert "- id: human" not in transfer_registry
    assert "- id: workflow" not in transfer_registry

    transfer = spans.split("- id: gen_ai.execute_tool.transfer.internal", 1)[1].split(
        "- id: gen_ai.execute_tool.load_skill.internal", 1
    )[0]
    assert "instead" in transfer
    assert "of the generic execute-tool span contract" in transfer
    assert "including when the transfer attempt fails" in transfer
    assert "remote agent invocation rather than a tool execution" not in transfer
    assert "dedicated in-process non-tool transfer" not in transfer


def test_google_adk_remote_client_records_available_version_and_errors():
    path = Path(__file__).parents[1] / "scenarios" / "google-adk" / "scenario.py"
    scenario = path.read_text(encoding="utf-8")

    assert '"gen_ai.agent.version": card.version' in scenario
    assert 'client_span.set_attribute("error.type", type(error).__qualname__)' in scenario


def test_langchain_transfer_graph_runs_under_workflow_span():
    path = Path(__file__).parents[1] / "scenarios" / "langchain" / "scenario.py"
    scenario = path.read_text(encoding="utf-8")
    transfer = scenario.split("async def run_tool_handoff_reference()", 1)[1].split(
        "async def run_workflow_reference()", 1
    )[0]

    workflow_start = transfer.index('f"invoke_workflow {workflow_name}"')
    graph_run = transfer.index("await graph.ainvoke")
    assert workflow_start < graph_run
    assert '"gen_ai.workflow.name": workflow_name' in transfer


def test_committed_transfer_scenarios_emit_transfer_attributes():
    scenarios_dir = Path(__file__).parents[1] / "scenarios"

    expected_attributes = {
        "google-adk": _TRANSFER_ATTRIBUTES,
        "langchain": _TRANSFER_ATTRIBUTES,
        "openai-agents": _TRANSFER_ATTRIBUTES,
    }

    for library, expected in expected_attributes.items():
        data = json.loads((scenarios_dir / library / "data.json").read_text(encoding="utf-8"))
        execute_tool = data["spans"].get("gen_ai.execute_tool.internal", [])
        transfer_refinement = data["span_refinements"]["gen_ai.execute_tool.transfer.internal"]

        for attribute in expected:
            assert attribute in transfer_refinement, (library, attribute)
            assert attribute not in execute_tool, (library, attribute)
        assert "gen_ai.transfer.target.type" not in transfer_refinement, library

        invoke_agent_internal = data["spans"].get("gen_ai.invoke_agent.internal", [])
        assert not any(attribute.startswith("gen_ai.transfer.") for attribute in invoke_agent_internal), library


def test_interaction_type_is_removed_from_committed_scenarios():
    for path in (Path(__file__).parents[1] / "scenarios").glob("*/data.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "gen_ai.agent.interaction.type" not in json.dumps(data), path


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    test_metric_specs_expose_recommended_agent_name()
    test_metric_specs_are_named_as_the_registry_names_them()
    test_execute_tool_duration_does_not_include_transfer_attributes()
    test_execute_tool_transfer_is_a_span_refinement()
    test_invoke_agent_client_owns_caller_attributes_without_refinement()
    test_transfer_examples_use_target_qualified_names_when_available()
    test_caller_attributes_are_documented_with_workflow_example()
    test_agent_interaction_refinements_are_grouped_by_base_span()
    test_generated_base_span_docs_include_caller_but_exclude_transfer_attributes()
    test_committed_refinement_reports_preserve_reference_coverage()
    test_committed_metrics_do_not_include_transfer_attributes()
    test_committed_google_adk_metrics_round_trip()
    test_entity_specs_expose_required_id()
    test_entities_keep_their_registry_names()
    test_registry_span_names_map_onto_report_keys()
    test_events_keep_their_registry_names()
    test_span_types_absent_from_a_data_file_are_not_reported()
    test_refinement_data_normalizes_against_transfer_refinement_spec()
    with TemporaryDirectory() as tmpdir:
        test_generated_transfer_refinement_report_keeps_base_coverage_separate(Path(tmpdir))
    test_span_specs_are_named_as_the_registry_names_them()
    test_span_refinement_specs_describe_the_same_physical_base_spans()
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_collect_span_refinement_coverage_uses_discriminators(tmp_path)
        test_collect_span_refinement_coverage_excludes_type_mismatches(tmp_path)
        test_update_span_refinement_coverage_preserves_runner_data(tmp_path)
        test_runner_output_paths_follow_conformance_flags(tmp_path)
        test_partial_runner_output_does_not_update_refinement_coverage(tmp_path)
        test_complete_runner_output_updates_selected_paths(tmp_path)
    test_invoke_agent_client_does_not_duplicate_transfer_target()
    test_committed_google_adk_remote_agent_covers_internal_and_client_spans()
    test_google_adk_remote_agent_uses_public_per_invocation_caller_state()
    test_google_adk_remote_client_records_available_version_and_errors()
    test_langchain_transfer_graph_runs_under_workflow_span()
    test_committed_transfer_scenarios_emit_transfer_attributes()
    test_interaction_type_is_removed_from_committed_scenarios()
    print("ok")
