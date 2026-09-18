# Refinement-only GenAI span attributes implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep caller and transfer attributes exclusive to their span refinements while preserving generated documentation and reference coverage for those refinements.

**Architecture:** The semantic model declares refinement-only attributes only on `span_refinements`. Reference scenario runs continue to emit one physical base span, then a repository-owned postprocessor reads the persisted raw Weaver reports and adds a separate `span_refinements` coverage object to each scenario's `data.json`. Report generation treats base spans and refinements as separate contracts backed by the same observed span.

**Tech Stack:** Weaver registry model and Markdown generation, Python 3.12, pytest, repository reference-report tooling.

## Global constraints

- Do not hand-edit generated tables in `docs/gen-ai/gen-ai-agent-spans.md` or `docs/gen-ai/gen-ai-spans.md`.
- `gen_ai.caller.*` must exist only on `gen_ai.invoke_agent.caller.client`, not `gen_ai.invoke_agent.client`.
- `gen_ai.transfer.*` must exist only on `gen_ai.execute_tool.transfer.internal`, not `gen_ai.execute_tool.internal`.
- A refinement changes the contract of the same physical span and must not create duplicate telemetry.
- Reference values must remain traceable to library API input, output, exception, or library state.
- The generated base-span reports and new refinement reports must remain deterministic and alphabetically ordered.

---

### Task 1: Isolate refinement attributes in the semantic model

**Files:**
- Modify: `reference/tests/test_metrics.py:60-135`
- Modify: `reference/tests/test_metrics.py:180-210`
- Modify: `model/gen-ai/spans.yaml:516-565`
- Modify: `model/gen-ai/spans.yaml:896-989`
- Modify: `docs/gen-ai/non-normative/examples-agent-interactions.md:54-68`

**Interfaces:**
- Consumes: Existing `span_refinements` support in Weaver.
- Produces: Base span definitions without caller or transfer attributes; refinement definitions remain unchanged.

- [ ] **Step 1: Change the model-placement tests so they require refinement-only attributes**

Update `test_execute_tool_transfer_is_a_span_refinement`:

```python
    for attribute in (
        "gen_ai.transfer.mode",
        _TRANSFER_TARGET_TYPE,
        "gen_ai.transfer.target.name",
    ):
        assert f"- ref: {attribute}" not in execute_tool
        assert f"- ref: {attribute}" in transfer
        assert "sampling_relevant" not in _attribute_block(transfer, attribute)
```

Update `test_invoke_agent_caller_is_a_span_refinement`:

```python
    for attribute in _CALLER_ATTRIBUTES:
        assert f"- ref: {attribute}" not in invoke_agent_client
        assert f"- ref: {attribute}" in caller
        assert "sampling_relevant: true" in _attribute_block(caller, attribute)
```

Update `test_caller_refinement_is_documented_with_workflow_example`:

```python
    assert "gen_ai.transfer.*` is not recorded because remote invocation" not in interaction_examples
```

Remove the existing assertion that requires the deleted sentence.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
cd reference
uv run pytest tests/test_metrics.py \
  -k "execute_tool_transfer_is_a_span_refinement or invoke_agent_caller_is_a_span_refinement or caller_refinement_is_documented_with_workflow_example" -q
```

Expected: three failures because both base spans still declare the refinement attributes and the non-normative sentence still exists.

- [ ] **Step 3: Remove the refinement attributes from the base span definitions**

Delete these entries from `gen_ai.invoke_agent.client` in `model/gen-ai/spans.yaml`:

```yaml
      - ref: gen_ai.caller.type
        requirement_level:
          conditionally_required: When the immediate logical caller is explicitly exposed by the instrumented library or protocol.
        sampling_relevant: true
      - ref: gen_ai.caller.name
        requirement_level:
          conditionally_required: When `gen_ai.caller.type` is present and the caller name is available.
        sampling_relevant: true
```

Delete these entries from `gen_ai.execute_tool.internal`:

```yaml
      - ref: gen_ai.transfer.mode
        requirement_level:
          conditionally_required: When the instrumented framework or protocol explicitly exposes the tool call as a transfer.
      - ref: gen_ai.transfer.target.type
        requirement_level:
          conditionally_required: When `gen_ai.transfer.mode` is present and the target type is known.
      - ref: gen_ai.transfer.target.name
        requirement_level:
          conditionally_required: When `gen_ai.transfer.mode` is present and the target name is available.
```

Do not change the corresponding entries under `span_refinements`.

- [ ] **Step 4: Remove the requested non-normative sentence**

Delete from `docs/gen-ai/non-normative/examples-agent-interactions.md`:

```markdown
`gen_ai.transfer.*` is not recorded because remote invocation does not by
itself imply a transfer of control.
```

Keep the surrounding caller-refinement explanation and Google ADK example.

- [ ] **Step 5: Run the focused tests and verify they pass**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 6: Commit the model isolation**

```bash
git add model/gen-ai/spans.yaml \
  docs/gen-ai/non-normative/examples-agent-interactions.md \
  reference/tests/test_metrics.py
git commit -m "fix(gen-ai): isolate span refinement attributes" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Record refinement coverage from raw spans

**Files:**
- Modify: `reference/src/semconv_genai/attribute_spec.py`
- Modify: `reference/src/semconv_genai/semconv_model.py`
- Create: `reference/src/semconv_genai/refinement_coverage.py`
- Modify: `reference/src/semconv_genai/run_scenario.py:95-115`
- Modify: `reference/tests/test_metrics.py`

**Interfaces:**
- Consumes: Raw Weaver report files at `<scenario>/output/weaver-reports/*.json`.
- Produces:
  - `SpanRefinementSpec`
  - `span_refinement_specs() -> dict[str, SpanRefinementSpec]`
  - `collect_span_refinement_coverage(report_dir: Path) -> dict[str, list[str]]`
  - `update_span_refinement_coverage(scenario_dir: Path) -> None`
  - `data.json["span_refinements"]`

- [ ] **Step 1: Write tests for refinement specifications and raw-report extraction**

Add these imports to `reference/tests/test_metrics.py`:

```python
from semconv_genai.refinement_coverage import (
    collect_span_refinement_coverage,
    update_span_refinement_coverage,
)
from semconv_genai.semconv_model import (
    metric_specs,
    span_refinement_specs,
    span_specs,
)
```

Replace the existing combined `semconv_model` import.

Add a helper:

```python
def _raw_span(kind: str, attributes: dict[str, object]) -> dict[str, object]:
    return {
        "span": {
            "kind": kind,
            "attributes": [
                {"name": name, "value": value}
                for name, value in attributes.items()
            ],
        }
    }
```

Add the tests:

```python
def test_span_refinement_specs_describe_the_same_physical_base_spans():
    caller = span_refinement_specs()["invoke_agent_caller_client"]
    transfer = span_refinement_specs()["execute_tool_transfer"]

    assert caller.registry_id == "gen_ai.invoke_agent.caller.client"
    assert caller.base_registry_id == "gen_ai.invoke_agent.client"
    assert caller.operation_name == "invoke_agent"
    assert caller.span_kind == "client"
    assert caller.discriminator == "gen_ai.caller.type"

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
                    "gen_ai.transfer.target.type": "agent",
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
            "gen_ai.transfer.target.type",
        ],
        "gen_ai.invoke_agent.caller.client": [
            "gen_ai.caller.name",
            "gen_ai.caller.type",
        ],
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
    assert updated["span_refinements"] == {
        "gen_ai.invoke_agent.caller.client": [
            "gen_ai.caller.name",
            "gen_ai.caller.type",
        ]
    }
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
cd reference
uv run pytest tests/test_metrics.py \
  -k "span_refinement_specs or collect_span_refinement_coverage or update_span_refinement_coverage" -q
```

Expected: collection errors because the new types and module do not exist.

- [ ] **Step 3: Add the refinement specification type**

Add to `reference/src/semconv_genai/attribute_spec.py`:

```python
@dataclass(frozen=True)
class SpanRefinementSpec(AttributeSpec):
    base_registry_id: str = ""
    operation_name: str = ""
    span_kind: str = ""
    discriminator: str = ""
```

- [ ] **Step 4: Define the two supported refinement contracts**

Import `SpanRefinementSpec` in `reference/src/semconv_genai/semconv_model.py` and add:

```python
@cache
def span_refinement_specs() -> dict[str, SpanRefinementSpec]:
    return {
        "invoke_agent_caller_client": SpanRefinementSpec(
            label="Invoke Agent Caller",
            required=("gen_ai.caller.type",),
            conditionally_required=("gen_ai.caller.name",),
            recommended=(),
            opt_in=(),
            registry_id="gen_ai.invoke_agent.caller.client",
            base_registry_id="gen_ai.invoke_agent.client",
            operation_name="invoke_agent",
            span_kind="client",
            discriminator="gen_ai.caller.type",
        ),
        "execute_tool_transfer": SpanRefinementSpec(
            label="Execute Tool Transfer",
            required=(),
            conditionally_required=(
                "gen_ai.transfer.mode",
                "gen_ai.transfer.target.name",
                "gen_ai.transfer.target.type",
            ),
            recommended=(),
            opt_in=(),
            registry_id="gen_ai.execute_tool.transfer.internal",
            base_registry_id="gen_ai.execute_tool.internal",
            operation_name="execute_tool",
            span_kind="internal",
            discriminator="gen_ai.transfer.mode",
        ),
    }
```

The explicit mapping follows the existing `_SPANS` report mapping. The model-placement tests from Task 1 remain the drift guard against `model/gen-ai/spans.yaml`.

- [ ] **Step 5: Implement raw Weaver report extraction**

Create `reference/src/semconv_genai/refinement_coverage.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from semconv_genai.data_files import attr_names
from semconv_genai.semconv_model import span_refinement_specs


def _observed_spans(report_dir: Path):
    for path in sorted(report_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for sample in report.get("samples", []):
            span = sample.get("span")
            if not isinstance(span, dict):
                continue
            attributes = {
                attribute["name"]: attribute.get("value")
                for attribute in span.get("attributes", [])
                if isinstance(attribute, dict) and isinstance(attribute.get("name"), str)
            }
            yield span.get("kind"), attributes


def collect_span_refinement_coverage(report_dir: Path) -> dict[str, list[str]]:
    if not report_dir.is_dir():
        raise RuntimeError(f"Missing Weaver report directory: {report_dir}")

    collected: dict[str, set[str]] = {}
    for span_kind, attributes in _observed_spans(report_dir):
        for spec in span_refinement_specs().values():
            if span_kind != spec.span_kind:
                continue
            if attributes.get("gen_ai.operation.name") != spec.operation_name:
                continue
            if attributes.get(spec.discriminator) is None:
                continue
            present = set(attr_names(spec)) & attributes.keys()
            collected.setdefault(spec.registry_id, set()).update(present)

    return {
        registry_id: sorted(attributes)
        for registry_id, attributes in sorted(collected.items())
    }


def update_span_refinement_coverage(scenario_dir: Path) -> None:
    data_file = scenario_dir / "data.json"
    if not data_file.is_file():
        raise RuntimeError(f"Missing conformance data file: {data_file}")

    data = json.loads(data_file.read_text(encoding="utf-8"))
    data["span_refinements"] = collect_span_refinement_coverage(
        scenario_dir / "output" / "weaver-reports"
    )
    data_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
```

Python preserves the runner's existing top-level key order. Assigning
`span_refinements` appends it after `entities`, while
`collect_span_refinement_coverage` sorts refinement IDs and attribute lists.

- [ ] **Step 6: Wire refinement extraction after successful scenario runs**

Import the updater in `reference/src/semconv_genai/run_scenario.py`:

```python
from semconv_genai.refinement_coverage import update_span_refinement_coverage
```

After `conformance.run(...)` returns:

```python
            if exit_code == 0:
                update_span_refinement_coverage(reference_project_dir(library))
```

Do not update `data.json` after a failed scenario.

- [ ] **Step 7: Run the focused tests and static checks**

Run:

```bash
cd reference
uv run pytest tests/test_metrics.py \
  -k "span_refinement_specs or collect_span_refinement_coverage or update_span_refinement_coverage" -q
uv run ruff check src tests
uv run mypy
```

Expected: all commands pass.

- [ ] **Step 8: Commit refinement coverage extraction**

```bash
git add reference/src/semconv_genai/attribute_spec.py \
  reference/src/semconv_genai/semconv_model.py \
  reference/src/semconv_genai/refinement_coverage.py \
  reference/src/semconv_genai/run_scenario.py \
  reference/tests/test_metrics.py
git commit -m "feat(reference): record span refinement coverage" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Generate separate refinement reports

**Files:**
- Modify: `reference/src/semconv_genai/data_files.py`
- Modify: `reference/src/semconv_genai/report.py`
- Modify: `reference/tests/test_metrics.py`
- Generate: `reference/README.md`
- Create: `reference/reports/invoke-agent-caller-client-span-refinement.md`
- Create: `reference/reports/execute-tool-transfer-span-refinement.md`

**Interfaces:**
- Consumes: `data.json["span_refinements"]` and `span_refinement_specs()`.
- Produces: Normalized `ScenarioDataEntry.span_refinements`, README refinement index, and two refinement report pages.

- [ ] **Step 1: Write normalization and report tests**

Add to `reference/tests/test_metrics.py`:

```python
def test_refinement_data_normalizes_against_refinement_specs():
    entry = _normalize_scenario_data_entry(
        {
            "span_refinements": {
                "gen_ai.invoke_agent.caller.client": [
                    "gen_ai.caller.name",
                    "gen_ai.caller.type",
                ]
            }
        },
        "google-adk",
    )

    assert entry.span_refinements["invoke_agent_caller_client"] == {
        "gen_ai.caller.name": "present",
        "gen_ai.caller.type": "present",
    }


def test_generated_refinement_reports_keep_base_and_refinement_coverage_separate(tmp_path):
    entry = _normalize_scenario_data_entry(
        {
            "spans": {
                "gen_ai.invoke_agent.client": [
                    "gen_ai.operation.name",
                    "gen_ai.provider.name",
                ]
            },
            "span_refinements": {
                "gen_ai.invoke_agent.caller.client": [
                    "gen_ai.caller.name",
                    "gen_ai.caller.type",
                ]
            },
        },
        "google-adk",
    )

    page = _render_signal_section(
        [entry],
        "invoke_agent_caller_client",
        span_refinement_specs()["invoke_agent_caller_client"],
        tmp_path,
        "Span Refinement",
        lambda item: item.span_refinements,
    )

    rendered = "\n".join(page)
    assert "# Invoke Agent Caller Span Refinement" in rendered
    assert "| gen_ai.caller.type | [google-adk] |" in rendered
    assert "gen_ai.operation.name" not in rendered
```

Import `_render_signal_section` from `semconv_genai.report` for this focused unit test.

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
cd reference
uv run pytest tests/test_metrics.py \
  -k "refinement_data_normalizes or generated_refinement_reports" -q
```

Expected: failures because `ScenarioDataEntry` has no refinement field and report generation does not read refinement specs.

- [ ] **Step 3: Extend scenario-data normalization**

In `reference/src/semconv_genai/data_files.py`:

```python
SPAN_REFINEMENT_ORDER = [
    "invoke_agent_caller_client",
    "execute_tool_transfer",
]
```

Add the field:

```python
@dataclass(frozen=True)
class ScenarioDataEntry:
    library: str
    spans: dict[str, dict[str, str]]
    span_refinements: dict[str, dict[str, str]]
    events: dict[str, dict[str, str]]
    metrics: dict[str, dict[str, str]]
```

Import `span_refinement_specs`, then normalize:

```python
        span_refinements=_normalize_attr_data(
            entry.get("span_refinements"),
            span_refinement_specs(),
        ),
```

`SpanRefinementSpec` inherits `AttributeSpec`, so `_normalize_attr_data` does not need a separate code path.

- [ ] **Step 4: Add refinement report generation**

In `reference/src/semconv_genai/report.py`, import `SPAN_REFINEMENT_ORDER` and `span_refinement_specs`.

Add:

```python
def _span_refinements_of(entry: ScenarioDataEntry) -> dict[str, dict[str, str]]:
    return entry.span_refinements
```

Add document links:

```python
    "invoke_agent_caller_client": "../../docs/gen-ai/gen-ai-agent-spans.md#caller-aware-remote-invocation",
    "execute_tool_transfer": "../../docs/gen-ai/gen-ai-agent-spans.md#tool-based-transfer",
```

In `generate_index_markdown`, insert after the spans table:

```python
    lines.extend(
        [
            "",
            "### Span refinements",
            "",
            "| Span refinement | Libraries |",
            "| --- | --- |",
        ]
    )
    for refinement_type in SPAN_REFINEMENT_ORDER:
        spec = span_refinement_specs()[refinement_type]
        filename = _report_filename(refinement_type, "span-refinement")
        supporting = _get_supporting_entries(
            entries,
            refinement_type,
            spec,
            _span_refinements_of,
        )
        lines.append(
            f"| [{spec.label}](reports/{filename}) | "
            f"{_library_dir_links(supporting)} |"
        )
```

In `write_report_pages`, add:

```python
    for refinement_type in SPAN_REFINEMENT_ORDER:
        spec = span_refinement_specs()[refinement_type]
        page_path = reports_dir / _report_filename(
            refinement_type,
            "span-refinement",
        )
        page_lines = _render_signal_section(
            entries,
            refinement_type,
            spec,
            reports_dir,
            "Span Refinement",
            _span_refinements_of,
        )
        page_path.write_text(
            _generate_detail_page(page_lines),
            encoding="utf-8",
        )
```

- [ ] **Step 5: Run the focused tests and static checks**

Run:

```bash
cd reference
uv run pytest tests/test_metrics.py \
  -k "refinement_data_normalizes or generated_refinement_reports" -q
uv run ruff check src tests
uv run mypy
```

Expected: all commands pass.

- [ ] **Step 6: Commit report support**

```bash
git add reference/src/semconv_genai/data_files.py \
  reference/src/semconv_genai/report.py \
  reference/tests/test_metrics.py
git commit -m "feat(reference): report span refinement coverage" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Regenerate scenarios, docs, and reports

**Files:**
- Generate: `docs/gen-ai/gen-ai-agent-spans.md`
- Generate: `docs/gen-ai/gen-ai-spans.md`
- Generate: `docs/registry/attributes/gen-ai.md`
- Generate: `reference/scenarios/google-adk/data.json`
- Generate: `reference/scenarios/langchain/data.json`
- Generate: `reference/scenarios/openai-agents/data.json`
- Generate: `reference/README.md`
- Generate: `reference/reports/invoke-agent-client-span.md`
- Generate: `reference/reports/execute-tool-span.md`
- Generate: `reference/reports/invoke-agent-caller-client-span-refinement.md`
- Generate: `reference/reports/execute-tool-transfer-span-refinement.md`
- Modify: `reference/tests/test_metrics.py`

**Interfaces:**
- Consumes: Completed model, extraction, and report changes from Tasks 1-3.
- Produces: Committed generated artifacts consistent with the refinement-only model and scenario telemetry.

- [ ] **Step 1: Add generated-output assertions**

Add helpers to `reference/tests/test_metrics.py`:

```python
def _generated_section(document: str, selector: str) -> str:
    start = document.index(f"<!-- weaver {selector} -->")
    end = document.index("<!-- endweaver -->", start)
    return document[start:end]
```

Add:

```python
def test_generated_base_span_docs_exclude_refinement_only_attributes():
    repository_root = Path(__file__).parents[2]
    agent_spans = (
        repository_root / "docs" / "gen-ai" / "gen-ai-agent-spans.md"
    ).read_text(encoding="utf-8")
    gen_ai_spans = (
        repository_root / "docs" / "gen-ai" / "gen-ai-spans.md"
    ).read_text(encoding="utf-8")

    invoke_agent_base = _generated_section(
        agent_spans,
        '.registry.spans[] | select(.type == "gen_ai.invoke_agent.client")',
    )
    caller_refinement = _generated_section(
        agent_spans,
        '.refinements.spans[] | select(.id == "gen_ai.invoke_agent.caller.client")',
    )
    execute_tool_base = _generated_section(
        gen_ai_spans,
        '.registry.spans[] | select(.type == "gen_ai.execute_tool.internal")',
    )
    transfer_refinement = _generated_section(
        agent_spans,
        '.refinements.spans[] | select(.id == "gen_ai.execute_tool.transfer.internal")',
    )

    assert "gen_ai.caller.type" not in invoke_agent_base
    assert "gen_ai.caller.name" not in invoke_agent_base
    assert "gen_ai.caller.type" in caller_refinement
    assert "gen_ai.caller.name" in caller_refinement

    assert "gen_ai.transfer.mode" not in execute_tool_base
    assert "gen_ai.transfer.target.name" not in execute_tool_base
    assert "gen_ai.transfer.target.type" not in execute_tool_base
    assert "gen_ai.transfer.mode" in transfer_refinement
    assert "gen_ai.transfer.target.name" in transfer_refinement
    assert "gen_ai.transfer.target.type" in transfer_refinement
```

Add report assertions:

```python
def test_committed_refinement_reports_preserve_reference_coverage():
    reports = Path(__file__).parents[1] / "reports"
    caller = (
        reports / "invoke-agent-caller-client-span-refinement.md"
    ).read_text(encoding="utf-8")
    transfer = (
        reports / "execute-tool-transfer-span-refinement.md"
    ).read_text(encoding="utf-8")

    assert "| gen_ai.caller.type | [google-adk] |" in caller
    assert "| gen_ai.caller.name | [google-adk] |" in caller
    assert "| gen_ai.transfer.mode | [google-adk], [langchain], [openai-agents] |" in transfer
    assert "| gen_ai.transfer.target.name | [google-adk], [langchain], [openai-agents] |" in transfer
    assert "| gen_ai.transfer.target.type | [google-adk], [openai-agents] |" in transfer
```

- [ ] **Step 2: Run the generated-output tests and verify they fail**

Run:

```bash
cd reference
uv run pytest tests/test_metrics.py \
  -k "generated_base_span_docs or committed_refinement_reports" -q
```

Expected: failures because generated docs and report pages have not been refreshed.

- [ ] **Step 3: Regenerate the three affected scenarios**

Run:

```bash
cd reference
uv run run-scenario google-adk
uv run run-scenario langchain
uv run run-scenario openai-agents
```

Expected: each command succeeds and writes a sorted `span_refinements` object to its `data.json`.

Check:

```bash
git diff -- reference/scenarios/google-adk/data.json \
  reference/scenarios/langchain/data.json \
  reference/scenarios/openai-agents/data.json
```

Expected:

- Google ADK records caller and transfer refinements.
- LangChain records the transfer refinement without `gen_ai.transfer.target.type`.
- OpenAI Agents records the transfer refinement with all three transfer attributes.

- [ ] **Step 4: Regenerate all repository-owned artifacts**

From the repository root:

```bash
make generate-all
```

Expected:

- base invoke-agent docs no longer list `gen_ai.caller.*`;
- base execute-tool docs no longer list `gen_ai.transfer.*`;
- refinement sections still list those attributes;
- base reference reports no longer list refinement-only attributes;
- two refinement report pages and README index entries are generated.

- [ ] **Step 5: Run the full focused validation**

Run:

```bash
cd reference
uv run pytest tests/test_metrics.py -q
uv run ruff check src tests
uv run mypy
cd ..
make generate-all
git diff --check
git status --short
```

Expected:

- all tests and static checks pass;
- the second `make generate-all` produces no additional changes;
- `git diff --check` reports no whitespace errors;
- `git status --short` lists only the intended implementation and generated files.

- [ ] **Step 6: Commit generated artifacts and final tests**

```bash
git add model/gen-ai/spans.yaml \
  docs/gen-ai/gen-ai-agent-spans.md \
  docs/gen-ai/gen-ai-spans.md \
  docs/gen-ai/non-normative/examples-agent-interactions.md \
  docs/registry/attributes/gen-ai.md \
  reference/README.md \
  reference/reports \
  reference/scenarios/google-adk/data.json \
  reference/scenarios/langchain/data.json \
  reference/scenarios/openai-agents/data.json \
  reference/tests/test_metrics.py
git commit -m "docs(gen-ai): separate span refinement attributes" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
