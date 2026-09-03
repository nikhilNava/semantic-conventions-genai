# GenAI transfer attributes implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `gen_ai.agent.interaction.type` with `gen_ai.transfer.*` while keeping `gen_ai.agent.*` tied to the agent executing each operation.

**Architecture:** The GenAI registry defines transfer mode and target identity. Caller-owned `execute_tool` and remote `invoke_agent` CLIENT operations opt into those attributes. Reference scenarios map library-owned transfer APIs to the new values, and generated docs and reports follow the model.

**Tech stack:** Weaver v2 YAML model, Markdown generation, Python reference scenarios, OpenTelemetry Python API, uv, Ruff

## Global constraints

- `gen_ai.agent.*` identifies the agent executing the recorded operation.
- `gen_ai.transfer.mode` uses `return_to_caller` and `pass_control`.
- `gen_ai.transfer.target.type` initially uses `agent`, `human`, and `workflow`.
- Do not infer transfer attributes from tool names, span hierarchy, timing, closure state, or application naming conventions.
- Every model attribute must have a signal and credible reference coverage.
- Generated registry pages and Weaver tables must not be hand-edited.
- Add one logical line to `changelog.d/447.enhancement.md`.

---

### Task 1: Add failing transfer coverage assertions

**Files:**
- Modify: `reference/tests/test_metrics.py:70-105`

**Interfaces:**
- Consumes: committed scenario `data.json` files and registry-derived span specifications
- Produces: regression assertions for transfer coverage and agent ownership

- [ ] **Step 1: Replace interaction coverage tests with transfer assertions**

```python
_TRANSFER_ATTRIBUTES = {
    "gen_ai.transfer.mode",
    "gen_ai.transfer.target.name",
    "gen_ai.transfer.target.type",
}


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
        serialized = json.dumps(data)
        assert "gen_ai.agent.interaction.type" not in serialized, path
```

- [ ] **Step 2: Run the assertions and confirm the expected failure**

Run:

```powershell
Set-Location reference
uv run --frozen python tests/test_metrics.py
```

Expected: failure because committed scenario data still contains `gen_ai.agent.interaction.type` and does not contain `gen_ai.transfer.*`.

- [ ] **Step 3: Commit the failing regression tests**

```powershell
git add -- reference/tests/test_metrics.py
git commit -m "Test transfer attribute coverage" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 2: Replace the interaction model

**Files:**
- Modify: `model/gen-ai/registry.yaml:458-500`
- Modify: `model/gen-ai/spans.yaml:155-175,500-655`
- Modify: `model/gen-ai/metrics.yaml:235-260`
- Modify: `docs/gen-ai/gen-ai-agent-spans.md:30-100`
- Modify: `changelog.d/447.enhancement.md`

**Interfaces:**
- Consumes: attribute names and semantics from the design spec
- Produces: Weaver definitions used by docs, reports, and instrumentation code generation

- [ ] **Step 1: Define the transfer attributes in the registry**

Replace `gen_ai.agent.interaction.type` with:

```yaml
  - key: gen_ai.transfer.mode
    type:
      members:
        - id: return_to_caller
          value: "return_to_caller"
          brief: The initiator waits for the target to finish and then resumes execution.
          stability: development
        - id: pass_control
          value: "pass_control"
          brief: The target takes control of the remaining work.
          stability: development
    brief: Describes how control passes to the transfer target.
    note: |
      Instrumentations MUST NOT infer this value from span hierarchy, tool names,
      timing, or application-specific conventions.
    stability: development
  - key: gen_ai.transfer.target.name
    type: string
    brief: The human-readable name of the transfer target.
    examples: ["weather_agent", "human_support"]
    stability: development
  - key: gen_ai.transfer.target.id
    type: string
    brief: The unique and stable identifier of the transfer target.
    examples: ["agent-42", "support-tier-2"]
    stability: development
  - key: gen_ai.transfer.target.type
    type:
      members:
        - id: agent
          value: "agent"
          brief: A GenAI agent.
          stability: development
        - id: human
          value: "human"
          brief: A person or group of people.
          stability: development
        - id: workflow
          value: "workflow"
          brief: A workflow or workflow step.
          stability: development
    brief: The type of the transfer target.
    stability: development
```

- [ ] **Step 2: Make `execute_tool` preserve source-agent identity**

Update `attributes.gen_ai.execute_tool.common` so `gen_ai.agent.name` reads:

```yaml
      - ref: gen_ai.agent.name
        brief: The name of the agent executing the tool.
        requirement_level:
          conditionally_required: When applicable.
```

Add the four transfer attributes to `gen_ai.execute_tool.internal`. Require mode and target type when the library explicitly exposes a transfer. Make target name and ID conditional on availability.

- [ ] **Step 3: Update remote invocation rules**

On `gen_ai.invoke_agent.client`, retain the existing meaning of `gen_ai.agent.*` as the invoked agent. Replace the interaction attribute with the four transfer attributes when the remote invocation has explicit transfer semantics. State that target values can match the invoked agent values.

- [ ] **Step 4: Update tool duration dimensions**

Replace the interaction dimension in `gen_ai.execute_tool.duration` with the four transfer attributes. State that `gen_ai.agent.name` always identifies the agent executing the tool.

- [ ] **Step 5: Rewrite the hand-written interaction guidance**

In `docs/gen-ai/gen-ai-agent-spans.md`, change the diagrams and prose to show:

```text
execute_tool transfer
agent.name = source
transfer.mode = return_to_caller | pass_control
transfer.target.type = agent
transfer.target.name = target
```

Keep the existing process boundaries and dedicated in-process non-tool capture gap.

- [ ] **Step 6: Update the changelog fragment**

```markdown
Add `gen_ai.transfer.*` attributes for control transfers on tool executions and remote agent invocations, while keeping `gen_ai.agent.*` tied to the agent executing each operation.
```

- [ ] **Step 7: Run model policy checks**

Run:

```powershell
make check-policies
```

Expected: Weaver completes without policy violations.

- [ ] **Step 8: Commit the model**

```powershell
git add -- model/gen-ai/registry.yaml model/gen-ai/spans.yaml model/gen-ai/metrics.yaml docs/gen-ai/gen-ai-agent-spans.md changelog.d/447.enhancement.md
git commit -m "Define GenAI transfer attributes" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 3: Update direct reference mappings

**Files:**
- Modify: `reference/scenarios/openai-agents/scenario.py:350-420`
- Modify: `reference/scenarios/google-adk/scenario.py:440-535`
- Modify: `reference/scenarios/langchain/scenario.py:400-440`
- Regenerate: `reference/scenarios/openai-agents/data.json`
- Regenerate: `reference/scenarios/google-adk/data.json`
- Regenerate: `reference/scenarios/langchain/data.json`

**Interfaces:**
- Consumes: library-owned transfer APIs and model attribute definitions
- Produces: runnable evidence for `return_to_caller`, `pass_control`, and agent targets

- [ ] **Step 1: Update OpenAI Agents agent-as-tool telemetry**

For `Agent.as_tool()`, emit:

```python
tool_span.set_attribute("gen_ai.agent.name", caller.name)
tool_span.set_attribute("gen_ai.transfer.mode", "return_to_caller")
tool_span.set_attribute("gen_ai.transfer.target.name", specialist.name)
tool_span.set_attribute("gen_ai.transfer.target.type", "agent")
```

Use the same attributes on `gen_ai.execute_tool.duration`. Do not emit `gen_ai.transfer.target.id` because this API does not expose a stable target ID.

- [ ] **Step 2: Update Google ADK AgentTool telemetry**

For `AgentTool(agent=specialist)`, emit:

```python
tool_span.set_attribute("gen_ai.agent.name", root_agent.name)
tool_span.set_attribute("gen_ai.transfer.mode", "return_to_caller")
tool_span.set_attribute("gen_ai.transfer.target.name", specialist.name)
tool_span.set_attribute("gen_ai.transfer.target.type", "agent")
```

The wrapper runs only after `root_agent` has been constructed, so read
`root_agent.name` from the closure. Use the same source and target values on
the duration metric.

- [ ] **Step 3: Update LangGraph control-pass telemetry**

For `Command(goto=target_name, graph=Command.PARENT)`, emit:

```python
tool_span.set_attribute("gen_ai.agent.name", source_name)
tool_span.set_attribute("gen_ai.transfer.mode", "pass_control")
tool_span.set_attribute("gen_ai.transfer.target.name", target_name)
tool_span.set_attribute("gen_ai.transfer.target.type", "agent")
```

Keep the target's `invoke_agent` INTERNAL span free of transfer attributes.

- [ ] **Step 4: Run each changed scenario**

Run:

```powershell
Set-Location reference
uv run run-scenario openai-agents
uv run run-scenario google-adk
uv run run-scenario langchain
```

Expected: each command exits successfully and rewrites its `data.json`. If package downloads fail with the known TLS error, do not retry repeatedly. Preserve the last valid data and update it only from verified emitted attributes.

- [ ] **Step 5: Run the regression assertions**

Run:

```powershell
Set-Location reference
uv run --frozen python tests/test_metrics.py
```

Expected: prints `ok`.

- [ ] **Step 6: Commit the reference mappings**

```powershell
git add -- reference/scenarios/openai-agents/scenario.py reference/scenarios/openai-agents/data.json reference/scenarios/google-adk/scenario.py reference/scenarios/google-adk/data.json reference/scenarios/langchain/scenario.py reference/scenarios/langchain/data.json reference/tests/test_metrics.py
git commit -m "Demonstrate GenAI transfer attributes" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 4: Regenerate docs and reports

**Files:**
- Regenerate: `docs/gen-ai/gen-ai-spans.md`
- Regenerate: `docs/gen-ai/gen-ai-metrics.md`
- Regenerate: `docs/gen-ai/gen-ai-agent-spans.md`
- Regenerate: `docs/registry/attributes/gen-ai.md`
- Regenerate: `reference/README.md`
- Regenerate: `reference/reports/execute-tool-span.md`
- Regenerate: `reference/reports/invoke-agent-client-span.md`

**Interfaces:**
- Consumes: completed model and scenario data
- Produces: committed generated documentation and coverage reports

- [ ] **Step 1: Run repository generation**

Run:

```powershell
make generate-all
```

Expected: registry docs, embedded Weaver tables, JSON schemas, and reference reports regenerate. If the JSON-schema uv command hits the known TLS error, run the unaffected targets separately:

```powershell
make update-upstream-links generate-registry generate-docs generate-reference-reports
```

- [ ] **Step 2: Confirm the removed attribute is gone**

Run:

```powershell
rg "gen_ai\.agent\.interaction\.type" model docs reference changelog.d
```

Expected: no matches.

- [ ] **Step 3: Confirm all new attributes appear in model and generated docs**

Run:

```powershell
rg "gen_ai\.transfer\.(mode|target\.(name|id|type))" model docs reference
```

Expected: matches in the registry, affected signals, generated docs, scenarios, data, tests, and reports.

- [ ] **Step 4: Commit generated output**

```powershell
git add -- docs/gen-ai/gen-ai-spans.md docs/gen-ai/gen-ai-metrics.md docs/gen-ai/gen-ai-agent-spans.md docs/registry/attributes/gen-ai.md reference/README.md reference/reports/execute-tool-span.md reference/reports/invoke-agent-client-span.md
git commit -m "Regenerate transfer convention docs" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 5: Validate and publish the branch

**Files:**
- Verify all changed files

**Interfaces:**
- Consumes: completed implementation
- Produces: a clean, pushed PR branch

- [ ] **Step 1: Run final model and reference checks**

Run:

```powershell
make check-policies
Set-Location reference
uv run --frozen python tests/test_metrics.py
uv run --frozen ruff check .
uv run --frozen ruff format --check .
Get-ChildItem -Recurse -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }
Set-Location ..
git diff --check
```

Expected: Weaver succeeds, tests print `ok`, Ruff succeeds, compilation has no output, and `git diff --check` has no output.

- [ ] **Step 2: Check repository consistency**

Run:

```powershell
git status --short
git diff --name-only --diff-filter=U
rg "^(<<<<<<<|=======|>>>>>>>)" .
```

Expected: no unmerged paths or conflict markers. Only intended files may be modified.

- [ ] **Step 3: Request code review**

Use the `requesting-code-review` skill. Fix high-confidence correctness issues, then rerun the checks from Step 1.

- [ ] **Step 4: Push the PR branch**

Run:

```powershell
git push fork HEAD:multi-agent-interaction-semantics
gh pr view 447 --repo open-telemetry/semantic-conventions-genai --json headRefOid,mergeable,mergeStateStatus
```

Expected: a normal non-force push succeeds and PR #447 points to the new head.
