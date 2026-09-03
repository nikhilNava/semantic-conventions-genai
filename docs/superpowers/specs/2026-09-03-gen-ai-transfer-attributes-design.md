# GenAI transfer attributes

## Goal

Represent control transfers without changing the meaning of `gen_ai.agent.*`.
On every span and metric, `gen_ai.agent.name` and `gen_ai.agent.id` identify
the agent executing the recorded operation. Transfer attributes identify the
recipient and whether the initiator resumes after the recipient finishes.

## Approaches considered

### Replace the interaction attribute

Replace `gen_ai.agent.interaction.type` with `gen_ai.transfer.*` on
caller-owned operations. This keeps agent identity consistent and models
agent, human, and workflow targets. This is the selected approach.

### Change only tool operations

Use transfer attributes on `execute_tool`, but retain the existing target
meaning of `gen_ai.agent.*` on `invoke_agent` CLIENT spans. This minimizes the
diff but gives the same attributes two meanings, so consumers still need
operation-specific interpretation.

### Emit both models

Add transfer attributes while retaining `gen_ai.agent.interaction.type`.
This offers a transition period, but these conventions are still in
development. Duplicate attributes would create conflicting sources of truth
without a compatibility requirement.

## Attribute model

Define these development attributes in the GenAI registry:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `gen_ai.transfer.mode` | enum | `return_to_caller` when the initiator waits for a result and resumes; `pass_control` when the target owns the remaining work |
| `gen_ai.transfer.target.name` | string | Human-readable target name |
| `gen_ai.transfer.target.id` | string | Stable target identifier |
| `gen_ai.transfer.target.type` | enum | Target category, initially `agent`, `human`, or `workflow` |

Instrumentation records transfer attributes only when the framework,
protocol, or application API exposes the transfer semantics and target.
Instrumentation must not infer them from tool names, span hierarchy, timing,
or application naming conventions.

`gen_ai.transfer.target.name` and `gen_ai.transfer.target.id` are independent.
Instrumentation records whichever values the API provides. Target type is
required when transfer mode is present because it tells consumers how to
interpret the target identity.

## Signal ownership

An `execute_tool` INTERNAL span records a transfer when the tool call invokes
another agent or passes control to another target:

* `gen_ai.agent.*` identifies the source agent executing the tool.
* `gen_ai.transfer.*` describes the transfer and target.
* `gen_ai.tool.*` continues to describe the tool.

An `invoke_agent` CLIENT span uses the same transfer attributes when a remote
agent invocation has known transfer semantics. Its `gen_ai.agent.*`
attributes retain their existing meaning and identify the invoked agent.
`gen_ai.transfer.target.*` describes that agent's role as the transfer target.
The target values may match `gen_ai.agent.*` on this operation.

The target agent's `invoke_agent` INTERNAL span identifies the target through
its own `gen_ai.agent.*` attributes. It does not repeat transfer attributes.

`gen_ai.execute_tool.duration` copies the attributes from the corresponding
tool span. This preserves source-agent attribution for tool metrics.

The current change does not add transfer attributes to workflow or graph-node
spans. The registry definitions support non-agent targets, but each additional
signal must define its own recording rules and reference coverage in a later
change.

## Reference coverage

Update the existing direct mappings:

| Scenario | Mode | Target type |
| --- | --- | --- |
| OpenAI Agents `Agent.as_tool()` | `return_to_caller` | `agent` |
| Google ADK `AgentTool` | `return_to_caller` | `agent` |
| LangGraph `Command(goto=..., graph=Command.PARENT)` | `pass_control` | `agent` |

Each `execute_tool` span and duration metric keeps the source agent in
`gen_ai.agent.name` and records the target in
`gen_ai.transfer.target.name`. Target IDs remain absent when the framework
does not expose a stable ID.

OpenAI Agents native handoff remains a documented capture gap because the
current convention has no caller-owned span for its dedicated in-process,
non-tool transfer.

## Migration and validation

Remove `gen_ai.agent.interaction.type` from the model, generated docs,
metrics, tests, reports, scenarios, and changelog fragment. Regenerate all
derived files.

Validation covers Weaver policies and generation, reference coverage
assertions, Ruff checks, Python compilation, and Git whitespace and conflict
checks. Scenario runs should confirm emitted telemetry when dependencies can
be downloaded.
