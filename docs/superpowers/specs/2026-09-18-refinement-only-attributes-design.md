# Refinement-only GenAI span attributes

## Goal

Keep attributes introduced by span refinements out of their base span
contracts and generated base-span documentation, while retaining reference
coverage for the refinements.

## Model changes

Remove `gen_ai.caller.type` and `gen_ai.caller.name` from
`gen_ai.invoke_agent.client`. Keep them on
`gen_ai.invoke_agent.caller.client` with their current requirement levels and
sampling relevance.

Remove `gen_ai.transfer.mode`, `gen_ai.transfer.target.type`, and
`gen_ai.transfer.target.name` from `gen_ai.execute_tool.internal`. Keep them on
`gen_ai.execute_tool.transfer.internal` with their current requirement levels.

The base span notes continue to direct instrumentation authors to the
applicable refinement. Applying a refinement changes the contract of the same
physical span and does not create another span.

## Documentation changes

Remove this sentence from the non-normative agent interaction example:

> `gen_ai.transfer.*` is not recorded because remote invocation does not by
> itself imply a transfer of control.

Regenerate the Weaver-managed documentation. The generated base invoke-agent
table in `docs/gen-ai/gen-ai-agent-spans.md` must not list `gen_ai.caller.*`.
The caller refinement section must continue to list those attributes.

The generated base execute-tool table in `docs/gen-ai/gen-ai-spans.md` must not
list `gen_ai.transfer.*`. The transfer refinement section in
`docs/gen-ai/gen-ai-agent-spans.md` must continue to list those attributes.

## Reference coverage

Extend the reference reporting model with span refinement specifications. Each
spec identifies its refinement ID, base span type, label, attributes, and
required discriminator:

- `gen_ai.invoke_agent.caller.client` uses `gen_ai.caller.type`.
- `gen_ai.execute_tool.transfer.internal` uses `gen_ai.transfer.mode`.

The conformance run still records one physical base span. Before the runner
reduces its raw Weaver reports to `data.json`, a repository-owned reducer
classifies that span as an applicable refinement when the required
discriminator is present. It evaluates the refinement attributes from the same
observed span.

The reducer keeps the existing `spans` object for base-span coverage and writes
a separate `span_refinements` object keyed by refinement ID. Report loading
normalizes both objects against their corresponding specifications. This avoids
duplicating telemetry while preserving refinement attributes that the existing
base-only reducer would otherwise discard.

Add refinement report pages and index entries. Caller coverage must identify
Google ADK. Transfer coverage must identify Google ADK, LangChain, and OpenAI
Agents where their scenarios emit the applicable discriminator.

## Validation

Tests must verify:

- refinement attributes are absent from the base span definitions;
- refinement attributes remain declared on the refinements;
- generated base-span tables exclude refinement-only attributes;
- generated refinement sections include their attributes;
- the non-normative sentence is absent;
- refinement reports use the attributes from the same physical base span;
- committed scenario data and reports preserve current caller and transfer
  coverage.

Run the focused reference tests, regenerate all model-derived documentation and
reports, and confirm the generated tree is clean after a second generation.
