# Agent transfer span refinement design

## Goal

Model tool-based agent transfer as a specialization of the generic
`gen_ai.execute_tool.internal` span. Keep ordinary tool execution generic while
retaining the transfer semantics already proposed in PR #447.

## Model

Add this entry under `span_refinements` in `model/gen-ai/spans.yaml`:

```yaml
- id: gen_ai.execute_tool.transfer.internal
  ref: gen_ai.execute_tool.internal
```

The refinement inherits the base span kind, operation name, tool attributes,
agent attributes, error handling, requirement level, and span name.

Move all transfer-specific text and attribute references from
`gen_ai.execute_tool.internal` into the refinement:

- `gen_ai.transfer.mode`
- `gen_ai.transfer.target.type`
- `gen_ai.transfer.target.name`

The refinement applies only when a framework or protocol explicitly exposes a
tool call as a transfer. Instrumentation must not infer a transfer from the tool
name, span hierarchy, timing, or application-specific conventions.

The base execute-tool span remains the convention for ordinary tools. A single
tool call produces one span, resolved against either the base convention or the
transfer refinement.

## Span identity

Keep the inherited span name:

```text
execute_tool {gen_ai.tool.name}
```

`gen_ai.transfer.mode` is the runtime discriminator that shows the transfer
semantics in emitted telemetry. The refinement ID organizes the semantic model
and generated documentation; instrumentation does not emit the ID as an
attribute.

## Documentation and examples

Regenerate the execute-tool documentation from the model. Update the
hand-written agent-interaction examples only where they describe the transfer
contract as part of the base execute-tool span.

The examples must state that:

- tool-based transfers use the transfer refinement;
- remote agent invocation remains a `gen_ai.invoke_agent` client operation;
- dedicated in-process non-tool transfers remain represented by the observable
  source and target `gen_ai.invoke_agent` internal spans.

## Reference coverage

Keep the existing Google ADK, LangChain, and OpenAI Agents scenarios that
exercise tool-based transfers. Update report resolution so spans carrying
`gen_ai.transfer.mode` are evaluated against
`gen_ai.execute_tool.transfer.internal`, while ordinary tool spans continue to
resolve to `gen_ai.execute_tool.internal`.

Every transfer attribute must remain traceable to an explicit library argument,
return value, event, or library state. Existing scenarios must not infer
transfer semantics from names or hierarchy.

## Validation

Run the smallest checks covering the change:

1. Generate documentation and reference reports.
2. Run targeted reference tests for span resolution and report coverage.
3. Run the affected Google ADK, LangChain, and OpenAI Agents scenarios if their
   committed outputs need regeneration.
4. Confirm generated output contains no new missing-attribute, span-name, or
   unresolved-refinement findings.

## Delivery

Commit the model, generated documentation, reference outputs, tests, and
directly related examples to PR #447. Push the commit to the fork branch
`multi-agent-interaction-semantics`.
