# Transfer refinement follow-up design

## Goal

Bring the agent-transfer refinement in PR #447 up to the level of detail used
by the skill refinements in PR #498 while preserving executable reference
coverage.

## Refinement behavior

The `gen_ai.execute_tool.transfer.internal` refinement will override the
execute-tool span-name guidance:

```text
execute_tool {gen_ai.tool.name} {gen_ai.transfer.target.name}
```

Instrumentation will use that form when the target name is readily available
and has low cardinality. Otherwise it will use the inherited
`execute_tool {gen_ai.tool.name}` form.

The refinement note will:

- list Google ADK, LangChain, and OpenAI Agents examples already demonstrated
  by the reference scenarios;
- state that a transfer attempt records its exposed mode and target even when
  the attempt fails;
- state that the refinement changes the contract of the existing execute-tool
  span and does not create another span;
- retain the prohibition against inferring transfers from names, hierarchy,
  timing, or application conventions.

The generic execute-tool note will state that specialized tool calls use one
applicable refinement and still produce one span.

## Sampling relevance

No transfer attribute will be marked `sampling_relevant`.

Google ADK and OpenAI Agents expose all three transfer fields before creating
the span. LangChain exposes the authoritative mode and target through the
returned `Command`, after span creation. The fields therefore fail the user's
condition that they be available across all demonstrated scenarios.

## Conformance compatibility

The current conformance runner resolves telemetry against the base span type
and does not select span refinements. With transfer attributes declared only on
the refinement, CI removes them from the Google ADK, LangChain, and OpenAI
Agents `data.json` coverage and fails the deterministic-data checks.

Restore the three `gen_ai.transfer.*` attribute references on
`gen_ai.execute_tool.internal` as compatibility declarations. The refinement
will remain the specialized semantic contract and will override its naming and
applicability guidance. This avoids losing reference proof while refinement
selection is unsupported by the runner.

## Documentation

The generated transfer-refinement section will contain the specialized name,
applicability examples, failure behavior, and one-span rule. The generic
execute-tool section will link to the specialized refinements. The
agent-to-agent overview and non-normative examples will link directly to the
tool-based transfer section.

## Validation

Run the model-structure and reference-report tests, regenerate all repository
outputs, run the three affected scenarios, and confirm their committed
`data.json` files remain deterministic. Push the resulting commits to PR #447
and verify all required checks.
