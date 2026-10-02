# azure-ai-inference

The OpenAI client configured with a Microsoft Foundry Models endpoint is a
**model-call boundary**. It calls the deployed model directly, so it owns
inference and embeddings. The directory retains its historical name because
the semantic convention provider value remains `azure.ai.inference`. Tool
calling is supported, but the client returns tool calls without executing
them.

| Operation | Should be instrumented here | Status |
| --- | --- | --- |
| inference (`chat`) | Yes — calls the model directly | ✅ Implemented |
| embeddings | Yes — calls the model directly | ✅ Implemented |
| execute_tool | No — the client doesn't execute tools; the tool runs in app code | ➖ Not instrumentable |
