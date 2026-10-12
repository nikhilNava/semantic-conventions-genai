# google-adk

The Google Agent Development Kit (ADK) is an agent framework. It calls models
through its model layer (e.g. `google-genai`), so it **delegates inference**. It
owns the agent, workflow, tool, memory, and remote-agent operations it runs
directly.

The `RemoteA2aAgent` scenario exercises two library-owned invocation paths:

- a `FunctionNode` calls public `Context.run_node` with a `RemoteA2aAgent`;
- a `routing_agent` invokes a `RemoteA2aAgent` through `AgentTool`. The
  execute-tool transfer span reads the source agent from
  `ToolContext.agent_name` and the target from public `AgentTool.agent`.

Both paths reach an injected public A2A `ClientFactory`, which wraps public
`Client.send_message` to create the invoke-agent CLIENT span. Two workflow
steps and the AgentTool call invoke the same remote-agent instance
concurrently, and the scenario verifies that each CLIENT span is a child of
its local `RemoteA2aAgent` execution. ADK marks its A2A integration as
experimental.

| Operation | Should be instrumented here | Status |
| --- | --- | --- |
| inference (`chat`) | No — delegates to the model client (`google-genai`) | ✅ Correctly not emitted |
| invoke_agent (internal) | Yes — agent run | ✅ Implemented |
| invoke_agent (client) | Yes — `RemoteA2aAgent` invokes a remote A2A agent | ✅ Implemented |
| invoke_workflow | Yes — graph-based `Workflow` nodes | ✅ Implemented |
| execute_tool | Yes — ADK runs the tool | ✅ Implemented |
| memory | Yes — memory service upsert / search | ✅ Implemented |
| skills | Yes — `SkillToolset` runs the skill tools | ✅ Implemented |
