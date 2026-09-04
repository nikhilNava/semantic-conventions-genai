# a2a

The A2A Python SDK invokes remote agents over the Agent2Agent protocol. The
client owns the remote `invoke_agent` CLIENT operation; the remote agent owns
its execution and any model or tool operations performed while handling it.

| Operation | Should be instrumented here | Status |
| --- | --- | --- |
| invoke_agent (client) | Yes - `Client.send_message` invokes the remote agent | ✅ Implemented |
| invoke_agent (internal) | No - belongs to the remote agent implementation | ➖ Not instrumented here |
| inference (`chat`) | No - belongs to the remote agent implementation | ➖ Not instrumentable |
| execute_tool | No - belongs to the remote agent implementation | ➖ Not instrumentable |
