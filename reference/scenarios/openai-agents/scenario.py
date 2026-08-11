"""Reference implementation for OpenAI Agents.

Exercises: agent run with tool calling, agent-as-tool delegation, native handoff,
and a multi-agent run with handoffs wrapped in a workflow span, against a mock
OpenAI server, with manual OTel spans.
"""

import asyncio
import contextlib
import json
import os
import time

import openai
from agents import Agent, RunConfig, Runner, function_tool, handoff
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.tool import FunctionTool, ToolContext
from opentelemetry import context as _context
from opentelemetry import trace as _trace
from reference_shared import flush_and_shutdown, reference_meter, reference_tracer, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_reference_tracer = reference_tracer()
_reference_meter = reference_meter()

# `gen_ai.execute_tool.duration` is recorded once per tool execution this scenario
# instruments, next to that execution's `execute_tool` span. It carries
# `gen_ai.agent.interaction.type` when the tool call is itself a delegation to
# another agent, and `gen_ai.agent.name` then identifies the target agent; for an
# ordinary tool call there is no interaction type and `gen_ai.agent.name` is the
# agent executing the tool (see model/gen-ai/metrics.yaml). The native handoff is
# not a tool execution, so it records no point here.
_execute_tool_duration = _reference_meter.create_histogram(
    "gen_ai.execute_tool.duration",
    unit="s",
    description="The duration of a single tool execution.",
)


@contextlib.contextmanager
def _invoke_agent_execution_spans(*, request_model, input_text, handoff_parent_context):
    """Bound each *logical* agent execution inside one public ``Runner.run`` with a
    single ``invoke_agent <agent.name>`` span, then restore the patched seam on exit.

    A native handoff switches the active agent mid-run, so one ``Runner.run`` covers
    two logical agent executions. The SDK exposes no public per-execution boundary,
    so this patches its private per-turn function ``agents.run.run_single_turn`` and
    keeps the state needed to derive the boundary: the span stays open across every
    consecutive turn of the same agent and is replaced only when the active agent
    changes, so a multi-turn execution is not split into one span per turn. Patching
    a private seam is allowed; the scenario's entry point stays the public
    ``Runner.run`` and this seam is never called directly.

    Every execution of one ``Runner.run`` belongs to a single trace. ``handoff_parent_context``
    is the per-run holder the transfer seam fills with the context of the caller-owned
    ``invoke_agent <target>`` handoff span: consuming it parents the target agent's
    execution directly to the handoff operation that caused it. When the active
    agent changes for any other reason, the fallback is this run's own context,
    so the executions still share one trace instead of starting a new one.

    ``input_text`` is the user's message, which is the input to the first (source)
    execution only. Each executing agent's own response goes on its own span, and no
    execution span carries ``gen_ai.agent.interaction.type`` -- that is caller-owned
    and belongs on the operation that directed the work.
    """
    import agents.run

    original_run_single_turn = agents.run.run_single_turn
    # `run_context` starts as the context this `Runner.run` is entered with and is
    # upgraded to the first execution span below, so it is a stable, non-empty
    # parent for the rest of the run. It lives in this generator call, so one run
    # never inherits the context of another.
    execution = {"span": None, "agent_name": None, "started": False, "run_context": _context.get_current()}

    async def _traced_run_single_turn(**kwargs):
        executing_agent = kwargs["bindings"].public_agent
        if execution["agent_name"] != executing_agent.name:
            if execution["span"] is not None:
                execution["span"].end()
            agent_span_attributes = {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.request.model": request_model,
                "gen_ai.agent.name": executing_agent.name,
            }
            # A handoff caused this switch, so the target's execution is parented to
            # the caller-owned `invoke_agent <target>` handoff operation that handed
            # the work over -- consumed once, since each transfer causes one switch.
            parent_context = handoff_parent_context.pop("context", None)
            if parent_context is None:
                parent_context = execution["run_context"]
            agent_span = _reference_tracer.start_span(
                f"invoke_agent {executing_agent.name}",
                context=parent_context,
                attributes=agent_span_attributes,
            )
            if not execution["started"]:
                # The first execution roots this run's trace, so later executions
                # fall back to it and stay in the same trace.
                execution["run_context"] = _trace.set_span_in_context(agent_span, execution["run_context"])
                # The user's message is the input to the first (source) execution only.
                agent_span.set_attribute(
                    "gen_ai.input.messages",
                    json.dumps([{"role": "user", "parts": [{"type": "text", "content": input_text}]}]),
                )
            execution.update(span=agent_span, agent_name=executing_agent.name, started=True)

        agent_span = execution["span"]
        with _trace.use_span(agent_span, end_on_exit=False):
            turn_result = await original_run_single_turn(**kwargs)
            # Only a final-output step exposes `.output` (a handoff step exposes
            # `.new_agent`); the executing agent's own response goes on its own span,
            # never on the span of the agent that handed the work over.
            next_step = turn_result.next_step
            if hasattr(next_step, "output"):
                agent_span.set_attribute(
                    "gen_ai.output.messages",
                    json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": str(next_step.output)}]}]),
                )
            return turn_result

    agents.run.run_single_turn = _traced_run_single_turn
    try:
        yield
    finally:
        agents.run.run_single_turn = original_run_single_turn
        if execution["span"] is not None:
            execution["span"].end()


@contextlib.contextmanager
def _patched_method(obj, name, replacement):
    """Temporarily replace the ``obj.name`` bound method with ``replacement`` as
    an instrumentation seam, always restoring the original in ``finally`` --
    including on exceptions. Repo rules allow patching a public or private method
    as a seam as long as the scenario still enters through the library's public
    API; this helper just guarantees the patch is symmetric.
    """
    original = getattr(obj, name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        setattr(obj, name, original)


@function_tool
def get_weather(ctx: ToolContext[None], location: str) -> str:
    """Get the current weather for a location."""
    tool_span_attributes = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "get_weather",
        "gen_ai.tool.type": "function",
    }
    # The agent executing the tool, known before the call: the span attribute
    # and the duration metric below both report it. An ordinary tool call is
    # not an agent interaction, so neither carries an interaction type.
    executing_agent_name = ctx.agent.name if ctx.agent is not None else None
    duration_attributes = {"gen_ai.tool.name": "get_weather", "gen_ai.tool.type": "function"}
    if executing_agent_name:
        duration_attributes["gen_ai.agent.name"] = executing_agent_name
    start = time.perf_counter()
    try:
        with _reference_tracer.start_as_current_span(
            "execute_tool get_weather", attributes=tool_span_attributes
        ) as tool_span:
            tool_span.set_attribute("gen_ai.tool.description", get_weather.description)
            if executing_agent_name:
                tool_span.set_attribute("gen_ai.agent.name", executing_agent_name)
            tool_span.set_attribute("gen_ai.tool.call.id", ctx.tool_call_id)
            tool_span.set_attribute("gen_ai.tool.call.arguments", json.dumps({"location": location}))
            result = "Sunny, 72°F"
            tool_span.set_attribute("gen_ai.tool.call.result", result)
            return result
    finally:
        # Record in `finally` so a failed tool execution is still timed.
        _execute_tool_duration.record(time.perf_counter() - start, duration_attributes)


async def run_agent():
    """Run a simple agent with the OpenAI Agents SDK, with manual spans."""
    client = openai.AsyncOpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")
    request_model = "gpt-4o-mini"
    model = OpenAIChatCompletionsModel(model=request_model, openai_client=client)

    tools = [get_weather]
    captured_responses = []
    agent = Agent(
        name="test-agent",
        instructions="You are a helpful assistant.",
        model=model,
        tools=tools,
    )
    input_text = "What's the weather in Seattle?"

    print("  [agent_run] agent with tool calling (reference implementation)")
    agent_span_attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.request.model": request_model,
        "gen_ai.agent.name": agent.name,
    }
    with _reference_tracer.start_as_current_span(
        "invoke_agent test-agent", attributes=agent_span_attributes
    ) as agent_span:
        agent_span.set_attribute(
            "gen_ai.system_instructions", json.dumps([{"type": "text", "content": agent.instructions}])
        )
        agent_span.set_attribute(
            "gen_ai.input.messages", json.dumps([{"role": "user", "parts": [{"type": "text", "content": input_text}]}])
        )
        agent_span.set_attribute(
            "gen_ai.tool.definitions",
            json.dumps(
                [
                    {
                        "type": "function",
                        "function": {"name": t.name, "description": t.description, "parameters": t.params_json_schema},
                    }
                    for t in tools
                    if isinstance(t, FunctionTool)
                ]
            ),
        )
        original_create = client.chat.completions.create

        async def _capture_create(*args, **kwargs):
            response = await original_create(*args, **kwargs)
            captured_responses.append(response)
            return response

        client.chat.completions.create = _capture_create
        try:
            result = await Runner.run(agent, input_text)
        finally:
            client.chat.completions.create = original_create
        usage = result.context_wrapper.usage
        if usage.total_tokens:
            agent_span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
            agent_span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        if captured_responses:
            last_response = captured_responses[-1]
            finish_reasons = [
                choice.finish_reason
                for choice in getattr(last_response, "choices", []) or []
                if getattr(choice, "finish_reason", None)
            ]
            if finish_reasons:
                agent_span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
        if result.final_output:
            agent_span.set_attribute(
                "gen_ai.output.messages",
                json.dumps(
                    [
                        {
                            "role": "assistant",
                            "parts": [{"type": "text", "content": str(result.final_output)}],
                        }
                    ]
                ),
            )
        print(f"    -> {str(result.final_output)[:60]}")


async def run_agent_as_tool_delegation():
    """Delegation: a caller agent invokes another agent exposed via `Agent.as_tool`.

    `Agent.as_tool()` is the SDK's explicit agent-as-tool API: the caller invokes
    the target agent as a function tool and, per its docstring, "the conversation
    is continued by the original agent" -- the caller expects a result back. That
    is a `delegation` interaction, mapped directly from the API being invoked (not
    inferred from the arguments).

    Two agents really execute here, and each logical execution gets exactly one
    span:

    * `invoke_agent assistant` wraps the public `Runner.run(caller, ...)` call --
      one logical invocation of the caller, however many turns it takes -- and
      owns the user's input and the caller's own final response.
    * `invoke_agent weather-specialist` wraps the delegated invoker call, which is
      what actually runs the target agent (`Agent.as_tool` invokes it through its
      own nested `Runner.run`). It is nested under the caller-owned
      `execute_tool weather-specialist` span and carries the target's response.
      Being the target's own execution, it stays free of
      `gen_ai.agent.interaction.type` -- that is caller-owned and lives on the
      `execute_tool` span, which names the target. Its `gen_ai.input.messages` is
      an honest omission: the invoker seam exposes the delegated input only as the
      tool's raw argument JSON, which the caller-owned `execute_tool` span already
      records verbatim as `gen_ai.tool.call.arguments`; the target's message list
      is built inside the nested `Runner.run` and never surfaces here.

    The delegated agent's model call belongs to the `openai` library, so no
    inference span is emitted here.
    """
    client = openai.AsyncOpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")
    request_model = "gpt-4o-mini"
    model = OpenAIChatCompletionsModel(model=request_model, openai_client=client)

    # The target agent has no tools, so its mock model call returns plain text.
    specialist = Agent(
        name="weather-specialist",
        instructions="You report the weather.",
        model=model,
    )
    weather_tool = specialist.as_tool(
        tool_name="weather-specialist",
        tool_description="Delegate weather questions to the weather specialist agent.",
    )

    # Wrap the tool's public invoker to open the caller-owned execute_tool span
    # around the real sub-agent invocation. The entry point stays `Runner.run`;
    # the patched invoker is restored in `finally` by `_patched_method` below.
    original_on_invoke_tool = weather_tool.on_invoke_tool

    async def _traced_on_invoke_tool(tool_context, input_json):
        tool_span_attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": weather_tool.name,
            "gen_ai.tool.type": "function",
        }
        start = time.perf_counter()
        try:
            with _reference_tracer.start_as_current_span(
                f"execute_tool {weather_tool.name}", attributes=tool_span_attributes
            ) as tool_span:
                # `direct`: `as_tool()` is an agent-as-tool delegation API, so the type
                # is intrinsic to the call, and the target is the wrapped agent's name.
                tool_span.set_attribute("gen_ai.agent.interaction.type", "delegation")
                tool_span.set_attribute("gen_ai.agent.name", specialist.name)
                tool_span.set_attribute("gen_ai.tool.call.id", tool_context.tool_call_id)
                tool_span.set_attribute("gen_ai.tool.call.arguments", input_json)
                # This invoker call *is* the target agent's execution (`as_tool` runs it
                # through its own nested `Runner.run`), so the target's single execution
                # span wraps it. No interaction type: that is the caller's, above.
                target_span_attributes = {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.request.model": request_model,
                    "gen_ai.agent.name": specialist.name,
                }
                with _reference_tracer.start_as_current_span(
                    f"invoke_agent {specialist.name}", attributes=target_span_attributes
                ) as target_span:
                    result = await original_on_invoke_tool(tool_context, input_json)
                    # Success-only: set the response only after the call returns, so a
                    # failure never manufactures a fabricated output.
                    target_span.set_attribute(
                        "gen_ai.output.messages",
                        json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": str(result)}]}]),
                    )
                tool_span.set_attribute("gen_ai.tool.call.result", str(result))
            return result
        finally:
            # Record in `finally` so a failed delegation is still timed. Every
            # dimension is the operation's target identity, known before the call.
            _execute_tool_duration.record(
                time.perf_counter() - start,
                {
                    "gen_ai.tool.name": weather_tool.name,
                    "gen_ai.tool.type": "function",
                    "gen_ai.agent.interaction.type": "delegation",
                    "gen_ai.agent.name": specialist.name,
                },
            )

    caller = Agent(
        name="assistant",
        instructions="You are a helpful assistant. Delegate weather questions to the specialist.",
        model=model,
        tools=[weather_tool],
    )
    input_text = "What's the weather in Seattle?"

    print("  [delegation] agent-as-tool via Agent.as_tool (reference implementation)")
    # `_patched_method` installs the caller-owned execute_tool invoker and restores
    # it in `finally`. The caller's whole logical invocation is the public
    # `Runner.run(caller, ...)` call, so its execution span wraps that call directly
    # and owns the caller's input and response.
    caller_span_attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.request.model": request_model,
        "gen_ai.agent.name": caller.name,
    }
    with (
        _patched_method(weather_tool, "on_invoke_tool", _traced_on_invoke_tool),
        _reference_tracer.start_as_current_span(
            f"invoke_agent {caller.name}", attributes=caller_span_attributes
        ) as caller_span,
    ):
        caller_span.set_attribute(
            "gen_ai.system_instructions", json.dumps([{"type": "text", "content": caller.instructions}])
        )
        caller_span.set_attribute(
            "gen_ai.input.messages", json.dumps([{"role": "user", "parts": [{"type": "text", "content": input_text}]}])
        )
        result = await Runner.run(caller, input_text)
        caller_span.set_attribute(
            "gen_ai.output.messages",
            json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": str(result.final_output)}]}]),
        )
    print(f"    -> {str(result.final_output)[:60]}")


async def run_agent_handoff():
    """Handoff: a caller agent transfers control to another agent via `handoff`.

    `handoff()` is the SDK's explicit handoff API, and it is *not* an ordinary tool
    execution: the SDK gives it its own `Handoff` object and its own invoker
    (`on_invoke_handoff`), whose whole purpose is to switch the active agent so the
    target owns the remaining work. That dedicated boundary is a caller-owned
    agent invocation, so it is modelled as an `invoke_agent` operation naming the
    **target** agent, with `gen_ai.agent.interaction.type=handoff` mapped directly
    from the API being invoked. The generated `transfer_to_<agent>` name the model
    sees is a prompt-level artifact of that API, not a tool the agent executes, so
    this operation emits no `gen_ai.tool.*` attributes and no
    `gen_ai.execute_tool.duration` point.

    Two agents really execute under one public `Runner.run(triage_agent, ...)`
    call, and each logical execution gets exactly one span:

    * `invoke_agent triage-agent` bounds the source agent's whole execution
      (every turn it takes before the switch) and owns the user's input.
    * `invoke_agent billing-agent` (caller-owned) is its child: the transfer
      itself, carrying `gen_ai.agent.interaction.type=handoff` and the target from
      `Handoff.agent_name`. The seam exposes the target's identity before the call
      but not its configuration, so this operation claims nothing else.
    * `invoke_agent billing-agent` (the target's own execution) bounds the target
      agent's whole execution after the switch and carries that agent's response.
      Being the target's own execution, it must stay free of the interaction
      attribute -- the interaction type is caller-owned. It is parented to the
      caller-owned handoff span, so the whole run -- source execution, handoff,
      target execution -- is one causally connected trace.

    The SDK has no public per-execution boundary, so `_invoke_agent_execution_spans`
    derives it from the active agent changing across the private `run_single_turn`
    seam, while the scenario's entry point stays the public `Runner.run`.
    """
    client = openai.AsyncOpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")
    request_model = "gpt-4o-mini"
    model = OpenAIChatCompletionsModel(model=request_model, openai_client=client)

    billing_agent = Agent(
        name="billing-agent",
        instructions="You handle billing questions.",
        model=model,
    )
    billing_handoff = handoff(billing_agent)

    # Per-run holder for the handoff's span context: the seam below fills it in
    # when the model calls `transfer_to_billing_agent`, and the execution-span seam
    # consumes it as the parent of the target agent's execution. Created per run,
    # so no context leaks into the next `Runner.run`.
    handoff_parent_context = {}

    # Wrap the handoff's public invoker to open the caller-owned invoke_agent span
    # around the real transfer. The entry point stays `Runner.run`; the patched
    # invoker is installed and restored in `finally` by `_patched_method` below.
    original_on_invoke_handoff = billing_handoff.on_invoke_handoff

    async def _traced_on_invoke_handoff(run_context, input_json=None):
        # `direct`: this seam belongs to the SDK's dedicated handoff API, so the
        # interaction type is intrinsic to the call, and the invoked agent is
        # `Handoff.agent_name` -- both known before the transfer runs.
        handoff_span_attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": billing_handoff.agent_name,
            "gen_ai.agent.interaction.type": "handoff",
        }
        with _reference_tracer.start_as_current_span(
            f"invoke_agent {billing_handoff.agent_name}", attributes=handoff_span_attributes
        ) as handoff_span:
            # This transfer is what causes the target agent to execute, so its
            # span context is handed to the execution-span seam to parent the
            # target's execution -- one trace, one causal chain.
            handoff_parent_context["context"] = _trace.set_span_in_context(handoff_span)
            return await original_on_invoke_handoff(run_context, input_json)

    triage_agent = Agent(
        name="triage-agent",
        instructions="You route the user to the correct specialist agent.",
        model=model,
        handoffs=[billing_handoff],
    )
    input_text = "I have a question about my bill."

    print("  [handoff] native handoff via handoff() (reference implementation)")
    # Both patched seams are installed and restored in `finally`: `_patched_method`
    # for the caller-owned handoff invoker, `_invoke_agent_execution_spans` for the
    # per-execution boundary that gives the source (triage) and the target (billing)
    # one span each. The target's response lands on the target's execution span,
    # which carries no interaction type; the caller-owned `invoke_agent billing-agent`
    # handoff op wrapped above is a child of the triage span and parents that target
    # execution, so the whole run is one trace. The entry point stays the public
    # `Runner.run(triage_agent, ...)`.
    with (
        _patched_method(billing_handoff, "on_invoke_handoff", _traced_on_invoke_handoff),
        _invoke_agent_execution_spans(
            request_model=request_model,
            input_text=input_text,
            handoff_parent_context=handoff_parent_context,
        ),
    ):
        result = await Runner.run(triage_agent, input_text)
    print(f"    -> {str(result.final_output)[:60]}")


async def run_workflow():
    """Run a multi-agent handoff wrapped in a workflow span representing the SDK workflow tracing."""
    client = openai.AsyncOpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")
    request_model = "gpt-4o-mini"
    model = OpenAIChatCompletionsModel(model=request_model, openai_client=client)

    agent_b = Agent(
        name="agent-b",
        instructions="You are agent B, tell the user the weather is sunny.",
        model=model,
    )
    agent_a = Agent(
        name="agent-a",
        instructions="You are agent A. Handoff to agent-b immediately to answer the user's weather question.",
        model=model,
        handoffs=[handoff(agent_b)],
    )
    input_text = "What's the weather in Seattle?"

    print("  [workflow_run] agent run as workflow (reference implementation)")
    workflow_name = "sequential-agents"
    workflow_span_attributes = {
        "gen_ai.operation.name": "invoke_workflow",
    }
    with _reference_tracer.start_as_current_span(
        f"invoke_workflow {workflow_name}", attributes=workflow_span_attributes
    ) as workflow_span:
        workflow_span.set_attribute("gen_ai.workflow.name", workflow_name)
        workflow_span.set_attribute(
            "gen_ai.input.messages", json.dumps([{"role": "user", "parts": [{"type": "text", "content": input_text}]}])
        )

        # Note: Agent spans (invoke_agent) are expected to be children of the
        # workflow span but are omitted for brevity in this scenario.
        result = await Runner.run(agent_a, input_text, run_config=RunConfig(workflow_name=workflow_name))

        if result.final_output:
            output_messages = json.dumps(
                [
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": str(result.final_output)}],
                    }
                ]
            )
            workflow_span.set_attribute("gen_ai.output.messages", output_messages)
        print(f"    -> {str(result.final_output)[:60]}")


def main():
    print("=== Reference Implementation: OpenAI Agents Reference Implementation ===")

    tp, lp, mp = setup_otel()

    asyncio.run(run_agent())
    asyncio.run(run_agent_as_tool_delegation())
    asyncio.run(run_agent_handoff())
    asyncio.run(run_workflow())

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
