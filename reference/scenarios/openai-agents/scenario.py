"""Reference implementation for OpenAI Agents.

Exercises: agent run with tool calling, agent-as-tool delegation, and native
handoff against a mock OpenAI server, with manual OTel spans.
"""

import asyncio
import contextlib
import json
import os
import time

from reference_shared import flush_and_shutdown, reference_meter, reference_tracer, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"

_reference_tracer = reference_tracer()
_reference_meter = reference_meter()

# `gen_ai.execute_tool.duration` carries `gen_ai.agent.interaction.type` when the
# tool call is a delegation/handoff to another agent; `gen_ai.agent.name` then
# identifies the target agent (see model/gen-ai/metrics.yaml).
_execute_tool_duration = _reference_meter.create_histogram(
    "gen_ai.execute_tool.duration",
    unit="s",
    description="The duration of a single tool execution.",
)


@contextlib.contextmanager
def _invoke_agent_span_per_turn(*, request_model, first_input_text=None):
    """Open one caller-owned ``invoke_agent <agent.name>`` span per real agent
    execution, by wrapping the SDK's private per-turn ``run_single_turn``, then
    restore it on exit.

    This is the shared multi-agent seam used by both the native-handoff and the
    agent-as-tool delegation scenarios. Patching this private function is
    allowed; the scenario's entry point stays the public ``Runner.run``, and this
    seam is never called directly. Because every agent turn -- including a nested
    agent run started by ``Agent.as_tool``'s own ``Runner.run`` -- flows through
    ``run_single_turn``, the wrapper also gives a delegated target agent its own
    ``invoke_agent`` span, correctly nested under the caller-owned
    ``execute_tool`` span that drove the delegation.

    Each executing agent's own response goes on its span; the span carries no
    ``gen_ai.agent.interaction.type`` -- that is caller-owned and set on the
    ``execute_tool`` operation instead. ``first_input_text``, when given, is set
    as ``gen_ai.input.messages`` on the first (source) execution only.
    """
    import agents.run

    original_run_single_turn = agents.run.run_single_turn
    state = {"first_turn": True}

    async def _traced_run_single_turn(**kwargs):
        executing_agent = kwargs["bindings"].public_agent
        agent_span_attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.request.model": request_model,
            "gen_ai.agent.name": executing_agent.name,
        }
        with _reference_tracer.start_as_current_span(
            f"invoke_agent {executing_agent.name}", attributes=agent_span_attributes
        ) as agent_span:
            # The user's message is the input to the first (source) execution only.
            if first_input_text is not None and state["first_turn"]:
                state["first_turn"] = False
                agent_span.set_attribute(
                    "gen_ai.input.messages",
                    json.dumps([{"role": "user", "parts": [{"type": "text", "content": first_input_text}]}]),
                )
            turn_result = await original_run_single_turn(**kwargs)
            # Only a final-output step exposes `.output` (a handoff step exposes
            # `.new_agent`); the executing agent's own response goes on its span,
            # never on the caller's.
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


async def run_agent():
    """Run a simple agent with the OpenAI Agents SDK, with manual spans."""
    import openai
    from agents import Agent, Runner, function_tool
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from agents.tool import FunctionTool, ToolContext

    @function_tool
    def get_weather(ctx: ToolContext[None], location: str) -> str:
        """Get the current weather for a location."""
        tool_span_attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "get_weather",
            "gen_ai.tool.type": "function",
        }
        with _reference_tracer.start_as_current_span(
            "execute_tool get_weather", attributes=tool_span_attributes
        ) as tool_span:
            tool_span.set_attribute("gen_ai.tool.description", get_weather.description)
            if ctx.agent is not None and ctx.agent.name:
                tool_span.set_attribute("gen_ai.agent.name", ctx.agent.name)
            tool_span.set_attribute("gen_ai.tool.call.id", ctx.tool_call_id)
            tool_span.set_attribute("gen_ai.tool.call.arguments", json.dumps({"location": location}))
            result = "Sunny, 72°F"
            tool_span.set_attribute("gen_ai.tool.call.result", result)
            return result

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

    Two agents really execute under one public `Runner.run(caller, ...)` call
    (the delegated agent runs through `Agent.as_tool`'s own nested `Runner.run`),
    so the shared `_invoke_agent_span_per_turn` seam emits:

    * the caller `invoke_agent assistant` span(s) for the caller's own turns, and
    * the target `invoke_agent weather-specialist` span, nested under the
      caller-owned `execute_tool weather-specialist` operation, carrying the
      delegated agent's response. Being the target's own execution, it stays free
      of `gen_ai.agent.interaction.type` -- that is caller-owned and lives on the
      `execute_tool` span, which names the target.

    The delegated agent's model call belongs to the `openai` library, so no
    inference span is emitted here.
    """
    import openai
    from agents import Agent, Runner
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

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
    # around the real sub-agent invocation. The entry point stays `Runner.run`.
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
                result = await original_on_invoke_tool(tool_context, input_json)
                # Success-only: set the result only after the call returns, so a
                # failure never manufactures a fabricated result attribute.
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

    weather_tool.on_invoke_tool = _traced_on_invoke_tool

    caller = Agent(
        name="assistant",
        instructions="You are a helpful assistant. Delegate weather questions to the specialist.",
        model=model,
        tools=[weather_tool],
    )
    input_text = "What's the weather in Seattle?"

    print("  [delegation] agent-as-tool via Agent.as_tool (reference implementation)")
    # The shared per-turn seam also wraps the nested `Agent.as_tool` `Runner.run`,
    # so the delegated target `weather-specialist` gets its own
    # `invoke_agent weather-specialist` span (child of the caller-owned
    # `execute_tool weather-specialist` span above) carrying its response and no
    # interaction type, while the caller's turns get their own `invoke_agent
    # assistant` spans. The entry point stays the public `Runner.run(caller, ...)`.
    with _invoke_agent_span_per_turn(request_model=request_model, first_input_text=input_text):
        result = await Runner.run(caller, input_text)
    print(f"    -> {str(result.final_output)[:60]}")


async def run_agent_handoff():
    """Handoff: a caller agent transfers control to another agent via `handoff`.

    `handoff()` is the SDK's explicit handoff API. The model calls the generated
    `transfer_to_<agent>` tool and the SDK switches the active agent, which then
    owns the remaining work -- a `handoff` interaction, mapped directly from the
    API being invoked.

    Two agents really execute under one public `Runner.run(triage_agent, ...)`
    call, so two `invoke_agent` spans are emitted:

    * the caller/source `invoke_agent triage-agent` span, which bounds the triage
      agent's own turn and (as its child) the immediate
      `execute_tool transfer_to_billing_agent` operation. That execute_tool span
      carries `gen_ai.agent.interaction.type=handoff` and names the target from
      `Handoff.agent_name`.
    * the target `invoke_agent billing-agent` span, which bounds the billing
      agent's real execution after the switch and carries that agent's response.
      Being the target's own execution, it must stay free of the interaction
      attribute -- the interaction type is caller-owned.

    Each agent turn runs through the SDK's private `run_single_turn`; the shared
    `_invoke_agent_span_per_turn` seam opens the per-agent span around the real
    execution while the scenario's entry point stays the public `Runner.run`.
    """
    import openai
    from agents import Agent, Runner, handoff
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

    client = openai.AsyncOpenAI(base_url=MOCK_BASE_URL, api_key="mock-key")
    request_model = "gpt-4o-mini"
    model = OpenAIChatCompletionsModel(model=request_model, openai_client=client)

    billing_agent = Agent(
        name="billing-agent",
        instructions="You handle billing questions.",
        model=model,
    )
    billing_handoff = handoff(billing_agent)

    # Wrap the handoff's public invoker to open the caller-owned execute_tool span
    # around the real transfer. The entry point stays `Runner.run`.
    original_on_invoke_handoff = billing_handoff.on_invoke_handoff

    async def _traced_on_invoke_handoff(run_context, input_json=None):
        tool_span_attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": billing_handoff.tool_name,
            "gen_ai.tool.type": "function",
        }
        start = time.perf_counter()
        try:
            with _reference_tracer.start_as_current_span(
                f"execute_tool {billing_handoff.tool_name}", attributes=tool_span_attributes
            ) as tool_span:
                # `direct`: `handoff()` is a handoff API, so the type is intrinsic to
                # the call, and the target is `Handoff.agent_name`.
                tool_span.set_attribute("gen_ai.agent.interaction.type", "handoff")
                tool_span.set_attribute("gen_ai.agent.name", billing_handoff.agent_name)
                target_agent = await original_on_invoke_handoff(run_context, input_json)
            return target_agent
        finally:
            # Record in `finally` so a failed transfer is still timed. Every
            # dimension is the operation's target identity, known before the call,
            # so no success-only result is manufactured on failure.
            _execute_tool_duration.record(
                time.perf_counter() - start,
                {
                    "gen_ai.tool.name": billing_handoff.tool_name,
                    "gen_ai.tool.type": "function",
                    "gen_ai.agent.interaction.type": "handoff",
                    "gen_ai.agent.name": billing_handoff.agent_name,
                },
            )

    billing_handoff.on_invoke_handoff = _traced_on_invoke_handoff

    triage_agent = Agent(
        name="triage-agent",
        instructions="You route the user to the correct specialist agent.",
        model=model,
        handoffs=[billing_handoff],
    )
    input_text = "I have a question about my bill."

    print("  [handoff] native handoff via handoff() (reference implementation)")
    # The shared per-turn seam gives the source (triage) and the target (billing)
    # each their own `invoke_agent` span: the target's response lands on the
    # target span, which carries no interaction type. The immediate transfer stays
    # the `execute_tool transfer_to_billing_agent` op wrapped above (a child of the
    # triage span). The entry point stays the public `Runner.run(triage_agent, ...)`.
    with _invoke_agent_span_per_turn(request_model=request_model, first_input_text=input_text):
        result = await Runner.run(triage_agent, input_text)
    print(f"    -> {str(result.final_output)[:60]}")


def main():
    print("=== Reference Implementation: OpenAI Agents Reference Implementation ===")

    tp, lp, mp = setup_otel()

    asyncio.run(run_agent())
    asyncio.run(run_agent_as_tool_delegation())
    asyncio.run(run_agent_handoff())

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
