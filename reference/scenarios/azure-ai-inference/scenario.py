"""Reference implementation for Microsoft Foundry Models."""

import json
import os
from urllib.parse import urlparse

import httpx2 as httpx
from openai import OpenAI
from reference_shared import (
    flush_and_shutdown,
    reference_event_logger,
    reference_tracer,
    setup_otel,
)

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]
FOUNDRY_BASE_URL = "https://foundry-resource.openai.azure.com/openai/v1/"

_reference_tracer = reference_tracer()


def _route_foundry_request_to_mock(request):
    foundry_url = urlparse(FOUNDRY_BASE_URL)
    mock_url = urlparse(MOCK_BASE_URL)
    foundry_path = foundry_url.path.rstrip("/")
    request_path = request.url.path
    if not request_path.startswith(foundry_path):
        raise ValueError(f"unexpected Foundry request path: {request_path}")

    request.url = request.url.copy_with(
        scheme=mock_url.scheme,
        host=mock_url.hostname,
        port=mock_url.port,
        path=f"{mock_url.path.rstrip('/')}/v1{request_path.removeprefix(foundry_path)}",
    )
    request.headers["host"] = request.url.netloc.decode("ascii")


def _inference_attributes(client, operation_name, request_model):
    endpoint = urlparse(str(client.base_url))
    if not endpoint.hostname or not endpoint.hostname.endswith(".openai.azure.com"):
        raise ValueError(f"unsupported Microsoft Foundry Models endpoint: {client.base_url}")

    attributes = {
        "gen_ai.operation.name": operation_name,
        "gen_ai.provider.name": "azure.ai.inference",
        "gen_ai.request.model": request_model,
        "server.address": endpoint.hostname,
    }
    if endpoint.port and endpoint.port != 443:
        attributes["server.port"] = endpoint.port
    return attributes


def run_chat_reference(client):
    """Scenario: basic chat completion with reference implementation."""
    print("  [chat] basic chat completion (reference implementation)")
    request_model = "gpt-4o-mini"
    span_attributes = _inference_attributes(client, "chat", request_model)
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini", attributes=span_attributes) as span:
        user_content = "Say hello."
        response = client.chat.completions.create(
            model=request_model,
            messages=[{"role": "user", "content": user_content}],
        )
        span.set_attribute("gen_ai.response.model", response.model)
        span.set_attribute("gen_ai.response.id", response.id)
        finish_reasons = [choice.finish_reason or "error" for choice in response.choices]
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
        if response.usage:
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)

        event_attributes = {
            **span_attributes,
            "gen_ai.response.id": response.id,
            "gen_ai.response.model": response.model,
            "gen_ai.response.finish_reasons": finish_reasons,
            "gen_ai.input.messages": json.dumps(
                [{"role": "user", "parts": [{"type": "text", "content": user_content}]}]
            ),
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": choice.message.content}],
                    }
                    for choice in response.choices
                ]
            ),
        }
        if response.usage:
            event_attributes["gen_ai.usage.input_tokens"] = response.usage.prompt_tokens
            event_attributes["gen_ai.usage.output_tokens"] = response.usage.completion_tokens
        reference_event_logger().emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attributes,
        )

        print(f"    -> {response.choices[0].message.content[:60]}")


def run_chat_tool_call_reference(client):
    """Scenario: chat with tool calling with reference implementation."""
    print("  [chat_tool_call] chat with tool calling (reference implementation)")
    request_model = "gpt-4o-mini"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    tool_definitions = [
        {
            "type": tool["type"],
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "parameters": tool["function"]["parameters"],
        }
        for tool in tools
    ]
    span_attributes = _inference_attributes(client, "chat", request_model)
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini", attributes=span_attributes) as span:
        span.set_attribute("gen_ai.tool.definitions", json.dumps(tool_definitions))
        response = client.chat.completions.create(
            model=request_model,
            messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
            tools=tools,
        )
        span.set_attribute("gen_ai.response.model", response.model)
        span.set_attribute("gen_ai.response.id", response.id)
        span.set_attribute(
            "gen_ai.response.finish_reasons",
            [choice.finish_reason or "error" for choice in response.choices],
        )
        if response.usage:
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
        choice = response.choices[0]
        if choice.message.tool_calls:
            print(f"    -> tool_call: {choice.message.tool_calls[0].function.name}")
        else:
            print(f"    -> {choice.message.content[:60]}")


def run_chat_streaming_reference(client):
    """Scenario: streaming chat completion with reference implementation."""
    print("  [chat_streaming] streaming chat completion (reference implementation)")
    request_model = "gpt-4o-mini"
    span_attributes = _inference_attributes(client, "chat", request_model)
    span_attributes["gen_ai.request.stream"] = True
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini", attributes=span_attributes) as span:
        stream = client.chat.completions.create(
            model=request_model,
            messages=[{"role": "user", "content": "Tell me a joke."}],
            stream=True,
            stream_options={"include_usage": True},
        )
        text = ""
        response_model = None
        response_id = None
        seen_choice_indexes = set()
        finish_reasons_by_index = {}
        input_tokens = None
        output_tokens = None
        for chunk in stream:
            response_model = response_model or chunk.model
            response_id = response_id or chunk.id
            for choice in chunk.choices:
                seen_choice_indexes.add(choice.index)
                if choice.index == 0 and choice.delta.content:
                    text += choice.delta.content
                if choice.finish_reason is not None:
                    finish_reasons_by_index[choice.index] = choice.finish_reason
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
        if response_model:
            span.set_attribute("gen_ai.response.model", response_model)
        if response_id:
            span.set_attribute("gen_ai.response.id", response_id)
        if seen_choice_indexes:
            span.set_attribute(
                "gen_ai.response.finish_reasons",
                [
                    finish_reasons_by_index.get(index, "error")
                    for index in range(max(seen_choice_indexes) + 1)
                ],
            )
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        print(f"    -> {text[:60]}")


def run_embeddings_reference(client):
    """Scenario: embedding generation with reference implementation."""
    print("  [embeddings] embedding generation (reference implementation)")
    request_model = "text-embedding-3-small"
    span_attributes = _inference_attributes(client, "embeddings", request_model)
    span_attributes["gen_ai.request.encoding_formats"] = ["float"]
    with _reference_tracer.start_as_current_span(
        "embeddings text-embedding-3-small", attributes=span_attributes
    ) as span:
        response = client.embeddings.create(
            model=request_model,
            input=["Hello, world!"],
            encoding_format="float",
        )
        span.set_attribute("gen_ai.response.model", response.model)
        span.set_attribute("gen_ai.embeddings.dimension.count", len(response.data[0].embedding))
        if response.usage:
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        print(f"    -> embedding dim: {len(response.data[0].embedding)}")


def main():
    print("=== Reference Implementation: Microsoft Foundry Models ===")

    tp, lp, mp = setup_otel()
    http_client = httpx.Client(event_hooks={"request": [_route_foundry_request_to_mock]})
    client = OpenAI(
        base_url=FOUNDRY_BASE_URL,
        api_key="mock-key",
        http_client=http_client,
    )

    try:
        run_chat_reference(client)
        run_chat_tool_call_reference(client)
        run_chat_streaming_reference(client)
        run_embeddings_reference(client)
    finally:
        client.close()
        flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
