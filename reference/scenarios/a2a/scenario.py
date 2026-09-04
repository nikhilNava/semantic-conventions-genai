"""Reference implementation for remote agent invocation with the A2A SDK."""

import asyncio
import json
import os
import uuid
from urllib.parse import urlparse

from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_message_text
from a2a.types import Message, Part, Role, SendMessageRequest
from httpx import AsyncClient
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from reference_shared import flush_and_shutdown, setup_otel

A2A_SERVER_URL = os.environ["A2A_SERVER_URL"]
_parsed = urlparse(A2A_SERVER_URL)
_SERVER_ADDRESS = _parsed.hostname or "localhost"
_SERVER_PORT = _parsed.port or 443

tracer = trace.get_tracer("gen_ai.client.a2a")


async def invoke_remote_agent() -> None:
    """Resolve an Agent Card and send a message to the advertised remote agent."""
    async with AsyncClient() as httpx_client:
        resolver = A2ACardResolver(httpx_client, A2A_SERVER_URL)
        agent_card = await resolver.get_agent_card()

    client = await create_client(
        agent_card,
        client_config=ClientConfig(streaming=False),
    )
    context_id = str(uuid.uuid4())
    request = SendMessageRequest(
        message=Message(
            role=Role.ROLE_USER,
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            parts=[Part(text="What's the weather in Seattle?")],
        )
    )
    input_text = get_message_text(request.message, delimiter=" ")

    span_attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": agent_card.name,
        "server.address": _SERVER_ADDRESS,
        "server.port": _SERVER_PORT,
    }
    try:
        with tracer.start_as_current_span(
            f"invoke_agent {agent_card.name}",
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
        ) as span:
            if agent_card.description:
                span.set_attribute("gen_ai.agent.description", agent_card.description)
            span.set_attribute("gen_ai.conversation.id", request.message.context_id)
            span.set_attribute(
                "gen_ai.input.messages",
                json.dumps([{"role": "user", "parts": [{"type": "text", "content": input_text}]}]),
            )
            output_text = ""
            async for response in client.send_message(request):
                if response.HasField("message"):
                    output_text = get_message_text(response.message, delimiter=" ")
                elif response.HasField("task"):
                    output_text = " ".join(
                        get_artifact_text(artifact, delimiter=" ") for artifact in response.task.artifacts
                    )
            if output_text:
                span.set_attribute(
                    "gen_ai.output.messages",
                    json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": output_text}]}]),
                )
            print(f"  [invoke_agent] A2A Client.send_message -> {agent_card.name}: {output_text}")
    finally:
        await client.close()


def main() -> None:
    print("=== Reference Implementation: A2A Python SDK ===")
    tp, lp, mp = setup_otel()
    asyncio.run(invoke_remote_agent())
    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
