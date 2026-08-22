#!/usr/bin/env python3
"""Voice session demo: audio in -> Omni (:8091) -> optional agent delegation (:8000).

Usage:
    .venv/bin/python scripts/voice_demo.py path/to/question.wav

Flow:
  1. Send the audio to the Omni server (it transcribes + reasons in one pass).
  2. If Omni emits an `ask_agent` tool call, forward the task to the agent
     model on :8000 (which has the big context and stronger tool use),
     then feed the result back to Omni for a final spoken-style answer.

Speech OUTPUT (TTS) is not served by vLLM (the Omni talker isn't supported
there) — pipe `reply_text` into any local TTS of your choice, or run Qwen's
own serving recipe from https://github.com/QwenLM/Qwen3-Omni for end-to-end
speech. Everything else in this file is production-shaped.
"""
import base64
import json
import sys
import urllib.request

OMNI = "http://localhost:8091/v1/chat/completions"
AGENT = "http://localhost:8000/v1/chat/completions"

ASK_AGENT_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_agent",
        "description": (
            "Delegate a complex task (data lookup, multi-step reasoning, "
            "internal systems) to the agent backend. Use for anything beyond "
            "a quick conversational answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
    },
}


def post(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main(wav_path):
    audio_b64 = base64.b64encode(open(wav_path, "rb").read()).decode()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant. Keep answers short and "
                "speakable. Delegate anything complex via ask_agent."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                }
            ],
        },
    ]

    resp = post(OMNI, {"model": "qwen3-omni-30b", "messages": messages,
                       "tools": [ASK_AGENT_TOOL]})
    msg = resp["choices"][0]["message"]

    if msg.get("tool_calls"):
        call = msg["tool_calls"][0]
        task = json.loads(call["function"]["arguments"])["task"]
        print(f"[delegating to agent] {task}")
        agent_resp = post(AGENT, {
            "model": "qwen3.6-35b-a3b",
            "messages": [{"role": "user", "content": task}],
        })
        result = agent_resp["choices"][0]["message"]["content"]
        messages += [msg, {"role": "tool", "tool_call_id": call["id"],
                           "content": result}]
        resp = post(OMNI, {"model": "qwen3-omni-30b", "messages": messages})
        msg = resp["choices"][0]["message"]

    reply_text = msg["content"]
    print(f"[assistant] {reply_text}")
    return reply_text


if __name__ == "__main__":
    main(sys.argv[1])
