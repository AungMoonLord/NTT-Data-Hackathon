"""Step 7 – LLM agent (Amazon Bedrock Converse API with tool use)."""
import json
import os
from datetime import datetime

import boto3
from botocore.config import Config

from tools import BKK, TOOL_SPECS, dispatch, parse_iso

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-pro-v1:0")
PRICE_IN = float(os.environ.get("PRICE_IN_PER_M", "0.8"))
PRICE_OUT = float(os.environ.get("PRICE_OUT_PER_M", "3.2"))
USD_THB = float(os.environ.get("USD_THB", "35"))

SYSTEM_PROMPT = """You are MechAgent, an AI reliability engineer for rotating equipment in a Thai factory.
Rules:
1. All numbers (degradation %, RUL, safety deadline) come from the incident context and tools. NEVER invent or alter them.
2. Workflow: get_machine_profile -> get_health_history -> search_maintenance_manual -> check_spare_parts
   -> get_production_schedule -> (adjust_monitoring_interval to 15-30 min if severity >= S2) -> submit_for_approval.
3. planned_start MUST be before the safety deadline. Prefer a planned production stop; if none fits, propose an
   emergency stop (or switch to the standby machine) and say so explicitly.
4. If the bearing is out of stock, propose a substitute or expedite purchase considering lead time.
5. If the history trend RUL conflicts strongly with the model RUL, mention it and recommend re-measurement.
6. After submitting, reply in Thai (concise, bullet points): root cause, plan & time window, parts, risks,
   technician steps from the manual."""


def run_agent(ctx, max_steps=10):
    br = boto3.client("bedrock-runtime", region_name=os.environ.get("BEDROCK_REGION") or None,
                      config=Config(read_timeout=120, retries={"max_attempts": 4, "mode": "adaptive"}))
    messages = [{"role": "user", "content": [{"text":
                "Incident context (JSON):\n" + json.dumps(ctx["brief"], ensure_ascii=False) +
                "\nPlan the maintenance and submit it for approval."}]}]
    trace, usage, final_text, step = [], {"inputTokens": 0, "outputTokens": 0}, "", 0
    for step in range(max_steps):
        resp = br.converse(modelId=MODEL_ID, system=[{"text": SYSTEM_PROMPT}], messages=messages,
                           toolConfig={"tools": TOOL_SPECS},
                           inferenceConfig={"maxTokens": 1500, "temperature": 0.2})
        for k in usage:
            usage[k] += resp.get("usage", {}).get(k, 0)
        msg = resp["output"]["message"]
        messages.append(msg)
        if resp["stopReason"] != "tool_use":
            final_text = "\n".join(c["text"] for c in msg["content"] if "text" in c)
            break
        results = []
        for c in msg["content"]:
            if "toolUse" in c:
                tu = c["toolUse"]
                out = dispatch(tu["name"], tu.get("input", {}), ctx)
                trace.append({"step": step + 1, "tool": tu["name"], "input": tu.get("input", {}),
                              "output": json.dumps(out, ensure_ascii=False)[:600]})
                results.append({"toolResult": {"toolUseId": tu["toolUseId"], "content": [{"json": out}]}})
        messages.append({"role": "user", "content": results})

    if not ctx.get("submitted"):   # GUARDRAIL: always hand off to a human
        now, deadline = datetime.now(BKK), parse_iso(ctx["deadline_iso"])
        start = now + (deadline - now) / 2 if deadline > now else now
        dispatch("submit_for_approval", {"recommended_action": ctx["report"]["summary"]["recommended_action"],
                                         "planned_start": start.isoformat(),
                                         "summary_th": "[Fallback] ระบบส่งแผนตามรายงานมาตรฐาน " + final_text[:500]}, ctx)
        trace.append({"step": "fallback", "tool": "submit_for_approval", "input": {}, "output": "auto"})
    cost_usd = (usage["inputTokens"] * PRICE_IN + usage["outputTokens"] * PRICE_OUT) / 1e6
    return {"final_text": final_text, "trace": trace, "steps": step + 1, "model_id": MODEL_ID,
            "usage": {**usage, "cost_usd": round(cost_usd, 5), "cost_thb": round(cost_usd * USD_THB, 3)}}
