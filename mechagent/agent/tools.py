"""Agent tools + shared AWS helpers (DynamoDB single-table, SNS/Telegram, audit, guardrails)."""
import hashlib
import hmac
import json
import math
import os
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

BKK = timezone(timedelta(hours=7))
_s3, _sns, _table, _MANUAL = boto3.client("s3"), boto3.client("sns"), None, None


def table():
    global _table
    if _table is None:
        _table = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])
    return _table


def to_ddb(o):
    return json.loads(json.dumps(o, default=str), parse_float=Decimal)


def from_ddb(o):
    if isinstance(o, list):
        return [from_ddb(x) for x in o]
    if isinstance(o, dict):
        return {k: from_ddb(v) for k, v in o.items()}
    if isinstance(o, Decimal):
        return int(o) if o == o.to_integral_value() else float(o)
    return o


def parse_iso(s):
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=BKK)


def update_job(job_id, **fields):
    names, values, sets = {"#u": "updated_at"}, {":u": datetime.now(BKK).isoformat()}, ["#u = :u"]
    for i, (k, v) in enumerate(fields.items()):
        names[f"#k{i}"], values[f":v{i}"] = k, to_ddb(v)
        sets.append(f"#k{i} = :v{i}")
    table().update_item(Key={"pk": f"JOB#{job_id}", "sk": "REPORT"}, UpdateExpression="SET " + ", ".join(sets),
                        ExpressionAttributeNames=names, ExpressionAttributeValues=values)


def audit(job_id, event, detail=None):
    table().put_item(Item=to_ddb({"pk": f"JOB#{job_id}", "sk": f"AUDIT#{datetime.now(BKK).isoformat()}",
                                  "event": event, "detail": detail or {}}))


def approval_token(job_id, action):
    secret = os.environ.get("APPROVAL_SECRET", "change-me").encode()
    return hmac.new(secret, f"{job_id}:{action}".encode(), hashlib.sha256).hexdigest()[:32]


def notify(subject, text, links=None):
    if os.environ.get("TOPIC_ARN"):
        body = text + ("\n\n" + "\n".join(f"{k.upper()}: {v}" for k, v in links.items()) if links else "")
        _sns.publish(TopicArn=os.environ["TOPIC_ARN"], Subject=subject[:99], Message=body)
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if tok and chat:
        payload = {"chat_id": chat, "text": text[:4000]}
        if links:
            payload["reply_markup"] = {"inline_keyboard": [[
                {"text": "Approve", "url": links["approve"]},
                {"text": "Postpone 24h", "url": links["postpone"]},
                {"text": "Reject", "url": links["reject"]}]]}
        req = urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:  # never fail the job because of chat delivery
            print("telegram error:", e)


# ------------------------------- TOOLS -------------------------------
def get_machine_profile(ctx, machine_id):
    item = table().get_item(Key={"pk": f"MACHINE#{machine_id}", "sk": "PROFILE"}).get("Item")
    return from_ddb(item) if item else {"error": f"machine {machine_id} not found"}


def exp_trend_rul(readings, failure_pct=100.0):
    pts = []
    for r in readings:
        deg = float(r.get("deg_pct", 0))
        if deg > 1:
            pts.append((parse_iso(r["sk"].split("READING#", 1)[1]).timestamp() / 3600, math.log(deg)))
    if len(pts) < 3:
        return {"status": "insufficient_history", "n_points": len(pts)}
    t0 = pts[0][0]
    xs, ys = [p[0] - t0 for p in pts], [p[1] for p in pts]
    n, mx, my = len(xs), sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return {"status": "insufficient_time_span"}
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if b <= 1e-6:
        return {"status": "no_growth_trend", "growth_rate_per_h": b}
    t_fail = (math.log(failure_pct) - a) / b
    return {"status": "ok", "model": "deg(t)=exp(a+b*t)", "growth_rate_per_h": round(b, 5),
            "rul_hours_trend": round(max(t_fail - xs[-1], 0), 1), "n_points": n}


def get_health_history(ctx, machine_id, limit=30):
    resp = table().query(KeyConditionExpression=Key("pk").eq(f"MACHINE#{machine_id}") & Key("sk").begins_with("READING#"),
                         ScanIndexForward=False, Limit=int(limit))
    items = from_ddb(resp.get("Items", []))[::-1]
    return {"readings": [{"ts": i["sk"][8:], "deg_pct": i.get("deg_pct"), "fault": i.get("fault"),
                          "rul_p50_h": i.get("rul_p50")} for i in items],
            "trend": exp_trend_rul(items)}


def _manual_sections():
    global _MANUAL
    if _MANUAL is None:
        body = _s3.get_object(Bucket=os.environ["BUCKET"],
                              Key=os.environ.get("MANUAL_KEY", "manuals/bearing_manual.md"))["Body"].read().decode()
        secs = []
        for block in body.split("\n## ")[1:]:
            title, _, text = block.partition("\n")
            m = re.search(r"tags:\s*(.*)", text)
            secs.append({"title": title.strip(), "text": text.strip(),
                         "tags": [t.strip() for t in m.group(1).split(",")] if m else []})
        _MANUAL = secs
    return _MANUAL


def search_maintenance_manual(ctx, query, fault_type=None, top_k=2):
    terms = [t.lower() for t in re.findall(r"\w+", query)]
    scored = []
    for s in _manual_sections():
        blob = (s["title"] + " " + s["text"]).lower()
        score = 5 * (fault_type in s["tags"]) + sum(t in blob for t in terms) + 2 * ("replacement" in s["tags"])
        scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return {"results": [{"title": s["title"], "content": s["text"][:1500]} for _, s in scored[:top_k]]}


def check_spare_parts(ctx, bearing_model):
    item = table().get_item(Key={"pk": f"PART#{bearing_model.strip()}", "sk": "STOCK"}).get("Item")
    return from_ddb(item) if item else {"error": f"no stock record for {bearing_model}"}


def get_production_schedule(ctx, machine_id):
    item = from_ddb(table().get_item(Key={"pk": f"MACHINE#{machine_id}", "sk": "SCHEDULE"}).get("Item") or {})
    now = datetime.now(BKK)
    stops = [s for s in item.get("planned_stops", []) if parse_iso(s["end"]) > now]
    return {"now": now.isoformat(), "safety_deadline": ctx.get("deadline_iso"), "planned_stops": stops}


def adjust_monitoring_interval(ctx, machine_id, interval_minutes):
    v = int(min(max(int(interval_minutes), 5), 1440))
    table().update_item(Key={"pk": f"MACHINE#{machine_id}", "sk": "PROFILE"},
                        UpdateExpression="SET monitoring_interval_min = :v", ExpressionAttributeValues={":v": v})
    audit(ctx["job_id"], "monitoring_interval_changed", {"machine_id": machine_id, "minutes": v})
    return {"ok": True, "monitoring_interval_min": v}


def submit_for_approval(ctx, recommended_action, planned_start, summary_th, planned_end=None, parts_plan=""):
    deadline, start = parse_iso(ctx["deadline_iso"]), parse_iso(planned_start)
    if start > deadline:   # GUARDRAIL: plan must respect the safety deadline
        audit(ctx["job_id"], "guardrail_block", {"planned_start": planned_start, "deadline": ctx["deadline_iso"]})
        return {"accepted": False,
                "guardrail": f"planned_start {start.isoformat()} is AFTER safety deadline {deadline.isoformat()}. "
                             "Choose an earlier window (emergency stop if no planned stop fits)."}
    job = ctx["job_id"]
    plan = {"recommended_action": recommended_action, "planned_start": start.isoformat(),
            "planned_end": planned_end, "parts_plan": parts_plan, "summary_th": summary_th}
    update_job(job, status="PENDING_APPROVAL", plan=plan)
    links = {a: f"{ctx['api_base']}/approve?job={job}&action={a}&token={approval_token(job, a)}"
             for a in ("approve", "postpone", "reject")}
    notify(f"MechAgent approval needed: {ctx['machine_id']}",
           ctx["report"]["telegram"] + f"\n\nแผนของ Agent:\n{summary_th}\nเริ่มงาน: {start:%d/%m %H:%M}\nอะไหล่: {parts_plan}",
           links)
    ctx["submitted"] = True
    audit(job, "submitted_for_approval", plan)
    return {"accepted": True, "status": "PENDING_APPROVAL"}


REGISTRY = {f.__name__: f for f in [get_machine_profile, get_health_history, search_maintenance_manual,
                                     check_spare_parts, get_production_schedule, adjust_monitoring_interval,
                                     submit_for_approval]}


def _spec(name, desc, props, required):
    return {"toolSpec": {"name": name, "description": desc,
                         "inputSchema": {"json": {"type": "object", "properties": props, "required": required}}}}


S = {"type": "string"}
TOOL_SPECS = [
    _spec("get_machine_profile", "Machine master data: bearing model, criticality, standby machine.", {"machine_id": S}, ["machine_id"]),
    _spec("get_health_history", "Recent degradation readings + exponential-trend RUL.", {"machine_id": S, "limit": {"type": "integer"}}, ["machine_id"]),
    _spec("search_maintenance_manual", "Search maintenance manual (RAG) for procedures.", {"query": S, "fault_type": S}, ["query"]),
    _spec("check_spare_parts", "Warehouse stock, lead time, substitutes for a bearing model e.g. 'SKF 6205-2RS'.", {"bearing_model": S}, ["bearing_model"]),
    _spec("get_production_schedule", "Upcoming planned production stops and the safety deadline.", {"machine_id": S}, ["machine_id"]),
    _spec("adjust_monitoring_interval", "Change vibration measurement interval in minutes (5-1440).", {"machine_id": S, "interval_minutes": {"type": "integer"}}, ["machine_id", "interval_minutes"]),
    _spec("submit_for_approval", "Submit final plan for human approval. planned_start = ISO-8601, must be before safety deadline.",
          {"recommended_action": S, "planned_start": S, "planned_end": S, "summary_th": S, "parts_plan": S},
          ["recommended_action", "planned_start", "summary_th"]),
]


def dispatch(name, args, ctx):
    fn = REGISTRY.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        out = fn(ctx, **(args or {}))
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}"}
    out = json.loads(json.dumps(out, default=str))
    return out if isinstance(out, dict) else {"result": out}
