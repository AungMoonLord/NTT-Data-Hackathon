"""Lambda entrypoint: API routes + async worker (Steps 3-8)."""
import base64
import html
import hmac
import json
import os
import traceback
import urllib.parse
import uuid
from datetime import datetime, timedelta

import boto3
from botocore.config import Config

from agent_core import run_agent
from report import build_report
from tools import BKK, approval_token, audit, from_ddb, notify, parse_iso, table, to_ddb, update_job

s3 = boto3.client("s3")
smr = boto3.client("sagemaker-runtime", config=Config(read_timeout=70, retries={"max_attempts": 3}))
lam = boto3.client("lambda")
ACTIONS = {"approve": "APPROVED", "postpone": "POSTPONED_24H", "reject": "REJECTED"}
HTML = "text/html; charset=utf-8"


def _resp(code, body, ctype="application/json"):
    return {"statusCode": code, "headers": {"Content-Type": ctype, "Access-Control-Allow-Origin": "*"},
            "body": body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, default=str)}


def _body(event):
    b = event.get("body") or ""
    return base64.b64decode(b).decode() if event.get("isBase64Encoded") else b


def lambda_handler(event, context):
    if event.get("mode") == "worker":
        return worker(event)
    route = event.get("routeKey", "")
    try:
        if route == "POST /analyze":
            return analyze(event)
        if route == "GET /reports/{job_id}":
            return get_report(event)
        if route == "GET /approve":
            return approve_page(event)
        if route == "POST /approve":
            return approve_submit(event)
        return _resp(404, {"error": f"unknown route {route}"})
    except Exception as e:
        traceback.print_exc()
        return _resp(500, {"error": str(e)})


def analyze(event):
    req = json.loads(_body(event) or "{}")
    missing = [k for k in ("machine_id", "rpm", "signal") if k not in req]
    if missing:
        return _resp(400, {"error": f"missing fields: {missing}"})
    job_id = uuid.uuid4().hex[:12]
    key = f"raw/{job_id}.json"
    s3.put_object(Bucket=os.environ["BUCKET"], Key=key,
                  Body=json.dumps({"signal": req["signal"], "fs": req.get("fs", 16000)}).encode())
    table().put_item(Item=to_ddb({"pk": f"JOB#{job_id}", "sk": "REPORT", "status": "QUEUED", "done": False,
                                  "machine_id": req["machine_id"], "created_at": datetime.now(BKK).isoformat()}))
    api_base = f"https://{event.get('requestContext', {}).get('domainName', '')}"
    lam.invoke(FunctionName=os.environ["SELF_FUNCTION_NAME"], InvocationType="Event",
               Payload=json.dumps({"mode": "worker", "job_id": job_id, "machine_id": req["machine_id"],
                                   "rpm": float(req["rpm"]), "bearing": req.get("bearing"),
                                   "s3_key": key, "api_base": api_base}).encode())
    return _resp(202, {"job_id": job_id, "status": "QUEUED", "status_url": f"{api_base}/reports/{job_id}"})


def gatekeeper(pred, profile, now):
    """Step 6 – rule-based filter: call the LLM only when needed (cost & noise control)."""
    reasons = []
    if pred["fault"] != "normal" and pred["confidence"] >= 0.6:
        reasons.append(f"AI detected {pred['fault']} ({pred['confidence']:.0%})")
    if pred["degradation_pct"] >= 30:
        reasons.append(f"degradation {pred['degradation_pct']:.1f}% >= 30%")
    if pred["rul_hours"]["p05"] <= 168:
        reasons.append(f"RUL P5 {pred['rul_hours']['p05']:.0f} h <= 7 days")
    if pred["physics"]["physics_fault"] != "normal" and pred["degradation_pct"] >= 10:
        reasons.append(f"physics peak at {pred['physics']['physics_fault']}")
    if not reasons:
        return False, ["healthy, no LLM call"]
    last = profile.get("last_alert_ts")
    if last and pred["rul_hours"]["p05"] > 24 and now - parse_iso(last) < timedelta(hours=12):
        return False, reasons + ["dedup: alerted < 12 h ago"]
    return True, reasons


def worker(ev):
    job_id, mid = ev["job_id"], ev["machine_id"]
    try:
        update_job(job_id, status="ANALYZING")
        raw = json.loads(s3.get_object(Bucket=os.environ["BUCKET"], Key=ev["s3_key"])["Body"].read())
        profile = from_ddb(table().get_item(Key={"pk": f"MACHINE#{mid}", "sk": "PROFILE"}).get("Item")) or {"machine_id": mid}
        bearing = ev.get("bearing") or profile.get("bearing_key", "6205")
        r = smr.invoke_endpoint(EndpointName=os.environ["ENDPOINT_NAME"], ContentType="application/json",
                                Accept="application/json",
                                Body=json.dumps({"signal": raw["signal"], "fs": raw.get("fs", 16000),
                                                 "rpm": ev["rpm"], "bearing": bearing, "mc_samples": 10}))
        pred = json.loads(r["Body"].read())
        now = datetime.now(BKK)
        report = build_report(pred, profile, now=now, job_id=job_id)
        table().put_item(Item=to_ddb({"pk": f"MACHINE#{mid}", "sk": f"READING#{now.isoformat()}",
                                      "deg_pct": pred["degradation_pct"], "fault": pred["fault"],
                                      "confidence": pred["confidence"], "rul_p50": pred["rul_hours"]["p50"],
                                      "rul_p05": pred["rul_hours"]["p05"]}))
        trigger, reasons = gatekeeper(pred, profile, now)
        result = {"prediction": pred, "report": report, "gatekeeper": {"trigger": trigger, "reasons": reasons}}
        audit(job_id, "analyzed", {"fault": pred["fault"], "deg": pred["degradation_pct"], "gatekeeper": reasons})
        if trigger:
            update_job(job_id, status="AGENT_RUNNING", **result)
            ctx = {"job_id": job_id, "machine_id": mid, "api_base": ev["api_base"], "report": report,
                   "deadline_iso": report["summary"]["replace_by_iso"],
                   "brief": {"job_id": job_id, "machine_id": mid, "now": now.isoformat(),
                             "safety_deadline_iso": report["summary"]["replace_by_iso"],
                             "gatekeeper_reasons": reasons, "report_summary": report["summary"],
                             "class_probs": pred["class_probs"]}}
            result["agent"] = run_agent(ctx)
            result["status"] = "PENDING_APPROVAL"
            table().update_item(Key={"pk": f"MACHINE#{mid}", "sk": "PROFILE"},
                                UpdateExpression="SET last_alert_ts = :t", ExpressionAttributeValues={":t": now.isoformat()})
        else:
            result["status"] = "OK_NO_ACTION"
        update_job(job_id, done=True, **result)
        return {"ok": True}
    except Exception as e:
        traceback.print_exc()
        update_job(job_id, status="FAILED", done=True, error=str(e))
        return {"ok": False}


def get_report(event):
    job_id = event["pathParameters"]["job_id"]
    items = from_ddb(table().query(KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq(f"JOB#{job_id}"))["Items"])
    job = next((i for i in items if i["sk"] == "REPORT"), None)
    if not job:
        return _resp(404, {"error": "job not found"})
    return _resp(200, {"job": job, "audit": [i for i in items if i["sk"].startswith("AUDIT#")]})


def _valid(job, action, token):
    return action in ACTIONS and hmac.compare_digest(str(token), approval_token(job, action))


def approve_page(event):
    """GET shows a confirm button, so email link-scanners cannot auto-approve."""
    q = event.get("queryStringParameters") or {}
    job, action, token = (html.escape(q.get(k, "")) for k in ("job", "action", "token"))
    if not _valid(job, action, token):
        return _resp(403, "<h3>Invalid or expired link</h3>", HTML)
    page = f"""<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"></head>
<body style="font-family:sans-serif;padding:24px"><h2>MechAgent: ยืนยันการตัดสินใจ</h2>
<p>Job: <b>{job}</b><br>Action: <b>{ACTIONS[action]}</b></p>
<form method="POST" action="/approve"><input type="hidden" name="job" value="{job}">
<input type="hidden" name="action" value="{action}"><input type="hidden" name="token" value="{token}">
<button type="submit" style="font-size:18px;padding:10px 28px">ยืนยัน</button></form></body></html>"""
    return _resp(200, page, HTML)


def approve_submit(event):
    q = {k: v[0] for k, v in urllib.parse.parse_qs(_body(event)).items()}
    job, action = q.get("job", ""), q.get("action", "")
    if not _valid(job, action, q.get("token", "")):
        return _resp(403, "<h3>Invalid request</h3>", HTML)
    update_job(job, status=ACTIONS[action], decision=action, decided_at=datetime.now(BKK).isoformat())
    audit(job, f"human_{action}")
    if action == "approve":
        item = from_ddb(table().get_item(Key={"pk": f"JOB#{job}", "sk": "REPORT"}).get("Item") or {})
        plan = item.get("plan", {})
        notify(f"MechAgent work order {job}",
               f"ใบสั่งงาน (Work Order): {item.get('machine_id')}\nงาน: {plan.get('recommended_action')}\n"
               f"เริ่ม: {plan.get('planned_start')}\nอะไหล่: {plan.get('parts_plan')}\n\n{plan.get('summary_th', '')}")
    return _resp(200, f"<h2>บันทึกแล้ว: {ACTIONS[action]}</h2><p>Job {html.escape(job)}</p>", HTML)
