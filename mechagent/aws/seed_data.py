"""Seed demo machines, schedules, spare parts, history and the maintenance manual (RAG source)."""
import argparse
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import boto3

BKK = timezone(timedelta(hours=7))
MACHINES = [
    {"machine_id": "PUMP-01", "name": "ปั๊มน้ำหล่อเย็น #1", "line": "Utility", "bearing_model": "SKF 6205-2RS",
     "bearing_key": "6205", "criticality": "A", "standby_machine": "PUMP-02", "rated_rpm": 1800},
    {"machine_id": "MOTOR-07", "name": "มอเตอร์สายพานลำเลียง", "line": "Packing", "bearing_model": "SKF 6205-2RS",
     "bearing_key": "6205", "criticality": "B", "standby_machine": None, "rated_rpm": 1500},
    {"machine_id": "FAN-03", "name": "พัดลมดูดอากาศ", "line": "Paint Shop", "bearing_model": "SKF 6208",
     "bearing_key": "6208", "criticality": "B", "standby_machine": None, "rated_rpm": 2950},
]
PARTS = [
    {"bearing_model": "SKF 6205-2RS", "qty": 2, "location": "คลัง A-03-2", "lead_time_days": 3,
     "substitutes": ["NSK 6205DDU", "FAG 6205-2RSR"], "unit_price_thb": 450},
    {"bearing_model": "SKF 6208", "qty": 0, "location": "คลัง A-03-4", "lead_time_days": 5,
     "substitutes": ["NSK 6208DDU"], "unit_price_thb": 890},
]
MANUAL = """# คู่มือซ่อมบำรุงตลับลูกปืน (Bearing Maintenance Manual)

## ความปลอดภัยก่อนเริ่มงาน (LOTO)
tags: safety, replacement
1. แจ้งฝ่ายผลิต และสลับไปเครื่องสำรอง (ถ้ามี)
2. ตัดไฟ ล็อก และติดป้าย (Lockout/Tagout) จากนั้นทดสอบว่าไม่มีพลังงานค้าง
3. รอให้ชิ้นส่วนเย็นตัวต่ำกว่า 50°C

## ขั้นตอนเปลี่ยนตลับลูกปืนมาตรฐาน
tags: replacement, inner_race, outer_race, ball, cage
1. ถอดคัปปลิ้ง และทำเครื่องหมายตำแหน่ง alignment เดิม
2. ใช้ bearing puller ถอดตลับลูกปืน (ห้ามใช้ค้อนตีโดยตรง)
3. ตรวจเพลาและ housing: ค่า runout ไม่เกิน 0.02 mm และไม่มีรอยสึก
4. ให้ความร้อนตลับลูกปืนใหม่ด้วย induction heater ไม่เกิน 110°C แล้วสวมเข้า
5. อัดจาระบีตามสเปก (ประมาณ 30% ของช่องว่าง) แล้ว laser alignment ให้ได้ ≤ 0.05 mm
6. เดินเครื่องทดสอบ 30 นาที แล้ววัด vibration baseline ใหม่

## ความเสียหายที่แหวนรอบใน (Inner Race)
tags: inner_race
ตรวจ fit ระหว่างเพลากับตลับลูกปืน, ปริมาณจาระบี และโหลด ถ้าพบรอยสึกที่เพลา (fretting) ต้องซ่อมเพลาก่อนติดตั้ง

## ความเสียหายที่แหวนรอบนอก (Outer Race)
tags: outer_race
ตรวจความกลมของ housing, soft foot และ misalignment รวมทั้งตรวจซีลกันสิ่งปนเปื้อน

## ความเสียหายที่เม็ดลูกปืน (Ball)
tags: ball
ตรวจการปนเปื้อนในจาระบี และกระแสไฟฟ้ารั่ว (ติดตั้ง shaft grounding ring ถ้าใช้ VFD)

## ความเสียหายที่รังลูกปืน (Cage)
tags: cage
ตรวจระบบหล่อลื่นทันที เนื่องจาก cage แตกอาจทำให้เครื่องล็อกเฉียบพลัน ควรหยุดเครื่องโดยเร็ว

## การหล่อลื่น
tags: lubrication, inner_race, ball, cage
ใช้จาระบี lithium complex NLGI 2 ตามรอบที่กำหนด ห้ามเติมเกิน เพราะทำให้อุณหภูมิสูงขึ้น
"""


def d(o):
    return json.loads(json.dumps(o), parse_float=Decimal)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--region", default=None)
    a = p.parse_args()
    t = boto3.resource("dynamodb", region_name=a.region).Table(a.table)
    now = datetime.now(BKK).replace(minute=0, second=0, microsecond=0)
    for m in MACHINES:
        t.put_item(Item=d({"pk": f"MACHINE#{m['machine_id']}", "sk": "PROFILE", **m, "monitoring_interval_min": 60}))
        stops = [{"start": (now + timedelta(days=dd)).replace(hour=8).isoformat(),
                  "end": (now + timedelta(days=dd)).replace(hour=12).isoformat(),
                  "reason": "Planned PM / product changeover"} for dd in (2, 6, 13)]
        t.put_item(Item=d({"pk": f"MACHINE#{m['machine_id']}", "sk": "SCHEDULE", "planned_stops": stops}))
    for part in PARTS:
        t.put_item(Item=d({"pk": f"PART#{part['bearing_model']}", "sk": "STOCK", **part}))
    for i, deg in enumerate([12, 15, 19, 24, 31, 38]):   # rising history for PUMP-01 (trend demo)
        ts = (now - timedelta(hours=24 * (6 - i))).isoformat()
        t.put_item(Item=d({"pk": "MACHINE#PUMP-01", "sk": f"READING#{ts}", "deg_pct": deg,
                           "fault": "inner_race", "rul_p50": 300 - 40 * i}))
    boto3.client("s3", region_name=a.region).put_object(Bucket=a.bucket, Key="manuals/bearing_manual.md",
                                                        Body=MANUAL.encode("utf-8"))
    print("Seed data loaded.")
