# MechAgent: Agentic AI for Bearing Predictive Maintenance

> ระบบ AI Agent ที่ "ฟัง" แรงสั่นสะเทือนของตลับลูกปืน แล้วบอกได้ว่า **เสียตรงไหน, เสื่อมไปกี่ %, ต้องเปลี่ยนภายในกี่วัน กี่ชั่วโมง กี่นาที**
> จากนั้นวางแผนซ่อม เช็คอะไหล่ และตารางผลิตเอง แล้วส่งให้หัวหน้างานกด Approve โดยทั้งหมดรันบน AWS แบบ Serverless

## 1. Project Name
**MechAgent: Physics-Informed Multitask AI Agent for Bearing Predictive Maintenance on AWS**

## 2. Use Case / Pain Point
ตลับลูกปืน (bearing) เป็นสาเหตุอันดับต้นๆ ที่ทำให้มอเตอร์และปั๊มในโรงงานเสีย ปัญหาที่พบในปัจจุบันคือ
- **ตรวจตามรอบเวลา (เดือนละครั้ง):** ความเสียหายที่เกิดระหว่างรอบไม่มีใครเห็น ทำให้เกิด Unplanned Downtime
- **วิเคราะห์ต้องใช้ผู้เชี่ยวชาญ:** การอ่าน spectrum และ envelope analysis ต้องใช้ vibration analyst ซึ่งหายากและแพง
- **รายงานช้า:** หัวหน้างานใช้เวลา 2–4 ชั่วโมงรวบรวมข้อมูล เช็คอะไหล่ และตารางผลิต ก่อนตัดสินใจ
- **ตอบได้ไม่ครบ:** ระบบทั่วไปบอกได้แค่ "ผิดปกติ" แต่ไม่บอกว่าเสียตรงไหน และเหลือเวลาอีกเท่าไร

## 3. AWS Solution
MechAgent แบ่งเป็น 9 ขั้นตอน ขั้นที่ 1–6 เป็นการคำนวณ (Deterministic + ML) ส่วน LLM ใช้เฉพาะขั้นที่ 7 เพื่อลดค่าใช้จ่ายและป้องกัน hallucination

| # | Step | เทคโนโลยี | AWS |
|---|---|---|---|
| 1 | Signal Simulator (physics-based run-to-failure) | NumPy/SciPy | S3 |
| 2 | Preprocessing: band-pass 1–5 kHz → Hilbert envelope → Order spectrum | SciPy | SageMaker Endpoint |
| 3 | Multitask AI: DistilHuBERT (HF) + Physics branch → Fault / Degradation % / RUL ± uncertainty | PyTorch + Transformers | SageMaker Training + Serverless Inference |
| 4 | Physics Verification: ตรวจ peak ที่ BPFI/BPFO/BSF/FTF | SciPy | SageMaker Endpoint |
| 5 | RUL: model RUL (P5/P50/P95, MC-Dropout) + exponential trend จากประวัติ | PyTorch / Python | Lambda + DynamoDB |
| 6 | Gatekeeper: rule-based เรียก LLM เฉพาะเมื่อจำเป็น + dedup | Python | Lambda |
| 7 | Agent: วางแผน, เรียก 7 tools, RAG คู่มือ, Guardrail เส้นตาย | Bedrock Converse Tool Use | Bedrock + Lambda |
| 8 | Human-in-the-loop: Approve / Postpone / Reject → Work Order | HMAC-signed links | SNS (Email) / Telegram |
| 9 | Dashboard + Audit Log | Streamlit | API Gateway + DynamoDB |

**จุดเด่นทางเทคนิค**
- **Multitask Learning:** 1 backbone และ 3 heads (classification + 2 regression) ใช้ Uncertainty Weighting (Kendall et al.) และ Gaussian NLL สำหรับ RUL
- **Physics-Informed:** ใช้ order domain จึงไม่ขึ้นกับความเร็วรอบ และทดสอบ zero-shot กับตลับลูกปืนรุ่นที่ไม่เคยเห็น (6208/6310)
- **Safety-first:** เส้นตายใช้ P5 (worst case) และ Agent ถูก guardrail บังคับให้วางแผนก่อนเส้นตาย
- **Cost-aware:** Gatekeeper ทำให้ LLM ถูกเรียกเฉพาะ incident จริง ประมาณ 1 บาทต่อ incident

## 4. AWS Architecture Diagram & Services Used

```text
                        ┌──────────────────────── TRAINING (offline) ────────────────────────┐
                        │ simulate.py ─► S3 data/ ─► SageMaker Training Job (GPU Spot)        │
                        │                              fine-tune DistilHuBERT-MTL            │
                        │                              └─► S3 model.tar.gz ─► SageMaker       │
                        │                                   Serverless Endpoint              │
                        └────────────────────────────────────────────────────────────────────┘

 Vibration Sensor /     HTTPS POST      ┌───────────────────┐   async   ┌──────────────────────────┐
 Gateway / Dashboard ─────────────────► │ Amazon API Gateway│─────────► │ AWS Lambda (Orchestrator)│
        ▲                               │   (HTTP API)      │           │ Steps 4-8                │
        │ GET /reports/{id}             └───────────────────┘           └──┬──────┬──────┬──────┬──┘
        │                                                                  │      │      │      │
        │            ┌─────────────────────────────┐   invoke_endpoint    │      │      │      │
        │            │ SageMaker Serverless         │◄────────────────────┘      │      │      │
        │            │ Fault / Deg% / RUL + Physics │                             │      │      │
        │            └─────────────────────────────┘                              │      │      │
        │            ┌─────────────────────────────┐   Converse + Tool Use        │      │      │
        │            │ Amazon Bedrock (Nova Pro)    │◄────────────────────────────┘      │      │
        │            └─────────────────────────────┘                                     │      │
        │            ┌─────────────────────────────┐   machines / readings / parts /    │      │
        └────────────│ Amazon DynamoDB              │◄── jobs / audit log ────────────────┘      │
                     └─────────────────────────────┘                                            │
                     ┌─────────────────────────────┐   approval links / work orders            │
 Foreman / Tech ◄────│ Amazon SNS (Email) / Telegram│◄──────────────────────────────────────────┘
                     └─────────────────────────────┘
 Amazon S3: raw signals (expire 7 days), datasets, model artifacts, maintenance manual (RAG)
 Amazon CloudWatch: Lambda/Endpoint logs, training metrics (val_f1, val_rul_mae_h)
```

| Service | หน้าที่ |
|---|---|
| Amazon SageMaker Training | Fine-tune DistilHuBERT แบบ Multitask บน GPU (รองรับ Spot) |
| Amazon SageMaker Serverless Inference | Host โมเดล จ่ายเฉพาะตอนเรียกใช้ |
| Amazon Bedrock | LLM Agent (Converse API + Tool Use) |
| AWS Lambda | Orchestrator, Gatekeeper, Agent tools, Approval |
| Amazon API Gateway (HTTP API) | REST endpoint สำหรับ sensor และ dashboard |
| Amazon DynamoDB | Machine profile, readings, spare parts, schedule, jobs, audit log |
| Amazon S3 | Dataset, model artifacts, raw signals, คู่มือซ่อม (RAG) |
| Amazon SNS | แจ้งเตือนอีเมลและใบสั่งงาน |
| Amazon CloudWatch | Logs และ metrics |
| AWS IAM | Least-privilege roles |

## 5. AWS Pricing Estimation (us-east-1, ราคาโดยประมาณ)
**สมมติฐาน:** โรงงาน 50 เครื่อง วัดทุก 10 นาที = 216,000 inferences/เดือน, เกิด incident 20 ครั้ง/เดือน

| Service | การใช้งาน | USD/เดือน |
|---|---|---|
| SageMaker Serverless Inference | 216k × ~0.5 s × 4 GB | ~9 |
| SageMaker Training | retrain เดือนละครั้ง, ml.g4dn.xlarge ~0.5 ชม. | ~0.4 (Spot ~0.15) |
| AWS Lambda (arm64) | ~430k invocations, ~1.5 s @ 512 MB | ~2.5 (free tier ช่วยได้) |
| API Gateway HTTP API | ~250k requests | ~0.3 |
| DynamoDB On-Demand | ~650k writes / 300k reads | ~1 |
| S3 | ~10 GB (raw หมดอายุใน 7 วัน) + PUT requests | ~1.5 |
| Amazon Bedrock (Nova Pro) | 20 incidents × ~30k input / 3k output tokens | ~0.7 |
| SNS + CloudWatch Logs | email + ~3 GB logs | ~2 |
| **รวม** | | **~USD 17–20/เดือน (~600–700 บาท)** |

- **ทางเลือก Real-time endpoint:** ml.m5.large 24/7 เพิ่มประมาณ USD 84/เดือน (latency ต่ำ ไม่มี cold start)
- **ค่า LLM ต่อ incident:** Nova Pro ประมาณ 1 บาท, Nova Lite ต่ำกว่า 0.1 บาท
- **AWS Pricing Calculator:** https://calculator.aws/#/ (ใส่ค่าตามตารางด้านบน แล้วกด *Share* เพื่อได้ลิงก์มาแนบ)
- ราคาอาจเปลี่ยนแปลง ควรตรวจสอบกับหน้า pricing ของแต่ละ service ก่อนส่ง

## 6. Before vs. After Improvement
| ด้าน | Before | After (MechAgent) |
|---|---|---|
| การตรวจสอบ | ตรวจด้วยมือเดือนละครั้ง | Monitor อัตโนมัติต่อเนื่อง ทุก 5–60 นาที (Agent ปรับเองได้) |
| การวิเคราะห์ | ต้องใช้ vibration analyst | AI + Physics บอกตำแหน่งเสีย, % เสื่อม, RUL |
| เวลาทำรายงานและแผนซ่อม | 2–4 ชั่วโมง | ประมาณ 2 นาที (รวมเช็คอะไหล่ ตารางผลิต และคู่มือ) |
| การบอกเวลาเปลี่ยน | "เร็วๆ นี้" | "ภายใน X วัน Y ชม. Z นาที" + ช่วงความไม่แน่นอน |
| Unplanned breakdown | ~6 ครั้ง/ปี | ~1–2 ครั้ง/ปี (เป้าหมาย) |
| การตัดสินใจ | กระจัดกระจาย ไม่มีบันทึก | Human-in-the-loop + Audit Log ทุก action |
| มูลค่า | - | ประหยัดได้ประมาณ 450,000 บาท/ปี (โรงงาน 50 เครื่อง) เทียบกับค่า AWS ประมาณ 8,000 บาท/ปี |

## Quick Start (Local)
```bash
pip install -r requirements.txt
python src/pipeline.py --quick      # smoke test ~5-10 นาทีบน CPU
python src/pipeline.py              # full pipeline (แนะนำ GPU)
```

## Limitations & Next Steps
- ข้อมูลเป็น synthetic ควร validate กับ dataset จริง (CWRU, Paderborn, XJTU-SY, IMS) ก่อนใช้งานจริง
- ต่อยอด: AWS IoT Core / Greengrass สำหรับ sensor จริง, Bedrock Knowledge Bases สำหรับ RAG, SageMaker Model Monitor