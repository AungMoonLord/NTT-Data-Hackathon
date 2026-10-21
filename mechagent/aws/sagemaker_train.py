"""Launch SageMaker training (HF DLC) and deploy a Serverless (or real-time) endpoint."""
import argparse
from pathlib import Path

import sagemaker
from sagemaker.huggingface import HuggingFace, HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

# If the SDK rejects a version combo, it prints the supported list → adjust here.
TRAIN_VERS = dict(transformers_version="4.36.0", pytorch_version="2.1.0", py_version="py310")
INFER_VERS = dict(transformers_version="4.37.0", pytorch_version="2.1.0", py_version="py310")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="mechagent")
    p.add_argument("--instance_type", default="ml.g4dn.xlarge")
    p.add_argument("--spot", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--backbone", default="ntu-spml/distilhubert")
    p.add_argument("--endpoint_name", default="mechagent-endpoint")
    p.add_argument("--model_data", default="", help="s3://.../model.tar.gz → skip training, deploy only")
    p.add_argument("--memory_mb", type=int, default=4096)
    p.add_argument("--realtime_instance", default="", help="e.g. ml.m5.large (instead of serverless)")
    a = p.parse_args()

    sess = sagemaker.Session(default_bucket=a.bucket)
    model_data = a.model_data
    if not model_data:
        est = HuggingFace(
            entry_point="train.py", source_dir=str(Path(__file__).resolve().parents[1] / "src"),
            role=a.role, instance_type=a.instance_type, instance_count=1, base_job_name="mechagent-mtl",
            hyperparameters={"epochs": a.epochs, "batch_size": a.batch_size, "backbone": a.backbone, "num_workers": 2},
            output_path=f"s3://{a.bucket}/{a.prefix}/models", sagemaker_session=sess,
            use_spot_instances=a.spot, max_run=4 * 3600, max_wait=6 * 3600 if a.spot else None,
            metric_definitions=[{"Name": "val:f1", "Regex": r"val_f1=([0-9\.]+)"},
                                {"Name": "val:deg_mae", "Regex": r"val_deg_mae=([0-9\.]+)"},
                                {"Name": "val:rul_mae_h", "Regex": r"val_rul_mae_h=([0-9\.]+)"}],
            **TRAIN_VERS)
        est.fit({"train": f"s3://{a.bucket}/{a.prefix}/data/"})
        model_data = est.model_data
        print("Model artifact:", model_data)

    model = HuggingFaceModel(model_data=model_data, role=a.role, sagemaker_session=sess, **INFER_VERS)
    if a.realtime_instance:
        model.deploy(initial_instance_count=1, instance_type=a.realtime_instance, endpoint_name=a.endpoint_name)
    else:
        model.deploy(serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=a.memory_mb, max_concurrency=5),
                     endpoint_name=a.endpoint_name)
    print("Endpoint ready:", a.endpoint_name)
