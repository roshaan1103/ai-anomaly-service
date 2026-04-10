from fastapi import FastAPI
import requests
import numpy as np
from sklearn.ensemble import IsolationForest
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import threading
from kubernetes import client, config

app = FastAPI()

PROMETHEUS_URL = "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090"

# Prometheus metrics
anomaly_score_metric = Gauge("aiops_anomaly_score", "Normalized anomaly score (0-1)")
anomaly_flag_metric = Gauge("aiops_anomaly_detected", "Anomaly detected (1 = yes, 0 = no)")

model = IsolationForest(contamination=0.1)

#GLOBAL VARIABLE
latest_score = 0.0
latest_prediction = 0


def fetch_cpu_metrics():
    query = 'rate(container_cpu_usage_seconds_total[1m])'
    url = f"{PROMETHEUS_URL}/api/v1/query"

    response = requests.get(url, params={"query": query})
    data = response.json()

    values = []
    for result in data["data"]["result"]:
        val = float(result["value"][1])
        values.append(val)

    return np.array(values).reshape(-1, 1)


@app.get("/")
def root():
    return {"message": "AIOps Anomaly Detection Service Running"}


@app.get("/analyze")
def analyze():
    global latest_score, latest_prediction

    try:
        data = fetch_cpu_metrics()

        if len(data) < 5:
            return {"status": "not enough data"}

        model.fit(data)

        scores = model.decision_function(data)   # negative normal, closer to 0 anomaly
        predictions = model.predict(data)        # -1 = anomaly, 1 = normal

        raw_score = scores[-1]
        pred = predictions[-1]

        #NORMALIZATION FIX
        normalized_score = 1 + raw_score  # convert range roughly to 0

        # clamp safety
        normalized_score = max(0.0, min(1.0, normalized_score))

        # STORE GLOBAL STATE
        latest_score = float(normalized_score)
        latest_prediction = int(pred)

        # EXPORT METRICS
        anomaly_score_metric.set(latest_score)
        anomaly_flag_metric.set(1 if pred == -1 else 0)

        return {
            "raw_score": float(raw_score),
            "normalized_score": latest_score,
            "anomaly": int(pred == -1)
        }

    except Exception as e:
        return {"error": str(e)}


def background_job():
    while True:
        try:
            analyze()
        except Exception as e:
            print(f"Background error: {e}")
        time.sleep(30)


threading.Thread(target=background_job, daemon=True).start()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/alert")
async def handle_alert(data: dict):
    global latest_score, latest_prediction

    try:
        config.load_incluster_config()
        apps_v1 = client.AppsV1Api()

        print(f"Latest Score: {latest_score}")
        print(f"Latest Prediction: {latest_prediction}")

        # DECISION ENGINE
        if latest_prediction == -1 or latest_score > 0.8:
            action = "restart"
        elif latest_score > 0.6:
            action = "scale"
        else:
            action = "ignore"

        print(f"Chosen action: {action}")

        # EXECUTION
        if action == "restart":
            patch = {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": str(time.time())
                            }
                        }
                    }
                }
            }

            apps_v1.patch_namespaced_deployment(
                name="aiops-app",
                namespace="default",
                body=patch
            )

        elif action == "scale":
            body = {"spec": {"replicas": 3}}

            apps_v1.patch_namespaced_deployment_scale(
                name="aiops-app",
                namespace="default",
                body=body
            )

        return {
            "status": "done",
            "score": latest_score,
            "prediction": latest_prediction,
            "action": action
        }

    except Exception as e:
        return {"error": str(e)}
