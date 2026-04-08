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

# Prometheus metrics we expose
anomaly_score = Gauge("aiops_anomaly_score", "Anomaly score")
anomaly_flag = Gauge("aiops_anomaly_detected", "Anomaly detected (1 = yes, 0 = no)")

model = IsolationForest(contamination=0.1)

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


def get_anomaly_score():
    try:
        query = "aiops_anomaly_score"
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query}
        )

        data = response.json()

        result = data["data"]["result"]

        if not result:
            return 0.0

        value = float(result[0]["value"][1])
        return value

    except Exception as e:
        print(f"Error fetching anomaly score: {e}")
        return 0.0


@app.get("/")
def root():
    return {"message": "AIOps Anomaly Detection Service Running"}


@app.get("/analyze")
def analyze():
    try:
        data = fetch_cpu_metrics()

        if len(data) < 5:
            return {"status": "not enough data"}

        model.fit(data)
        scores = model.decision_function(data)
        predictions = model.predict(data)

        latest_score = scores[-1]
        latest_pred = predictions[-1]

        anomaly_score.set(float(latest_score))
        anomaly_flag.set(1 if latest_pred == -1 else 0)

        return {
            "score": float(latest_score),
            "anomaly": int(latest_pred == -1)
        }

    except Exception as e:
        return {"error": str(e)}

def background_job():
    while True:
        try:
            analyze()
        except:
            pass
        time.sleep(30)

threading.Thread(target=background_job, daemon=True).start()

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/alert")
async def handle_alert(data: dict):
    try:
        config.load_incluster_config()
        apps_v1 = client.AppsV1Api()

        #GET REAL-TIME ANOMALY SCORE
        anomaly_score = get_anomaly_score()
        print(f"Anomaly Score: {anomaly_score}")

        #DECISION ENGINE (THIS IS THE CORE)
        if anomaly_score > 0.8:
            action = "restart"
        elif anomaly_score > 0.5:
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
                                "kubectl.kubernetes.io/restartedAt": "now"
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
            "score": anomaly_score,
            "action": action
        }

    except Exception as e:
        return {"error": str(e)}
