🚀 AI-Driven Self-Healing Kubernetes Platform

📌 Overview:
Built a closed-loop AIOps system that automatically detects anomalies using machine learning and triggers self-healing actions in Kubernetes.

NOTE: This app platform is based on 3 GitHub Repositories. /aiops-gitops  and /k8s-Argo-Helm-app

🧠 Architecture:
Prometheus → ML Anomaly Detection → AlertManager → AI Decision Engine → Kubernetes API → Auto-Healing

⚙️ Tech Stack:
Kubernetes
Prometheus + AlertManager
Grafana
FastAPI
Scikit-learn (Isolation Forest)
ArgoCD (GitOps)
Docker + GitHub Actions (CI/CD)

🔥 Key Features:
📊 Real-time metric collection via Prometheus
🤖 Anomaly detection using Isolation Forest
🚨 Alerting pipeline via AlertManager
🧠 AI decision engine (threshold + logic-based actions)
🔁 Automated remediation:
Pod restart
Horizontal scaling
🔄 GitOps-based deployment with ArgoCD

🧪 Demo Scenario:
Simulated CPU spike in application
Anomaly score increased
Alert triggered via Prometheus
AI service evaluated anomaly
Kubernetes deployment automatically restarted/scaled
System recovered without manual intervention

📊 Observability:
Visualized in Grafana:

Anomaly score trends
CPU usage spikes
Deployment scaling/restarts

🎯 Outcome:
Implemented a self-healing system that reduces manual intervention and demonstrates real-world AIOps principles.
