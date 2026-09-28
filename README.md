# Pharma MLOps Pipeline — Local Setup

<!-- brag:start -->
<p align="center">
  <a href="brag-output/brag.mp4"><img src="brag-output/brag.gif" alt="Pharma MLOps Pipeline launch video" width="100%"></a>
  <br>
  <sub>▶ <a href="brag-output/brag.mp4"><b>Watch the full launch video with voice-over</b></a> (43s, sound on)</sub>
</p>
<!-- brag:end -->


## Architecture Overview
```
Data Ingestion → Feature Engineering → Model Training → Validation → Registry → Serving → Monitoring
     ↑                                                                                        |
     └──────────────────── Retrain on Drift ──────────────────────────────────────────────────┘
```

## Quick Start (run these in order)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start MLflow UI (reads local ./mlruns, run in separate terminal)
mlflow ui --host 127.0.0.1 --port 5000

# 3. Run full pipeline
python pipelines/run_pipeline.py

# 4. Start model serving API (run in separate terminal)
python serving/serve.py

# 5. Start monitoring dashboard (run in separate terminal)
python monitoring/monitor.py

# 6. Launch UI dashboard
python ui/dashboard.py
```

## System Components
| Component | Tool | Port/Path |
|-----------|------|-----------|
| Experiment Tracking | MLflow | http://localhost:5000 |
| Model Serving | FastAPI | http://localhost:8000 |
| Monitoring | Evidently + FastAPI | http://localhost:8001 |
| UI Dashboard | Streamlit | http://localhost:8501 |
| Audit Logs | Local JSON | ./audit/ |
| Feature Store | Local SQLite | ./data/features/ |
