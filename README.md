# ⚡ SupplyTrace
### AI-Powered Supply Chain Disruption Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PySpark](https://img.shields.io/badge/PySpark-3.4.1-orange?style=flat-square&logo=apache-spark)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-2.4.0-00B4D8?style=flat-square)
![MLlib](https://img.shields.io/badge/Spark_MLlib-GBT-8B5CF6?style=flat-square)
![AUC](https://img.shields.io/badge/AUC--ROC-0.9751-34D399?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)

> A production-grade big data pipeline that ingests 6 real-world data sources into a **Medallion Lakehouse Architecture** (Bronze → Silver → Gold), engineers supply chain risk features using PySpark, and trains a Gradient Boosted Trees classifier to predict delivery disruptions with **97.5% AUC-ROC**.

---

## 📸 Dashboard Preview

![SupplyTrace Dashboard - KPI Cards and Charts](docs/screenshots/dashboard_1.png)
![SupplyTrace Dashboard - Market and Supplier Tables](docs/screenshots/dashboard_2.png)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                            │
│  DataCo CSV │ World Bank LPI │ NOAA Storms │ FRED API │ News    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BRONZE LAYER (Delta Lake)                     │
│  Raw ingestion │ Schema enforcement │ Metadata columns           │
│  6 Delta tables │ Append-only │ ACID transactions                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SILVER LAYER (Delta Lake)                     │
│  PySpark joins │ Feature engineering │ Data quality              │
│  180,519 rows │ 8 risk features │ Partitioned by market          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GOLD LAYER (Delta Lake)                      │
│  GBT Risk Classifier │ KPI Aggregations │ Supplier Leaderboard   │
│  AUC 0.9751 │ Route Risk Rankings │ Interactive Dashboard        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Model AUC-ROC** | **0.9751** |
| **Model Accuracy** | **97.44%** |
| **F1 Score** | **0.9743** |
| Total Orders Analyzed | 180,519 |
| Overall Delay Rate | 57.3% |
| Total Pipeline Records | 326,000+ |
| Training Time | 13.9 seconds |
| News Articles Analyzed | 720 |
| Countries with LPI Scores | 61 |
| Storm Events (2022–2023) | 145,480 |

---

## 🗃️ Data Sources

| Source | Type | Records | Description |
|--------|------|---------|-------------|
| [DataCo Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) | CSV | 180,519 | Orders, shipping, customers across 5 global markets |
| [World Bank LPI 2023](https://lpi.worldbank.org/) | Synthetic | 61 countries | Logistics Performance Index scores per country |
| [NOAA Storm Events](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/) | CSV (gz) | 145,480 | Real US weather disruptions 2022–2023 |
| [Google News RSS](https://news.google.com/rss) | API | 720 articles | Live supply chain disruption headlines |
| [FRED API](https://fred.stlouisfed.org/) | API | 18,963 | Oil, steel, aluminium, gas commodity prices |
| [Python Faker](https://faker.readthedocs.io/) | Synthetic | 2,000 | Supplier master dimension table |

---

## 🔧 Feature Engineering

8 supply chain risk features engineered in the Silver layer:

| Feature | Description | Source |
|---------|-------------|--------|
| `delay_days` | Actual vs scheduled shipping days | DataCo |
| `is_delayed` | Binary delay flag | DataCo |
| `route_delay_rate` | Historical delay rate per country/region/mode | DataCo |
| `weather_risk_score` | Avg storm severity score per US state | NOAA |
| `supplier_risk_flag` | 1 if country LPI tier is High/Very High Risk | World Bank |
| `lpi_score` | Country logistics performance score (1–5) | World Bank |
| `commodity_shock_flag` | 1 if WTI oil price exceeds $85/barrel | FRED |
| `news_risk_level` | LOW/MEDIUM/HIGH based on NLP keyword scoring | Google News |

---

## 🌲 ML Model — Gradient Boosted Trees

```
Algorithm:     GBTClassifier (Spark MLlib)
Target:        late_delivery_risk (binary)
Features:      15 (including 8 engineered + shipping mode, market, etc.)
Train/Test:    80% / 20% split (seed=42)
Training set:  144,567 rows
Test set:      35,952 rows
```

**Confusion Matrix (test set):**
```
                Predicted: On-Time    Predicted: Late
Actual: On-Time      15,406               922
Actual: Late              0            19,624
```

**Accuracy by Market:**
| Market | Accuracy |
|--------|----------|
| Africa | 97.8% |
| USCA | 97.6% |
| Pacific Asia | 97.5% |
| Europe | 97.4% |
| LATAM | 97.3% |

---

## 📁 Project Structure

```
supplytrace/
├── src/
│   ├── ingestion/                  # Bronze layer ingestion scripts
│   │   ├── shipping_ingest.py      # DataCo supply chain CSV → Delta
│   │   ├── lpi_ingest.py           # World Bank LPI → Delta
│   │   ├── weather_ingest.py       # NOAA Storm Events → Delta
│   │   ├── news_ingest.py          # Google News RSS → Delta
│   │   ├── commodity_ingest.py     # FRED API → Delta
│   │   └── supplier_ingest.py      # Faker synthetic suppliers → Delta
│   ├── transforms/                 # Silver layer transforms
│   │   ├── silver_master.py        # Join all Bronze + feature engineering
│   │   └── fix_supplier_join.py    # Supplier dimension join fix
│   ├── ml/                         # Gold layer ML + KPIs
│   │   ├── risk_classifier.py      # GBT model training + evaluation
│   │   └── gold_kpis.py            # KPI aggregations + leaderboards
│   └── dashboard/
│       └── dashboard.py            # HTML dashboard generator
├── data/
│   ├── raw/                        # Source files (not committed)
│   ├── bronze/                     # Delta tables — raw layer
│   ├── silver/                     # Delta tables — enriched layer
│   └── gold/                       # Delta tables — ML + KPIs
├── docs/
│   └── dashboard.html              # Interactive dashboard
├── tests/
│   └── test_setup.py               # Environment verification
├── .env                            # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.11
- Java OpenJDK 11
- macOS or Linux

### Installation

```bash
# Clone the repo
git clone https://github.com/vaidehi-613/supplytrace.git
cd supplytrace

# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
echo "FRED_API_KEY=your_key_here" > .env
export JAVA_HOME=/opt/homebrew/opt/openjdk@11
```

### Get a Free FRED API Key
Register at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) — instant approval, no cost.

### Run the Full Pipeline

```bash
# Bronze — ingest all data sources
python src/ingestion/shipping_ingest.py
python src/ingestion/lpi_ingest.py
python src/ingestion/weather_ingest.py
python src/ingestion/news_ingest.py
python src/ingestion/commodity_ingest.py
python src/ingestion/supplier_ingest.py

# Silver — join and engineer features
python src/transforms/silver_master.py
python src/transforms/fix_supplier_join.py

# Gold — train ML model + build KPIs
python src/ml/risk_classifier.py
python src/ml/gold_kpis.py

# Dashboard — generate interactive HTML
python src/dashboard/dashboard.py
open docs/dashboard.html
```

---

## 📈 Key Insights

- **57.3% of all orders are delayed** globally across 5 markets
- **First Class shipping has a 100% delay rate** in this dataset — counterintuitively the riskiest mode
- **Europe has the highest delay rate** at 57.7%, followed by Pacific Asia at 57.3%
- **Africa has the lowest average LPI score** (2.88) — highest logistics risk region
- **Washington PLC** (Tier 1, East Asia) is the best performing supplier at just 1.1% delay rate
- **WTI crude oil at $71.13/barrel** (March 2026) — below the $85 shock threshold
- **News risk level: MEDIUM** — 720 headlines analyzed with NLP keyword scoring
- Model accuracy is **remarkably consistent** across all 5 global markets (97.3–97.8%)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Processing | Apache PySpark 3.4.1 |
| Storage Format | Delta Lake 2.4.0 (ACID, time travel) |
| ML Framework | Spark MLlib (GBTClassifier) |
| Language | Python 3.11 |
| Data Sources | FRED API, NOAA, Google News RSS |
| Synthetic Data | Python Faker 19.6.2 |
| Dashboard | Vanilla HTML/CSS/JS + Chart.js |
| Environment | Java OpenJDK 11, venv |

---

## 🗺️ Roadmap

- [ ] Migrate to Databricks Community Edition
- [ ] Add Databricks `ai_query()` for LLM-powered news risk summaries
- [ ] MLflow experiment tracking and model registry
- [ ] Unity Catalog data lineage
- [ ] Databricks Workflows DAG orchestration
- [ ] Real-time NOAA NWS API integration
- [ ] Streamlit interactive dashboard

---

## 👩‍💻 Author

**Vaidehi Pawar**
MS Computer Science — San Diego State University (GPA 3.86, May 2026)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/vaidehi-pawar)
[![GitHub](https://img.shields.io/badge/GitHub-vaidehi--613-181717?style=flat-square&logo=github)](https://github.com/vaidehi-613)

---

## 📄 License

This project is licensed under the MIT License.

---

*Built as a course project for Big Data Tools & Methods, SDSU CS Department*
