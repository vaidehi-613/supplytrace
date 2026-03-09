import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11"

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import json

builder = (SparkSession.builder
    .appName("SupplyTrace-Dashboard")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "8")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark ready — loading Gold tables...\n")

df_market   = spark.read.format("delta").load("data/gold/market_dashboard")
df_supplier = spark.read.format("delta").load("data/gold/supplier_leaderboard")
df_route    = spark.read.format("delta").load("data/gold/route_risk")
df_kpi      = spark.read.format("delta").load("data/gold/kpi_summary")

kpis = {row["metric"]: row["value"] for row in df_kpi.collect()}

markets = [row.asDict() for row in
    df_market.select("market","total_orders","delay_rate_pct",
                     "avg_delay_days","total_sales_usd","avg_lpi_score")
    .orderBy("delay_rate_pct", ascending=False).collect()]

best_suppliers = [row.asDict() for row in
    df_supplier.orderBy("delay_rate_pct").limit(10).collect()]

worst_suppliers = [row.asDict() for row in
    df_supplier.orderBy("delay_rate_pct", ascending=False).limit(10).collect()]

routes = [row.asDict() for row in
    df_route.filter("shipping_mode != 'First Class'")
    .orderBy("delay_rate_pct", ascending=False).limit(10).collect()]

print("✅ Data loaded — generating dashboard HTML...")
spark.stop()

# ── Pre-compute all dynamic values BEFORE the HTML f-string ──────
total_orders   = f"{int(kpis.get('total_orders','0')):,}"
delayed_orders = f"{int(kpis.get('delayed_orders','0')):,}"
delay_rate     = kpis.get('delay_rate_pct','57.3')
total_sales_m  = f"${float(kpis.get('total_sales_usd','0'))/1e6:.1f}M"
model_auc      = kpis.get('model_auc','0.9751')
model_acc      = f"{float(kpis.get('model_accuracy','0.9744'))*100:.1f}%"
model_f1       = kpis.get('model_f1','0.9743')
news_risk      = kpis.get('news_risk_level','MEDIUM')

market_labels  = json.dumps([r['market'] for r in markets])
market_data    = json.dumps([float(r['delay_rate_pct']) for r in markets])

def sales_fmt(val):
    return f"${float(str(val).replace('E','e'))/1e6:.2f}M"

def lpi_width(val):
    return f"{float(val)/5*100:.0f}%"

def delay_pill(val):
    cls = "pill-yellow" if float(val) > 57 else "pill-green"
    lbl = "HIGH" if float(val) > 57 else "MEDIUM"
    return f'<span class="pill {cls}">{lbl}</span>'

market_rows = ""
for r in markets:
    market_rows += f"""<tr>
        <td><b>{r['market']}</b></td>
        <td>{int(r['total_orders']):,}</td>
        <td><div>{r['delay_rate_pct']}%</div>
            <div class="metric-bar"><div class="metric-fill"
            style="width:{r['delay_rate_pct']}%;background:#f87171"></div></div></td>
        <td>{r['avg_delay_days']}</td>
        <td>{sales_fmt(r['total_sales_usd'])}</td>
        <td><div>{r['avg_lpi_score']}</div>
            <div class="metric-bar"><div class="metric-fill"
            style="width:{lpi_width(r['avg_lpi_score'])}"></div></div></td>
        <td>{delay_pill(r['delay_rate_pct'])}</td>
    </tr>"""

best_rows = ""
for r in best_suppliers:
    best_rows += f"""<tr>
        <td><b>{r['supplier_name']}</b></td>
        <td><span class="pill pill-blue">T{r['supplier_tier']}</span></td>
        <td><span class="pill pill-green">{r['delay_rate_pct']}%</span></td>
        <td>{r['quality_score']}</td>
    </tr>"""

worst_rows = ""
for r in worst_suppliers:
    worst_rows += f"""<tr>
        <td><b>{r['supplier_name']}</b></td>
        <td><span class="pill pill-blue">T{r['supplier_tier']}</span></td>
        <td><span class="pill pill-red">{r['delay_rate_pct']}%</span></td>
        <td>{r['quality_score']}</td>
    </tr>"""

route_rows = ""
for r in routes:
    route_rows += f"""<tr>
        <td><b>{r['order_country']}</b></td>
        <td style="color:#94a3b8;font-size:12px">{r['order_region']}</td>
        <td>{r['shipping_mode']}</td>
        <td>{r['order_count']}</td>
        <td><div>{r['delay_rate_pct']}%</div>
            <div class="metric-bar"><div class="metric-fill"
            style="width:{r['delay_rate_pct']}%;background:#fb923c"></div></div></td>
        <td>{r['avg_lpi_score']}</td>
    </tr>"""

# ── HTML (no dynamic values inside — all pre-computed above) ──────
html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SupplyTrace Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'Segoe UI', sans-serif; background:#0f172a; color:#e2e8f0; }
  header { background:linear-gradient(135deg,#1e3a5f,#0ea5e9); padding:24px 32px; }
  header h1 { font-size:28px; font-weight:700; }
  header p  { font-size:14px; opacity:0.85; margin-top:4px; }
  .badge { display:inline-block; background:rgba(255,255,255,0.2);
           border-radius:999px; padding:3px 12px; font-size:12px; margin-left:10px; }
  .container { max-width:1400px; margin:0 auto; padding:28px 24px; }
  .kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
              gap:16px; margin-bottom:28px; }
  .kpi-card { background:#1e293b; border-radius:12px; padding:20px;
              border-left:4px solid #0ea5e9; }
  .kpi-card .label { font-size:11px; text-transform:uppercase;
                     letter-spacing:1px; color:#94a3b8; }
  .kpi-card .value { font-size:26px; font-weight:700; margin-top:6px; color:#f1f5f9; }
  .kpi-card .sub   { font-size:12px; color:#64748b; margin-top:4px; }
  .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:28px; }
  .card { background:#1e293b; border-radius:12px; padding:20px; }
  .card h3 { font-size:14px; font-weight:600; color:#94a3b8;
             text-transform:uppercase; letter-spacing:1px; margin-bottom:16px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { text-align:left; padding:8px 10px; color:#64748b;
       font-size:11px; text-transform:uppercase; letter-spacing:0.5px;
       border-bottom:1px solid #334155; }
  td { padding:9px 10px; border-bottom:1px solid #1e293b; }
  tr:hover td { background:#263548; }
  .pill { display:inline-block; border-radius:999px; padding:2px 10px; font-size:11px; }
  .pill-green  { background:#064e3b; color:#34d399; }
  .pill-yellow { background:#451a03; color:#fbbf24; }
  .pill-red    { background:#450a0a; color:#f87171; }
  .pill-blue   { background:#0c1a3a; color:#60a5fa; }
  .metric-bar  { height:6px; background:#334155; border-radius:3px; margin-top:4px; }
  .metric-fill { height:100%; border-radius:3px; background:#0ea5e9; }
  canvas { max-height:260px; }
  @media(max-width:768px) { .grid-2 { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>⚡ SupplyTrace <span class="badge">Live Dashboard</span></h1>
  <p>AI-Powered Supply Chain Disruption Intelligence &nbsp;|&nbsp;
     PySpark + Delta Lake + Spark MLlib &nbsp;|&nbsp;
     AUC-ROC: """ + model_auc + """ &nbsp;|&nbsp; News Risk: """ + news_risk + """</p>
</header>

<div class="container">
  <div class="kpi-grid">
    <div class="kpi-card" style="border-color:#0ea5e9">
      <div class="label">Total Orders</div>
      <div class="value">""" + total_orders + """</div>
      <div class="sub">Across 5 global markets</div>
    </div>
    <div class="kpi-card" style="border-color:#f87171">
      <div class="label">Delayed Orders</div>
      <div class="value">""" + delay_rate + """%</div>
      <div class="sub">""" + delayed_orders + """ orders late</div>
    </div>
    <div class="kpi-card" style="border-color:#34d399">
      <div class="label">Total Sales</div>
      <div class="value">""" + total_sales_m + """</div>
      <div class="sub">USD across all regions</div>
    </div>
    <div class="kpi-card" style="border-color:#a78bfa">
      <div class="label">Model AUC-ROC</div>
      <div class="value">""" + model_auc + """</div>
      <div class="sub">GBT Classifier</div>
    </div>
    <div class="kpi-card" style="border-color:#fbbf24">
      <div class="label">Model Accuracy</div>
      <div class="value">""" + model_acc + """</div>
      <div class="sub">F1: """ + model_f1 + """</div>
    </div>
    <div class="kpi-card" style="border-color:#fb923c">
      <div class="label">News Risk Level</div>
      <div class="value">""" + news_risk + """</div>
      <div class="sub">720 articles analyzed</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>📊 Delay Rate by Market</h3>
      <canvas id="marketChart"></canvas>
    </div>
    <div class="card">
      <h3>🚢 Delay Rate by Shipping Mode</h3>
      <canvas id="shippingChart"></canvas>
    </div>
  </div>

  <div class="card" style="margin-bottom:20px">
    <h3>🌍 Market Performance Dashboard</h3>
    <table>
      <tr><th>Market</th><th>Orders</th><th>Delay Rate</th>
          <th>Avg Delay Days</th><th>Total Sales</th><th>Avg LPI</th><th>Risk</th></tr>
      """ + market_rows + """
    </table>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>🏆 Top 10 Best Suppliers</h3>
      <table>
        <tr><th>Supplier</th><th>Tier</th><th>Delay %</th><th>Quality</th></tr>
        """ + best_rows + """
      </table>
    </div>
    <div class="card">
      <h3>⚠️ Top 10 Worst Suppliers</h3>
      <table>
        <tr><th>Supplier</th><th>Tier</th><th>Delay %</th><th>Quality</th></tr>
        """ + worst_rows + """
      </table>
    </div>
  </div>

  <div class="card" style="margin-bottom:28px">
    <h3>🗺️ Riskiest Routes (excluding First Class)</h3>
    <table>
      <tr><th>Country</th><th>Region</th><th>Mode</th>
          <th>Orders</th><th>Delay Rate</th><th>Avg LPI</th></tr>
      """ + route_rows + """
    </table>
  </div>

  <div style="text-align:center;color:#475569;font-size:12px;padding:20px 0">
    SupplyTrace &nbsp;|&nbsp; PySpark 3.4.1 + Delta Lake 2.4.0 + Spark MLlib
    &nbsp;|&nbsp; Vaidehi Pawar, SDSU MS CS 2026
  </div>
</div>

<script>
new Chart(document.getElementById('marketChart'), {
  type: 'bar',
  data: {
    labels: """ + market_labels + """,
    datasets: [{
      label: 'Delay Rate %',
      data: """ + market_data + """,
      backgroundColor: ['#f87171','#fb923c','#fbbf24','#34d399','#60a5fa'],
      borderRadius: 6,
    }]
  },
  options: {
    plugins: { legend: { display:false } },
    scales: {
      y: { beginAtZero:true, max:100, grid:{color:'#334155'},
           ticks:{color:'#94a3b8'} },
      x: { grid:{display:false}, ticks:{color:'#94a3b8'} }
    }
  }
});

new Chart(document.getElementById('shippingChart'), {
  type: 'doughnut',
  data: {
    labels: ['First Class (100%)', 'Second Class (79.7%)', 'Same Day (47.8%)', 'Standard (39.8%)'],
    datasets: [{
      data: [100, 79.7, 47.8, 39.8],
      backgroundColor: ['#f87171','#fb923c','#fbbf24','#34d399'],
      borderWidth: 0,
    }]
  },
  options: {
    plugins: {
      legend: { position:'bottom', labels:{color:'#94a3b8',font:{size:11}} }
    }
  }
});
</script>
</body>
</html>"""

os.makedirs("docs", exist_ok=True)
with open("docs/dashboard.html", "w") as f:
    f.write(html)

print("✅ Dashboard saved to: docs/dashboard.html")
print("   Opening in browser...")
