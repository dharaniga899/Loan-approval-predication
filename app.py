# ============================================================
#  LOAN APPROVAL PREDICTION — Hugging Face Spaces app.py
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

import gradio as gr

print("Loading dataset...")

# NOTE: filename must match exactly what you uploaded to the Space.
# Your Space currently has: "loan approval datasets.csv"
df = pd.read_csv("loan approval datasets.csv")
print("Dataset loaded:", df.shape)

# ── Preprocessing ──────────────────────────────────────────
df['loan_approved'] = df['loan_approved'].apply(lambda x: 1 if x == True else 0)
df = df.drop(['name', 'city', 'points'], axis=1)

np.random.seed(42)
for col in ['income', 'credit_score', 'loan_amount', 'years_employed']:
    df[col] = df[col] + np.random.normal(0, df[col].std() * 0.30, len(df))

# ── Train/Test Split ───────────────────────────────────────
X = df.drop('loan_approved', axis=1)
y = df['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── SMOTE Balancing ────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ── Scaling (kept for parity, RF doesn't need it) ──────────
scaler = StandardScaler()
scaler.fit(X_train_bal)

# ── Train Random Forest (best model) ───────────────────────
best_model = RandomForestClassifier(
    n_estimators=50, max_depth=4,
    min_samples_split=40, min_samples_leaf=20,
    max_features='sqrt', n_jobs=-1, random_state=42
)
best_model.fit(X_train_bal, y_train_bal)

acc = best_model.score(X_test, y_test)
roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"Model trained. Accuracy: {acc:.4f}  ROC-AUC: {roc:.4f}")

# ── Gradio Frontend ─────────────────────────────────────────
custom_css = """
body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    color: #E8E8F0 !important;
    font-family: 'Inter', sans-serif !important;
}
.gr-box, .gr-panel, .gr-card, .block {
    background-color: rgba(30, 27, 75, 0.85) !important;
    border: 1px solid #5B4FBE !important;
    border-radius: 12px !important;
}
label, .gr-form-label, .svelte-1gfkn6j {
    color: #A78BFA !important;
    font-weight: 600 !important;
    font-size: 0.88em !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
input[type=range] { accent-color: #7C3AED; }
.gr-button-primary {
    background: linear-gradient(90deg, #7C3AED, #4F46E5) !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
.gr-button-primary:hover {
    background: linear-gradient(90deg, #6D28D9, #4338CA) !important;
}
.gradio-slider input[type=range] { accent-color: #8B5CF6; }
"""

def predict_loan(income, credit_score, loan_amount, years_employed):
    sample = pd.DataFrame({
        'income'        : [income],
        'credit_score'  : [credit_score],
        'loan_amount'   : [loan_amount],
        'years_employed': [years_employed]
    })

    proba      = best_model.predict_proba(sample)[0][1]
    prediction = best_model.predict(sample)[0]

    if prediction == 1:
        badge_color = "#00C896"
        badge_bg    = "rgba(0,200,150,0.12)"
        badge_text  = "✅ APPROVED"
        badge_sub   = "Loan application looks strong"
    else:
        badge_color = "#FF5C7A"
        badge_bg    = "rgba(255,92,122,0.12)"
        badge_text  = "❌ NOT APPROVED"
        badge_sub   = "Application needs improvement"

    bar_pct   = int(proba * 100)
    bar_color = "#00C896" if proba >= 0.5 else "#FF5C7A"

    report_html = f"""
    <div style='font-family:Inter,sans-serif; padding:4px;'>
      <div style='
        background:{badge_bg};
        border:2px solid {badge_color};
        border-radius:14px;
        padding:20px 24px;
        margin-bottom:18px;
        text-align:center;
      '>
        <div style='font-size:2em; font-weight:900; color:{badge_color}; letter-spacing:0.05em;'>
          {badge_text}
        </div>
        <div style='color:#B0A8D8; font-size:0.9em; margin-top:6px;'>{badge_sub}</div>
      </div>

      <div style='
        background:rgba(255,255,255,0.05);
        border-radius:10px;
        padding:16px 20px;
        margin-bottom:16px;
      '>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
          <span style='color:#A78BFA; font-size:0.8em; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;'>Approval Probability</span>
          <span style='color:#E8E8F0; font-weight:800; font-size:1.1em;'>{bar_pct}%</span>
        </div>
        <div style='background:rgba(255,255,255,0.1); border-radius:999px; height:10px; overflow:hidden;'>
          <div style='
            width:{bar_pct}%;
            height:100%;
            background:linear-gradient(90deg, {bar_color}, {"#7C3AED" if prediction==1 else "#FF8FAB"});
            border-radius:999px;
            transition: width 0.5s ease;
          '></div>
        </div>
      </div>

      <div style='
        background:rgba(255,255,255,0.04);
        border-radius:10px;
        padding:16px 20px;
      '>
        <div style='color:#A78BFA; font-size:0.78em; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;'>📋 Application Summary</div>
        <table style='width:100%; color:#E8E8F0; font-size:0.9em; border-collapse:collapse;'>
          <tr>
            <td style='padding:5px 0; color:#9D8FCC;'>Annual Income</td>
            <td style='padding:5px 0; text-align:right; font-weight:700;'>₹{income:,.0f}</td>
          </tr>
          <tr>
            <td style='padding:5px 0; color:#9D8FCC;'>Credit Score</td>
            <td style='padding:5px 0; text-align:right; font-weight:700;'>{credit_score:.0f}</td>
          </tr>
          <tr>
            <td style='padding:5px 0; color:#9D8FCC;'>Loan Amount</td>
            <td style='padding:5px 0; text-align:right; font-weight:700;'>₹{loan_amount:,.0f}</td>
          </tr>
          <tr>
            <td style='padding:5px 0; color:#9D8FCC;'>Years Employed</td>
            <td style='padding:5px 0; text-align:right; font-weight:700;'>{years_employed:.0f} yrs</td>
          </tr>
        </table>
      </div>
    </div>
    """

    recs = []
    if credit_score < 500:
        recs.append(("🔴", "Critical Credit Score", "Score is very low. Immediate steps needed to rebuild credit history and payment records."))
    elif credit_score < 650:
        recs.append(("🟡", "Below-Average Credit", "Score under 650 raises risk flags. Aim to cross 700+ before reapplying."))
    else:
        recs.append(("🟢", "Healthy Credit Score", "Strong score above 650. This is a major positive for your application."))

    if years_employed < 2:
        recs.append(("🟡", "Low Employment History", "Less than 2 years employed. Lenders prefer stable income track records of 2+ years."))
    else:
        recs.append(("🟢", "Stable Employment", "Good work history detected. This adds credibility to your repayment capacity."))

    if loan_amount > income * 0.5:
        recs.append(("🔴", "High Loan-to-Income Ratio", f"Loan of ₹{loan_amount:,.0f} vs income of ₹{income:,.0f} looks risky. Consider requesting a lower amount."))
    else:
        recs.append(("🟢", "Reasonable Loan Amount", "Loan amount is within acceptable range relative to your annual income."))

    if proba >= 0.75:
        recs.append(("🟢", "Strong Profile", "Very high approval likelihood. Maintain your current financial profile."))
    elif proba >= 0.5:
        recs.append(("🟡", "Moderate Profile", "Borderline approval. Small improvements in credit or employment could push this over."))
    else:
        recs.append(("🔴", "Weak Profile", "Approval unlikely at current metrics. Focus on credit score and income before reapplying."))

    rec_rows = ""
    for icon, title, desc in recs:
        rec_rows += f"""
        <div style='
          display:flex;
          gap:12px;
          padding:12px 0;
          border-bottom:1px solid rgba(255,255,255,0.06);
        '>
          <div style='font-size:1.3em; line-height:1;'>{icon}</div>
          <div>
            <div style='color:#E8E8F0; font-weight:700; font-size:0.9em;'>{title}</div>
            <div style='color:#9D8FCC; font-size:0.82em; margin-top:3px; line-height:1.4;'>{desc}</div>
          </div>
        </div>
        """

    recs_html = f"""
    <div style='font-family:Inter,sans-serif; padding:4px;'>
      <div style='
        background:rgba(255,255,255,0.04);
        border-radius:10px;
        padding:16px 20px;
      '>
        <div style='color:#A78BFA; font-size:0.78em; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>
          💡 Smart Recommendations
        </div>
        {rec_rows}
      </div>
    </div>
    """

    return report_html, recs_html


# ── Build UI ──────────────────────────────────────────────
with gr.Blocks(css=custom_css, title="Loan Approval Predictor") as demo:

    gr.HTML("""
    <div style='text-align:center; padding:24px 0 8px;'>
      <div style='font-size:2.2em; font-weight:900; color:#A78BFA; letter-spacing:-0.02em;'>
        🏦 Loan Approval Predictor
      </div>
      <div style='color:#7C6FB0; font-size:0.95em; margin-top:8px;'>
        Random Forest · Trained on your dataset · Live prediction
      </div>
    </div>
    """)

    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            gr.HTML("<div style='color:#A78BFA; font-weight:700; font-size:0.8em; text-transform:uppercase; letter-spacing:0.1em; padding:4px 0 12px;'>📥 Applicant Details</div>")

            income_sl = gr.Slider(minimum=10000, maximum=500000, value=85000, step=1000, label="Annual Income (₹)")
            credit_sl = gr.Slider(minimum=300, maximum=850, value=680, step=5, label="Credit Score (300 – 850)")
            loan_sl   = gr.Slider(minimum=1000, maximum=300000, value=20000, step=1000, label="Loan Amount (₹)")
            emp_sl    = gr.Slider(minimum=0, maximum=40, value=5, step=1, label="Years Employed")

            predict_btn = gr.Button("🔍  Predict Now", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_result = gr.HTML(label="Decision")
            output_recs   = gr.HTML(label="Recommendations")

    inputs  = [income_sl, credit_sl, loan_sl, emp_sl]
    outputs = [output_result, output_recs]

    for sl in inputs:
        sl.change(fn=predict_loan, inputs=inputs, outputs=outputs)

    predict_btn.click(fn=predict_loan, inputs=inputs, outputs=outputs)
    demo.load(fn=predict_loan, inputs=inputs, outputs=outputs)

# ── Launch (HF Spaces handles hosting automatically) ───────
demo.launch()
