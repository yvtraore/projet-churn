import pickle
import base64
from pathlib import Path

import numpy as np
import streamlit as st

# =====================================================
# 0. CONFIG GLOBALE & CHARGEMENT DU MODELE
# =====================================================

st.set_page_config(
    page_title="Churn Télécom – Guinée",
    page_icon="📡",
    layout="wide",
)

BASE_DIR   = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
IMG_DIR    = BASE_DIR / "img"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

REGION_MAP = {
    "Conakry": 0, "Kankan": 1, "Labé": 2,
    "N'Zérékoré": 3, "Boké": 4, "Faranah": 5,
}
SEXE_MAP           = {"M": 0, "F": 1}
ABO_MAP            = {"Prépayé": 0, "Postpayé": 1}
OUI_NON_MAP        = {"Non": 0, "Oui": 1}
MOYEN_PAIEMENT_MAP = {
    "Mobile Money": 0, "Carte Bancaire": 1,
    "Virement": 2, "Cash": 3,
}

# =====================================================
# 1. CSS (ancien style conservé + enrichissements about)
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
[data-testid="stAppViewContainer"] { background: #f0f2f7; }
[data-testid="stHeader"]           { background: transparent; }
section[data-testid="stMain"] > div { padding-top: 0.6rem; }

/* ── HEADER ── */
.full-header {
    background: linear-gradient(120deg, #0d1b3e 0%, #1a3a6b 55%, #1565c0 100%);
    border-radius: 16px;
    margin-bottom: 18px;
    overflow: hidden;
    display: flex;
    align-items: stretch;
    box-shadow: 0 6px 24px rgba(13,27,62,0.22);
    min-height: 110px;
}
.fh-logo {
    background: rgba(255,255,255,0.08);
    border-right: 1px solid rgba(255,255,255,0.10);
    padding: 16px 20px;
    display: flex; align-items: center; justify-content: center;
    min-width: 120px; max-width: 160px; flex-shrink: 0;
}
.fh-logo img {
    max-height: 70px; width: auto; object-fit: contain;
    border-radius: 6px;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.30));
}
.fh-text {
    padding: 18px 26px;
    display: flex; flex-direction: column; justify-content: center; flex: 1;
}
.fh-text h1 { color: #ffffff; font-size: 21px; font-weight: 700; margin: 0 0 5px; }
.fh-text p  { color: rgba(255,255,255,0.55); font-size: 13px; margin: 0 0 10px; }
.fh-badges  { display: flex; gap: 7px; flex-wrap: wrap; }
.fh-badge {
    display: inline-block;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    color: #90caf9; font-size: 10px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
}
.fh-badge.gold {
    background: rgba(251,191,36,0.14); border-color: rgba(251,191,36,0.28); color: #fbbf24;
}
.fh-stats {
    padding: 18px 24px;
    display: flex; flex-direction: column; justify-content: center; gap: 10px;
    border-left: 1px solid rgba(255,255,255,0.09); min-width: 200px;
}
.fh-stat-row { display: flex; align-items: center; gap: 10px; }
.fh-stat-icon {
    font-size: 14px; width: 28px; height: 28px;
    background: rgba(255,255,255,0.08); border-radius: 6px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.fh-stat-label { font-size: 10px; color: rgba(255,255,255,0.40); }
.fh-stat-value { font-size: 13px; font-weight: 600; color: #ffffff; }

/* ── BARRE CONTEXTE ── */
.ctx-bar { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
.ctx-card {
    flex: 1; min-width: 100px;
    background: linear-gradient(135deg, #0d1b3e 0%, #1565c0 100%);
    border-radius: 12px; padding: 14px 16px;
    display: flex; align-items: center; gap: 12px;
    box-shadow: 0 3px 12px rgba(13,27,62,0.18);
}
.ctx-icon {
    font-size: 20px; width: 38px; height: 38px;
    background: rgba(255,255,255,0.10); border-radius: 8px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.ctx-val { font-size: 20px; font-weight: 700; color: #ffffff; line-height: 1.1; }
.ctx-lbl { font-size: 10px; color: rgba(255,255,255,0.50); margin-top: 2px; font-weight: 500; }

/* ── TABS ── */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important; color: #6b7280 !important;
    border-radius: 8px !important; padding: 7px 18px !important;
    border: 1px solid #d1d5db !important; background: #ffffff !important; margin-right: 6px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #1565c0 !important; border-color: #1565c0 !important;
    color: #ffffff !important; font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(21,101,192,0.28) !important;
}
[data-baseweb="tab-list"] { border-bottom: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ── SECTION LABEL ── */
.section-label {
    font-size: 11px; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
    color: #6b7280; border-left: 3px solid #1565c0; padding-left: 9px;
    margin-bottom: 14px; margin-top: 4px;
}

/* ── WIDGETS ── */
label[data-testid="stWidgetLabel"] p {
    font-size: 12px !important; font-weight: 500 !important; color: #374151 !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {
    border-radius: 8px !important; border: 1px solid #d1d5db !important;
    background: #f9fafb !important; font-size: 13px !important; color: #111827 !important;
}
div[data-baseweb="select"] > div {
    border-radius: 8px !important; border: 1px solid #d1d5db !important;
    background: #f9fafb !important; font-size: 13px !important; color: #111827 !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #1565c0 !important; border-color: #1565c0 !important;
}

/* ── BOUTON ── */
div[data-testid="stButton"] > button {
    border-radius: 999px !important;
    background: linear-gradient(90deg, #1565c0, #1976d2) !important;
    color: #ffffff !important; border: none !important;
    padding: 0.65rem 2rem !important; font-size: 14px !important;
    font-weight: 600 !important; font-family: 'Inter', sans-serif !important;
    box-shadow: 0 4px 14px rgba(21,101,192,0.28) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(21,101,192,0.42) !important;
}

/* ── RESULT ── */
.result-wrapper {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 14px; padding: 22px 26px; margin-top: 16px;
    display: flex; gap: 22px; align-items: flex-start; flex-wrap: wrap;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.prob-block {
    background: linear-gradient(140deg, #0d1b3e, #1a3a6b);
    border-radius: 14px; padding: 18px 24px; text-align: center;
    min-width: 145px; box-shadow: 0 4px 14px rgba(13,27,62,0.18);
}
.prob-number      { font-size: 42px; font-weight: 800; line-height: 1.1; }
.prob-number.high { color: #f87171; }
.prob-number.low  { color: #34d399; }
.prob-sub   { font-size: 11px; color: rgba(255,255,255,0.48); margin-top: 4px; }
.gauge-bar  { height: 6px; background: rgba(255,255,255,0.12); border-radius: 3px; overflow: hidden; margin-top: 10px; }
.gauge-inner { height: 100%; background: linear-gradient(90deg, #22c55e 0%, #f59e0b 55%, #ef4444 100%); border-radius: 3px; }
.info-block { flex: 1; min-width: 210px; }
.risk-pill  { display: inline-block; border-radius: 10px; padding: 9px 16px; font-size: 14px; font-weight: 700; margin-bottom: 7px; }
.risk-pill.high { background: rgba(239,68,68,0.08);  border: 1px solid rgba(239,68,68,0.22);  color: #dc2626; }
.risk-pill.low  { background: rgba(34,197,94,0.08);  border: 1px solid rgba(34,197,94,0.22);  color: #16a34a; }
.risk-caption   { font-size: 11px; color: #9ca3af; margin-bottom: 14px; }
.mini-metrics   { display: flex; gap: 10px; flex-wrap: wrap; }
.mini-card {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 10px 14px; text-align: center; min-width: 75px;
}
.mini-val { font-size: 17px; font-weight: 700; color: #1565c0; }
.mini-lbl { font-size: 10px; color: #9ca3af; margin-top: 2px; font-weight: 500; }

/* ── ABOUT ── */
.about-item {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-left: 3px solid #1565c0; border-radius: 0 10px 10px 0;
    padding: 12px 16px; margin-bottom: 10px;
    font-size: 13px; color: #374151;
}
.about-item strong { color: #1565c0; }

/* Pipeline */
.flow-row { display: flex; align-items: center; gap: 0; margin-bottom: 20px; flex-wrap: wrap; }
.flow-step {
    flex: 1; min-width: 90px; background: #ffffff;
    border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px 12px;
    text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.flow-step-icon  { font-size: 22px; margin-bottom: 5px; }
.flow-step-title { font-size: 11px; font-weight: 700; color: #0d1b3e; margin-bottom: 3px; }
.flow-step-desc  { font-size: 10px; color: #6b7280; line-height: 1.4; }
.flow-arrow      { font-size: 16px; color: #1565c0; padding: 0 5px; flex-shrink: 0; }

/* Arbre */
.rf-visual {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 14px; padding: 18px 20px; margin-bottom: 18px;
}
.rf-title {
    font-size: 11px; font-weight: 700; color: #0d1b3e;
    margin-bottom: 14px; text-align: center;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.tree-box {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 7px 11px;
    font-size: 11px; color: #374151; font-weight: 500;
    text-align: center; flex: 1; min-width: 55px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.tree-box.root { background: linear-gradient(135deg, #0d1b3e, #1a3a6b); color: white; border-color: transparent; font-weight: 700; }
.tree-box.leaf-yes { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.25); color: #dc2626; font-weight: 700; }
.tree-box.leaf-no  { background: rgba(34,197,94,0.08);  border-color: rgba(34,197,94,0.25);  color: #16a34a; font-weight: 700; }

/* Feature cards */
.feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 18px; }
.feature-card {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 13px 15px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    border-top: 3px solid transparent;
}
.feature-card.c1 { border-top-color: #ef4444; }
.feature-card.c2 { border-top-color: #f59e0b; }
.feature-card.c3 { border-top-color: #1565c0; }
.feature-card.c4 { border-top-color: #10b981; }
.feature-card.c5 { border-top-color: #8b5cf6; }
.feature-card.c6 { border-top-color: #ec4899; }
.feature-card-icon { font-size: 18px; margin-bottom: 5px; }
.feature-card-name { font-size: 12px; font-weight: 700; color: #0d1b3e; margin-bottom: 3px; }
.feature-card-desc { font-size: 11px; color: #6b7280; line-height: 1.5; }
.feature-impact {
    display: inline-block; font-size: 9px; font-weight: 700;
    letter-spacing: 0.07em; padding: 2px 7px; border-radius: 8px;
    margin-top: 5px; text-transform: uppercase;
}
.impact-high { background: rgba(239,68,68,0.10); color: #dc2626; }
.impact-med  { background: rgba(245,158,11,0.10); color: #d97706; }
.impact-low  { background: rgba(16,185,129,0.10); color: #059669; }

/* Métriques */
.metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }
.metric-card {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 14px; text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.metric-val  { font-size: 24px; font-weight: 800; color: #1565c0; }
.metric-name { font-size: 10px; color: #9ca3af; margin-top: 2px; font-weight: 500; }
.metric-bar  { height: 4px; border-radius: 2px; background: #e5e7eb; margin-top: 8px; overflow: hidden; }
.metric-bar-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #1565c0, #1976d2); }

img { border-radius: 10px !important; }
[data-testid="stCaptionContainer"] p { color: #9ca3af !important; font-size: 11px !important; }
hr  { border-color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 2. HEADER PLEINE LARGEUR
# =====================================================

logo_path = IMG_DIR / "logo_app.jpg"
if logo_path.exists():
    with open(str(logo_path), "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}" alt="Logo EC2LT"/>'
else:
    logo_html = '<span style="font-size:38px;">📡</span>'

st.markdown(f"""
<div class="full-header">
    <div class="fh-logo">{logo_html}</div>
    <div class="fh-text">
        <h1>Churn Télécom – Guinée</h1>
        <p>Outil de prédiction du risque de résiliation client basé sur un modèle Random Forest,<br>entraîné sur des données simulées de clients télécom en Guinée.</p>
        <div class="fh-badges">
            <span class="fh-badge">🤖 Random Forest</span>
            <span class="fh-badge">📊 Data Mining</span>
            <span class="fh-badge gold">🇬🇳 Guinée</span>
            <span class="fh-badge">EC2LT</span>
        </div>
    </div>
    <div class="fh-stats">
        <div class="fh-stat-row">
            <div class="fh-stat-icon">🗺️</div>
            <div>
                <div class="fh-stat-label">Régions couvertes</div>
                <div class="fh-stat-value">6 régions</div>
            </div>
        </div>
        <div class="fh-stat-row">
            <div class="fh-stat-icon">🌳</div>
            <div>
                <div class="fh-stat-label">Algorithme</div>
                <div class="fh-stat-value">Random Forest</div>
            </div>
        </div>
        <div class="fh-stat-row">
            <div class="fh-stat-icon">🎯</div>
            <div>
                <div class="fh-stat-label">Seuil de décision</div>
                <div class="fh-stat-value">0.50</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_model = st.tabs(["🔮 Prédiction client", "📊 À propos du modèle"])


# =====================================================
# 3. ONGLET PREDICTION
# =====================================================

with tab_predict:

    st.markdown("""
<div class="ctx-bar">
    <div class="ctx-card">
        <div class="ctx-icon">🌍</div>
        <div><div class="ctx-val">6</div><div class="ctx-lbl">Régions de Guinée</div></div>
    </div>
    <div class="ctx-card">
        <div class="ctx-icon">📋</div>
        <div><div class="ctx-val">18</div><div class="ctx-lbl">Variables d'entrée</div></div>
    </div>
    <div class="ctx-card">
        <div class="ctx-icon">🌳</div>
        <div><div class="ctx-val">RF</div><div class="ctx-lbl">Random Forest</div></div>
    </div>
    <div class="ctx-card">
        <div class="ctx-icon">🎯</div>
        <div><div class="ctx-val">0.50</div><div class="ctx-lbl">Seuil de décision</div></div>
    </div>
    <div class="ctx-card">
        <div class="ctx-icon">💳</div>
        <div><div class="ctx-val">4</div><div class="ctx-lbl">Moyens de paiement</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">🧑‍💼 Informations client</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        region = st.selectbox("Région", list(REGION_MAP.keys()))
        sexe   = st.selectbox("Sexe", ["M", "F"])
        age    = st.slider("Âge", 18, 80, 32)
    with c2:
        revenu = st.number_input(
            "Revenu estimé (GNF)",
            min_value=0, max_value=5_000_000, value=1_500_000, step=50_000,
        )
        anciennete      = st.slider("Ancienneté (mois)", 1, 120, 24)
        type_abonnement = st.selectbox("Type d'abonnement", ["Prépayé", "Postpayé"])
        moyen_paiement  = st.selectbox("Moyen de paiement", list(MOYEN_PAIEMENT_MAP.keys()))
    with c3:
        forfait_international   = st.selectbox("Forfait international", ["Oui", "Non"])
        messagerie_vocale       = st.selectbox("Messagerie vocale", ["Oui", "Non"])
        minutes_internationales = st.number_input(
            "Minutes internationales / mois",
            min_value=0.0, max_value=500.0, value=5.0, step=1.0, format="%.2f",
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📱 Usage & comportement</div>', unsafe_allow_html=True)

    u1, u2, u3 = st.columns(3)
    with u1:
        recharge_mensuelle = st.number_input(
            "Recharge mensuelle moyenne (GNF)",
            min_value=0, max_value=2_000_000, value=200_000, step=50_000,
        )
        minutes_jour = st.number_input(
            "Minutes en journée / mois", min_value=0.0, max_value=10_000.0, value=0.0
        )
    with u2:
        minutes_nuit = st.number_input(
            "Minutes de nuit / mois", min_value=0.0, max_value=10_000.0, value=0.0
        )
        donnees_mo = st.number_input(
            "Données Internet (Mo / mois)", min_value=0.0, max_value=100_000.0, value=0.0
        )
    with u3:
        nombre_sms            = st.number_input("Nombre de SMS / mois", min_value=0, max_value=10_000, value=0)
        appels_service_client = st.number_input("Appels au service client (30 jours)", min_value=0, max_value=100, value=0)

    c4, c5, c6 = st.columns(3)
    with c4:
        pannes_signalees_30j  = st.number_input("Pannes signalées (30 jours)", min_value=0, max_value=60, value=0)
    with c5:
        retard_paiement_jours = st.number_input("Retard de paiement (jours)", min_value=0, max_value=90, value=0)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        predict_btn = st.button("Lancer la prédiction 🔍", use_container_width=True)

    if predict_btn:
        x_vec = np.array([
            REGION_MAP[region], SEXE_MAP[sexe], age, revenu, anciennete,
            ABO_MAP[type_abonnement], OUI_NON_MAP[forfait_international],
            OUI_NON_MAP[messagerie_vocale], recharge_mensuelle,
            MOYEN_PAIEMENT_MAP[moyen_paiement], minutes_jour, minutes_nuit,
            minutes_internationales, donnees_mo, nombre_sms,
            appels_service_client, pannes_signalees_30j, retard_paiement_jours,
        ], dtype=float).reshape(1, -1)

        proba_churn = float(model.predict_proba(x_vec)[0, 1])
        pred        = int(proba_churn >= 0.5)
        risk_label  = "Client à risque de résiliation" if pred == 1 else "Client plutôt fidèle"
        risk_class  = "high" if pred == 1 else "low"
        risk_icon   = "🔴" if pred == 1 else "🟢"
        gauge_width = int(proba_churn * 100)

        st.markdown(f"""
<div class="result-wrapper">
    <div class="prob-block">
        <div class="prob-number {risk_class}">{proba_churn*100:.1f}%</div>
        <div class="prob-sub">Probabilité de churn</div>
        <div class="gauge-bar">
            <div class="gauge-inner" style="width:{gauge_width}%"></div>
        </div>
    </div>
    <div class="info-block">
        <div class="risk-pill {risk_class}">{risk_icon} {risk_label}</div>
        <div class="risk-caption">
            Seuil : 0.50 &nbsp;·&nbsp; Score : {proba_churn:.4f} &nbsp;·&nbsp; Classe prédite : <strong>{pred}</strong>
        </div>
        <div class="mini-metrics">
            <div class="mini-card">
                <div class="mini-val">{int(appels_service_client)}</div>
                <div class="mini-lbl">Appels SAV</div>
            </div>
            <div class="mini-card">
                <div class="mini-val">{int(pannes_signalees_30j)}</div>
                <div class="mini-lbl">Pannes</div>
            </div>
            <div class="mini-card">
                <div class="mini-val">{int(retard_paiement_jours)}j</div>
                <div class="mini-lbl">Retard pmt</div>
            </div>
            <div class="mini-card">
                <div class="mini-val" style="font-size:12px">{type_abonnement}</div>
                <div class="mini-lbl">Type abo.</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.caption(
            "Interprétation : **1 = client susceptible de résilier**, "
            "**0 = client fidèle**. Seuil de décision fixé à 0.5."
        )


# =====================================================
# 4. ONGLET A PROPOS DU MODELE
# =====================================================

with tab_model:

    # ── Pipeline ──
    st.markdown('<div class="section-label">⚙️ Pipeline de la prédiction</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="flow-row">
  <div class="flow-step">
    <div class="flow-step-icon">🧑‍💼</div>
    <div class="flow-step-title">Données client</div>
    <div class="flow-step-desc">18 variables saisies (usage, profil, comportement)</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-step">
    <div class="flow-step-icon">🔧</div>
    <div class="flow-step-title">Encodage</div>
    <div class="flow-step-desc">Variables catégorielles converties en valeurs numériques</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-step">
    <div class="flow-step-icon">🌳</div>
    <div class="flow-step-title">Forêt d'arbres</div>
    <div class="flow-step-desc">Chaque arbre analyse les données et émet un vote (0 ou 1)</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-step">
    <div class="flow-step-icon">🗳️</div>
    <div class="flow-step-title">Vote majoritaire</div>
    <div class="flow-step-desc">Agrégation des votes → probabilité de churn calculée</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-step">
    <div class="flow-step-icon">🎯</div>
    <div class="flow-step-title">Décision finale</div>
    <div class="flow-step-desc">Si probabilité ≥ 0.50 → client à risque de résiliation</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Arbre de décision ──
    st.markdown('<div class="section-label">🌳 Exemple d\'arbre de décision interne</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="rf-visual">
  <div class="rf-title">Visualisation simplifiée d'un arbre parmi N dans la forêt</div>
  <div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
    <div style="display:flex; justify-content:center;">
      <div class="tree-box root" style="max-width:210px;">Appels SAV &gt; 3 ?</div>
    </div>
    <div style="display:flex; justify-content:center; gap:130px; color:#9ca3af; font-size:11px;">
      <span>✓ OUI</span><span>✗ NON</span>
    </div>
    <div style="display:flex; justify-content:center; gap:30px; width:100%;">
      <div style="display:flex; flex-direction:column; align-items:center; gap:4px; flex:1;">
        <div class="tree-box" style="max-width:170px;">Retard paiement &gt; 10j ?</div>
        <div style="display:flex; gap:50px; color:#9ca3af; font-size:10px;"><span>OUI</span><span>NON</span></div>
        <div style="display:flex; gap:8px;">
          <div class="tree-box leaf-yes">🔴 CHURN</div>
          <div class="tree-box" style="max-width:130px;">Pannes &gt; 2 ?</div>
        </div>
        <div style="display:flex; justify-content:flex-end; gap:30px; color:#9ca3af; font-size:10px; width:100%; padding-right:8px;"><span>OUI</span><span>NON</span></div>
        <div style="display:flex; justify-content:flex-end; gap:8px; width:100%;">
          <div class="tree-box leaf-yes">🔴 CHURN</div>
          <div class="tree-box leaf-no">🟢 FIDÈLE</div>
        </div>
      </div>
      <div style="display:flex; flex-direction:column; align-items:center; gap:4px; flex:1;">
        <div class="tree-box" style="max-width:170px;">Ancienneté &lt; 6 mois ?</div>
        <div style="display:flex; gap:40px; color:#9ca3af; font-size:10px;"><span>OUI</span><span>NON</span></div>
        <div style="display:flex; gap:8px;">
          <div class="tree-box leaf-yes">🔴 CHURN</div>
          <div class="tree-box leaf-no">🟢 FIDÈLE</div>
        </div>
      </div>
    </div>
  </div>
  <div style="text-align:center; margin-top:12px; font-size:11px; color:#9ca3af;">
    💡 Le Random Forest combine des <strong>dizaines à centaines</strong> d'arbres comme celui-ci. La décision finale = vote majoritaire.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Variables influentes ──
    st.markdown('<div class="section-label">📊 Variables les plus influentes sur le churn</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="feature-grid">
  <div class="feature-card c1">
    <div class="feature-card-icon">📞</div>
    <div class="feature-card-name">Appels service client</div>
    <div class="feature-card-desc">Un client qui contacte souvent le SAV exprime une insatisfaction croissante — fort signal de désengagement.</div>
    <span class="feature-impact impact-high">Impact fort</span>
  </div>
  <div class="feature-card c2">
    <div class="feature-card-icon">💸</div>
    <div class="feature-card-name">Retard de paiement</div>
    <div class="feature-card-desc">Un retard prolongé signale un désengagement financier ou une intention imminente de quitter l'opérateur.</div>
    <span class="feature-impact impact-high">Impact fort</span>
  </div>
  <div class="feature-card c3">
    <div class="feature-card-icon">🔧</div>
    <div class="feature-card-name">Pannes signalées</div>
    <div class="feature-card-desc">De nombreuses pannes témoignent d'une mauvaise expérience réseau, facteur clé de migration vers un concurrent.</div>
    <span class="feature-impact impact-high">Impact fort</span>
  </div>
  <div class="feature-card c4">
    <div class="feature-card-icon">📅</div>
    <div class="feature-card-name">Ancienneté</div>
    <div class="feature-card-desc">Les nouveaux clients (moins de 6 mois) sont plus volatils. La fidélité se renforce naturellement avec le temps.</div>
    <span class="feature-impact impact-med">Impact moyen</span>
  </div>
  <div class="feature-card c5">
    <div class="feature-card-icon">📶</div>
    <div class="feature-card-name">Recharge mensuelle</div>
    <div class="feature-card-desc">Une baisse soudaine du montant rechargé peut indiquer une utilisation réduite et un départ imminent.</div>
    <span class="feature-impact impact-med">Impact moyen</span>
  </div>
  <div class="feature-card c6">
    <div class="feature-card-icon">📱</div>
    <div class="feature-card-name">Usage data & voix</div>
    <div class="feature-card-desc">Un usage très faible ou en forte baisse peut signaler que le client utilise une autre SIM en parallèle.</div>
    <span class="feature-impact impact-low">Impact modéré</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Métriques ──
    st.markdown('<div class="section-label">📈 Performances indicatives du modèle</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="metrics-row">
  <div class="metric-card">
    <div class="metric-val">~85%</div>
    <div class="metric-name">Accuracy</div>
    <div class="metric-bar"><div class="metric-bar-fill" style="width:85%"></div></div>
  </div>
  <div class="metric-card">
    <div class="metric-val">~82%</div>
    <div class="metric-name">Précision</div>
    <div class="metric-bar"><div class="metric-bar-fill" style="width:82%"></div></div>
  </div>
  <div class="metric-card">
    <div class="metric-val">~80%</div>
    <div class="metric-name">Rappel</div>
    <div class="metric-bar"><div class="metric-bar-fill" style="width:80%"></div></div>
  </div>
  <div class="metric-card">
    <div class="metric-val">~81%</div>
    <div class="metric-name">F1-Score</div>
    <div class="metric-bar"><div class="metric-bar-fill" style="width:81%"></div></div>
  </div>
</div>
<div style="font-size:11px; color:#9ca3af; margin-bottom:18px;">
  ⚠️ Métriques indicatives basées sur des données simulées. Un déploiement en production nécessite une validation sur données réelles.
</div>
""", unsafe_allow_html=True)

    # ── Images réduites (4 colonnes) ──
    st.markdown('<div class="section-label">🖼️ Illustrations</div>', unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        rf_img = IMG_DIR / "random_forest.png"
        if rf_img.exists():
            st.image(str(rf_img), caption="Architecture Random Forest", use_container_width=True)
    with col_b:
        fi_img = IMG_DIR / "data_processing.jpg"
        if fi_img.exists():
            st.image(str(fi_img), caption="Pipeline traitement des données", use_container_width=True)
    with col_c:
        map_img = IMG_DIR / "guinee_map.png"
        if map_img.exists():
            st.image(str(map_img), caption="Couverture géographique — Guinée", use_container_width=True)
    with col_d:
        ml_img = IMG_DIR / "ml_models.webp"
        if ml_img.exists():
            st.image(str(ml_img), caption="Familles de modèles ML", use_container_width=True)

    # ── Récapitulatif ──
    st.markdown('<div class="section-label">📋 Récapitulatif technique</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="about-item"><strong>Algorithme :</strong> Random Forest Classifier (scikit-learn) — ensemble de N arbres de décision CART entraînés indépendamment</div>
<div class="about-item"><strong>Objectif :</strong> Classification binaire · Prédire si un client va <em>résilier (churn = 1)</em> ou <em>rester fidèle (churn = 0)</em></div>
<div class="about-item"><strong>Variables clés :</strong> appels SAV, pannes signalées, retard de paiement, ancienneté, recharge mensuelle, usage voix & données internet</div>
<div class="about-item"><strong>Données :</strong> Dataset simulé représentant des clients télécom en Guinée · 6 régions couvertes · encodage label pour les variables catégorielles</div>
<div class="about-item"><strong>Seuil de décision :</strong> 0.50 · Si P(churn) ≥ 0.50 alors classe = 1 (risque de résiliation détecté)</div>
<div class="about-item"><strong>Avantages du Random Forest :</strong> robuste au surapprentissage, gère les variables mixtes, fournit les importances de features, ne nécessite pas de normalisation</div>
""", unsafe_allow_html=True)