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
# 1. CSS
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
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 120px;
    max-width: 160px;
    flex-shrink: 0;
}
.fh-logo img {
    max-height: 70px;
    width: auto;
    object-fit: contain;
    border-radius: 6px;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.30));
}
.fh-text {
    padding: 18px 26px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    flex: 1;
}
.fh-text h1 { color: #ffffff; font-size: 21px; font-weight: 700; margin: 0 0 5px; }
.fh-text p  { color: rgba(255,255,255,0.55); font-size: 13px; margin: 0 0 10px; }
.fh-badges  { display: flex; gap: 7px; flex-wrap: wrap; }
.fh-badge {
    display: inline-block;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    color: #90caf9;
    font-size: 10px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
}
.fh-badge.gold {
    background: rgba(251,191,36,0.14);
    border-color: rgba(251,191,36,0.28);
    color: #fbbf24;
}
.fh-stats {
    padding: 18px 24px;
    display: flex; flex-direction: column;
    justify-content: center; gap: 10px;
    border-left: 1px solid rgba(255,255,255,0.09);
    min-width: 200px;
}
.fh-stat-row { display: flex; align-items: center; gap: 10px; }
.fh-stat-icon {
    font-size: 14px; width: 28px; height: 28px;
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.fh-stat-label { font-size: 10px; color: rgba(255,255,255,0.40); }
.fh-stat-value { font-size: 13px; font-weight: 600; color: #ffffff; }

/* ── BARRE CONTEXTE ── */
.ctx-bar {
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}
.ctx-card {
    flex: 1;
    min-width: 100px;
    background: linear-gradient(135deg, #0d1b3e 0%, #1565c0 100%);
    border-radius: 12px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 3px 12px rgba(13,27,62,0.18);
}
.ctx-icon {
    font-size: 20px;
    width: 38px; height: 38px;
    background: rgba(255,255,255,0.10);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.ctx-val  { font-size: 20px; font-weight: 700; color: #ffffff; line-height: 1.1; }
.ctx-lbl  { font-size: 10px; color: rgba(255,255,255,0.50); margin-top: 2px; font-weight: 500; }

/* ── TABS ── */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: #6b7280 !important;
    border-radius: 8px !important; padding: 7px 18px !important;
    border: 1px solid #d1d5db !important;
    background: #ffffff !important; margin-right: 6px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #1565c0 !important;
    border-color: #1565c0 !important;
    color: #ffffff !important; font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(21,101,192,0.28) !important;
}
[data-baseweb="tab-list"] { border-bottom: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ── SECTION LABEL ── */
.section-label {
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.09em; text-transform: uppercase;
    color: #6b7280;
    border-left: 3px solid #1565c0;
    padding-left: 9px;
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
    border-radius: 10px; padding: 10px 14px;
    text-align: center; min-width: 75px;
}
.mini-val { font-size: 17px; font-weight: 700; color: #1565c0; }
.mini-lbl { font-size: 10px; color: #9ca3af; margin-top: 2px; font-weight: 500; }

/* ── ABOUT ── */
.about-item {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-left: 3px solid #1565c0;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px; margin-bottom: 10px;
    font-size: 13px; color: #374151;
}
.about-item strong { color: #1565c0; }

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
        <p>Démo basée sur un modèle Random Forest entraîné sur des données simulées de clients télécom en Guinée.</p>
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

    # ── Barre contexte : HTML pur dans st.markdown, PAS dans st.columns ──
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

    # ── Section infos client ──
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

    # ── Section usage ──
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

    # ── Bouton centré ──
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        predict_btn = st.button("Lancer la prédiction 🔍", use_container_width=True)

    # ── Résultat ──
    if predict_btn:
        x_vec = np.array([
            REGION_MAP[region],
            SEXE_MAP[sexe],
            age,
            revenu,
            anciennete,
            ABO_MAP[type_abonnement],
            OUI_NON_MAP[forfait_international],
            OUI_NON_MAP[messagerie_vocale],
            recharge_mensuelle,
            MOYEN_PAIEMENT_MAP[moyen_paiement],
            minutes_jour,
            minutes_nuit,
            minutes_internationales,
            donnees_mo,
            nombre_sms,
            appels_service_client,
            pannes_signalees_30j,
            retard_paiement_jours,
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
    st.markdown("""
<div class="section-label">📊 À propos du modèle</div>
<div class="about-item"><strong>Algorithme :</strong> Random Forest Classifier (scikit-learn)</div>
<div class="about-item"><strong>Objectif :</strong> prédire si un client va <em>résilier (churn = 1)</em> ou <em>rester (churn = 0)</em></div>
<div class="about-item"><strong>Variables principales :</strong> usage voix, données, pannes signalées, appels SAV, retard de paiement, type d'abonnement</div>
<div class="about-item"><strong>Données :</strong> Dataset simulé représentant des clients télécom en Guinée · 6 régions couvertes</div>
<div class="about-item"><strong>Seuil de décision :</strong> 0.50 sur la probabilité de classe positive</div>
""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        rf_img = IMG_DIR / "random_forest.png"
        if rf_img.exists():
            st.image(str(rf_img), caption="Schéma simplifié d'un Random Forest", use_container_width=True)
        fi_img = IMG_DIR / "data_processing.jpg"
        if fi_img.exists():
            st.image(str(fi_img), caption="Processus global de traitement de données", use_container_width=True)
    with col_b:
        map_img = IMG_DIR / "guinee_map.png"
        if map_img.exists():
            st.image(str(map_img), caption="Contexte : Guinée – clients télécom", use_container_width=True)
        ml_img = IMG_DIR / "ml_models.webp"
        if ml_img.exists():
            st.image(str(ml_img), caption="Famille de modèles de Machine Learning", use_container_width=True)