import os
import io
import tempfile
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import math
import numpy as np
import base64
from PIL import Image 

# =========================================================
# GÖRSEL MİMARİ: LOGO ŞİFRELEME VE FAVICON DÜZELTİCİ
# =========================================================
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return ""

def get_square_favicon(image_path):
    
    try:
        img = Image.open(image_path)
        max_dim = max(img.size)
        square_img = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
        offset = ((max_dim - img.size[0]) // 2, (max_dim - img.size[1]) // 2)
        square_img.paste(img, offset)
        return square_img
    except Exception:
        return "📦" 

logo_path = "man_logo.png" 
logo_base64 = get_base64_image(logo_path)
favicon_img = get_square_favicon(logo_path)

from core.data_pipeline import load_and_preprocess_data
from app import (
    build_global_dataset,
    train_and_validate_model,
    forecast_future_for_sku,
    optimize_inventory,
)

# =========================================================
# 1. SAYFA VE TEMA AYARLARI
# =========================================================
st.set_page_config(
    page_title="MAN Türkiye | Siparişleme Algoritması",
    page_icon=favicon_img, 
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: radial-gradient(circle at 15% 20%, #1f2a44 0%, transparent 22%),
                    radial-gradient(circle at 85% 15%, #312e81 0%, transparent 25%),
                    radial-gradient(circle at 50% 80%, #0f766e 0%, transparent 25%),
                    linear-gradient(135deg, #0b1020 0%, #111827 45%, #0a0f1f 100%);
        color: #e5e7eb;
    }
    .glass-card { background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.16); border-radius: 18px; backdrop-filter: blur(8px); padding: 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
    .hero-title { font-size: 2.2rem; font-weight: 800; color: #D4AF37; margin-bottom: 0.2rem; }
    .hero-sub { color: #cbd5e1; font-size: 1.1rem; font-weight: 500; }
    
    .kpi { border-radius: 16px; padding: 1.2rem; background: linear-gradient(145deg, rgba(212, 175, 55, 0.15), rgba(212, 175, 55, 0.05)); border: 1px solid rgba(212, 175, 55, 0.3); border-left: 5px solid #D4AF37; transition: 0.3s; }
    .kpi:hover { transform: translateY(-3px); box-shadow: 0 6px 15px rgba(0,0,0,0.2); }
    .kpi-title { color: #cbd5e1; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: flex; align-items: center; gap: 6px; }
    .kpi-value { color: #ffffff; font-size: 1.9rem; font-weight: 800; margin-top: 5px; }
    
    .tooltip-icon {
        display: inline-flex; align-items: center; justify-content: center;
        width: 18px; height: 18px; border-radius: 50%;
        background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(212, 175, 55, 0.5);
        color: #D4AF37; font-size: 0.75rem; font-weight: bold; cursor: help;
        margin-left: 5px; transition: 0.2s;
    }
    .tooltip-icon:hover { background: rgba(212, 175, 55, 0.2); box-shadow: 0 0 8px rgba(212, 175, 55, 0.4); }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 0; }
    .stTabs [data-baseweb="tab"] { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.16); border-radius: 12px; color: #d1d5db; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background: linear-gradient(90deg, #D4AF37, #FDE047); color: #000 !important; font-weight: bold; }
    .stButton > button { background: linear-gradient(90deg, #D4AF37, #FDE047); color: #000; border: 0; border-radius: 10px; font-weight: 800; padding: 0.8rem; font-size: 1.1rem; transition: transform 0.2s; }
    .stButton > button:hover { transform: translateY(-2px); }

    .glass-metric {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        backdrop-filter: blur(10px); border: 1px solid rgba(212, 175, 55, 0.2);
        border-top: 4px solid #D4AF37; border-radius: 16px; padding: 25px 15px;
        text-align: center; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .glass-metric:hover {
        transform: translateY(-5px); box-shadow: 0 12px 25px rgba(0,0,0,0.4);
        border-color: rgba(212, 175, 55, 0.6); background: linear-gradient(135deg, rgba(212,175,55,0.1) 0%, rgba(255,255,255,0.03) 100%);
    }
    .m-title { color: #cbd5e1; font-size: 0.95rem; font-weight: 600; letter-spacing: 1px; margin-bottom: 12px; display: flex; justify-content: center; align-items: center; gap: 6px; }
    .m-val { color: #ffffff; font-size: 2.8rem; font-weight: 800; margin-bottom: 10px; }
    .m-sub { font-size: 0.85rem; font-weight: 700; padding: 5px 12px; border-radius: 20px; display: inline-block; background: rgba(255,255,255,0.05); letter-spacing: 0.5px; }
    </style>
    """, unsafe_allow_html=True
)

@st.cache_data(show_spinner=False)
def load_demand(path: str): return load_and_preprocess_data(path)

@st.cache_data(show_spinner=False)
def load_plan(path: str, ext: str): return pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)

for key, default in {"is_processed": False, "results_df": pd.DataFrame(), "metrics": {}, "financials": {}, "feat_imp": pd.DataFrame()}.items():
    if key not in st.session_state: st.session_state[key] = default

with st.sidebar:
   
    if logo_base64:
        st.markdown(f'<div style="text-align: center; margin-bottom: 20px;"><img src="data:image/png;base64,{logo_base64}" width="160"></div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ man_logo.png bulunamadı!")
        
    st.markdown("<h4 style='text-align: center; color: #D4AF37; margin-bottom: 20px;'>Türkiye Fabrikası</h4>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top:0; margin-bottom:20px; border: 1px solid rgba(212, 175, 55, 0.3);'>", unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Operasyon Parametreleri")
    review_period = st.slider("📆 Gözden Geçirme (Hafta)", 1, 12, 4, help="Siparişlerin ne sıklıkla değerlendirilip sisteme girileceğini belirler. Örneğin: 4, ayda bir kez sipariş verileceği anlamına gelir.")
    service_level = st.slider("🛡️ Hedef Hizmet Düzeyi", 0.80, 0.99, 0.95, 0.01, help="İstenen stok bulunabilirlik oranıdır. %95, üretim bandında parçanın %95 ihtimalle hazır bulunmasını hedefler.")
    forecast_steps = st.slider("🔮 Tahmin Ufku (Hafta)", 4, 16, 8, help=" Algoritmanın gelecekteki kaç haftanın tüketimini öngöreceğini belirler.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(
"""<div style="background: linear-gradient(145deg, #1e293b, #0f172a); border: 1px solid rgba(212, 175, 55, 0.25); border-radius: 16px; padding: 22px 18px; box-shadow: 0 10px 20px rgba(0,0,0,0.3); position: relative; overflow: hidden;">
<div style="position: absolute; top: -15px; right: -15px; opacity: 0.08; font-size: 90px;">🎓</div>
<p style="margin:0; font-size: 1.15rem; color: #D4AF37; font-weight: 800; letter-spacing: 0.5px;">GAZİ ÜNİVERSİTESİ</p>
<p style="margin:0 0 15px 0; font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Endüstri Mühendisliği</p>
<div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin-bottom: 12px;">
<p style="margin:0 0 3px 0; font-size: 0.7rem; color: #a1a1aa; text-transform: uppercase; letter-spacing: 1px;">Proje Danışmanı</p>
<p style="margin:0; font-size: 0.9rem; color: #f8fafc; font-weight: 600;">Prof. Dr. Gül Didem Batur Sir</p>
</div>
<div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px;">
<p style="margin:0 0 5px 0; font-size: 0.7rem; color: #a1a1aa; text-transform: uppercase; letter-spacing: 1px;">Geliştirici Ekip</p>
<p style="margin:0; font-size: 0.85rem; color: #D4AF37; font-weight: 600;">• M. Hanefi Yazar</p>
<p style="margin:0 0 4px 0; font-size: 0.85rem; color: #e2e8f0;">• Ayşegül Çoban</p>
<p style="margin:0 0 4px 0; font-size: 0.85rem; color: #e2e8f0;">• Ezgi Ece Mart</p>
</div>
</div>""", unsafe_allow_html=True)


if logo_base64:
    img_tag = f'<img src="data:image/png;base64,{logo_base64}" width="110" style="margin-right: 25px;">'
else:
    img_tag = ""

st.markdown(
f"""<div class="glass-card" style="display: flex; align-items: center; border-left: 8px solid #D4AF37;">
{img_tag}
<div>
<div class="hero-title">MAN Türkiye A.Ş. | Siparişleme Algoritması</div>
<p class="hero-sub" style="margin: 0;">Kanban Parçalar için Çoklu Ürün Ağacı (BOM) & Hibrit (ADIDA+SBA+TSB+CatBoost) Envanter Optimizasyonu</p>
</div>
</div>""", unsafe_allow_html=True)
st.write("")

tab_in, tab_dash, tab_table, tab_ai = st.tabs(["📥 Veri Girişi", "🎯 Yönetici Özeti", "📋 Sipariş Tablosu", "🧠 Model Performans Analizi"])

with tab_in:
    st.info("💡 Lütfen geçmiş tüketim verilerini ve 10 farklı aracın üretim planını yükleyerek analizi başlatın.")
    c1, c2 = st.columns(2)
    with c1: up_demand = st.file_uploader("📄 Geçmiş Tüketim (.xlsx / .xls)", type=["xlsx", "xls", "csv"], help="40 Parçanın geçmişte haftalık olarak kaç adet tüketildiğini içeren tablo.")
    with c2: up_plan = st.file_uploader("🧾 Üretim Planı (.xlsx / .xls)", type=["xlsx", "xls", "csv"], help="Önümüzdeki haftalarda 10 farklı MAN/NEOPLAN modelinden kaçar adet üretileceğini gösteren plan.")
    
    if st.button("🚀 ANALİZİ VE OPTİMİZASYONU BAŞLAT", use_container_width=True, help="Tüm verileri algoritmaya gönderir ve optimum sipariş adetlerini hesaplar."):
        if not (up_demand and up_plan): st.error("⚠️ Lütfen her iki dosyayı da yüklediğinizden emin olun!")
        else:
            ext_d = os.path.splitext(up_demand.name)[1].lower()
            ext_p = os.path.splitext(up_plan.name)[1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext_d) as td, tempfile.NamedTemporaryFile(delete=False, suffix=ext_p) as tp:
                td.write(up_demand.getvalue()); tp.write(up_plan.getvalue())
                d_path, p_path = td.name, tp.name

            try:
                with st.spinner("Çoklu araç üretim planları makine öğrenmesine entegre ediliyor ve siparişler hesaplanıyor..."):
                    df = load_demand(d_path); plan_df = load_plan(p_path, ext_p)
                    g_df = build_global_dataset(df)
                    
                    X = g_df.drop(["date", "demand", "mevcut_stok", "lot_size", "birim_fiyat"], axis=1, errors='ignore')
                    y = g_df["demand"]
                    
                    model, metrics, feat_imp = train_and_validate_model(X, y, ["sku","parca_ailesi"])
                    st.session_state.metrics = metrics
                    st.session_state.feat_imp = feat_imp 
                    
                    res_list = []; total_yeni_maliyet = 0.0; total_eski_maliyet = 0.0
                    for sku in df["sku"].unique():
                        s_data = g_df[g_df["sku"] == sku]
                        if s_data.empty: continue
                        
                        hist_demand = s_data["demand"].values
                        last = s_data.drop(["date", "demand"], axis=1).iloc[-1:]
                        
                        lt = last["lead_time"].values[0]
                        stok = last["mevcut_stok"].values[0]
                        lot = last["lot_size"].values[0]
                        fiyat = last["birim_fiyat"].values[0]
                        
                        fcast = forecast_future_for_sku(model, last, plan_df, forecast_steps)
                        opt = optimize_inventory(fcast, hist_demand, lt, review_period, service_level, stok, lot, fiyat)
                        total_yeni_maliyet += opt["toplam_maliyet_euro"]
                        
                        mean_hist = np.mean(hist_demand)
                        std_hist = np.std(hist_demand) if np.std(hist_demand) > 0 else 0.1
                        
                        eski_emniyet = 2.33 * std_hist * math.sqrt(lt)
                        eski_s = (mean_hist * lt) + eski_emniyet
                        eski_S = (mean_hist * (lt + review_period)) + eski_emniyet
                        
                        if stok <= eski_s:
                            ham_eski_siparis = max(eski_S - stok, 0)
                            eski_siparis_adet = math.ceil(ham_eski_siparis / lot) * lot
                        else:
                            eski_siparis_adet = 0
                            
                        total_eski_maliyet += (eski_siparis_adet * fiyat)
                        
                        res_list.append({
                            "SKU": sku, 
                            "Aile": last["parca_ailesi"].values[0], 
                            "Lead Time": int(lt),
                            "Stok": int(stok), 
                            "Fiyat": fiyat,
                            "Sipariş": opt["final_order_qty"], 
                            "Maliyet": opt["toplam_maliyet_euro"],
                            "Risk": opt["final_stockout_risk"], 
                            "Yeterlilik (Hf)": opt["wos"]
                        })
                    
                    st.session_state.results_df = pd.DataFrame(res_list).sort_values("Risk", ascending=False).reset_index(drop=True)
                    
                    fark = total_eski_maliyet - total_yeni_maliyet
                    st.session_state.financials = {
                        "new": total_yeni_maliyet, 
                        "old": total_eski_maliyet, 
                        "fark": fark
                    }
                    st.session_state.is_processed = True
                    st.success("✅ 40 parçalık Çoklu BOM analizi başarıyla tamamlandı!")
            except Exception as e: st.error(f"Sistem Hatası: {type(e).__name__} - {e}")
            finally: 
                if os.path.exists(d_path): os.unlink(d_path)
                if os.path.exists(p_path): os.unlink(p_path)

with tab_dash:
    if not st.session_state.is_processed: st.warning("Lütfen veri yükleyip analizi başlatın.")
    else:
        f = st.session_state.financials; r = st.session_state.results_df
        k1, k2, k3 = st.columns(3)
        
        if f["fark"] > 0:
            fark_text = f"<div style='color:#10b981; font-weight:bold; font-size: 0.9rem; margin-top:3px;'>↓ €{f['fark']:,.2f} Stok Tasarrufu</div>"
        elif f["fark"] < 0:
            fark_text = f"<div style='color:#f59e0b; font-weight:bold; font-size: 0.9rem; margin-top:3px;'>↑ €{abs(f['fark']):,.2f} Risk Önleyici Ek Sipariş</div>"
        else:
            fark_text = "<div style='color:#94a3b8; font-weight:bold; font-size: 0.9rem; margin-top:3px;'>Geleneksel Sistemle Eşit</div>"
        
        k1.markdown(f'<div class="kpi"><div class="kpi-title">📉 Mevcut Bütçe <span class="tooltip-icon" title="Eğer hibrit makine öğrenmesi modeli kullanılmasaydı; fabrikanın sadece geçmiş tüketim ortalamasına ve Min-Max güvenlik katsayılarına dayanarak vereceği kör siparişlerin maliyetidir.">?</span></div><div class="kpi-value">€{f["old"]:,.2f}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi"><div class="kpi-title">💸 Optimize Bütçe <span class="tooltip-icon" title="Modelin gelecekteki üretim planını (MRP) hesaba katarak hesapladığı optimum siparişlerin maliyetidir.">?</span></div><div class="kpi-value">€{f["new"]:,.2f}</div>{fark_text}</div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi"><div class="kpi-title">📦 Analiz Edilen Parça <span class="tooltip-icon" title="Sistemde başarıyla işlenen toplam C-Sınıfı ortak (BOM) parça sayısı.">?</span></div><div class="kpi-value">{len(r)} Adet</div></div>', unsafe_allow_html=True)
        
        st.write("")
        c_left, c_right = st.columns([1.5, 1])
        with c_left: st.plotly_chart(px.bar(r, x="SKU", y="Sipariş", color="Aile", title="Parça Ailesine Göre Sipariş Dağılımı", template="plotly_dark"), use_container_width=True)
        with c_right: st.plotly_chart(px.pie(r, names="Aile", values="Maliyet", hole=0.5, title="Bütçe Dağılımı", template="plotly_dark"), use_container_width=True)

with tab_table:
    if not st.session_state.is_processed:
        st.warning("Lütfen veri yükleyip analizi başlatın.")
    else:
        st.markdown("### 📋 Satın Alma İş Emirleri")
        
        def style_dataframe(df):
            return df.style.format({
                "Fiyat": "€{:,.2f}", 
                "Maliyet": "€{:,.2f}", 
                "Risk": "{:.1%}", 
                "Yeterlilik (Hf)": "{:.1f}"
            }).background_gradient(subset=["Risk"], cmap="RdYlGn_r").map(
                lambda val: 'font-weight: bold; color: #ef4444;' if val > 0.05 else ('font-weight: bold; color: #f59e0b;' if val > 0.01 else 'color: #10b981;'),
                subset=["Risk"]
            ).set_properties(**{
                'background-color': '#1e293b',
                'color': '#f8fafc',
                'border': '1px solid #334155'
            })
            
        st.dataframe(style_dataframe(st.session_state.results_df), use_container_width=True, height=500, hide_index=True)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.results_df.to_excel(writer, index=False, sheet_name='Siparisler')
        
        st.download_button(
            label="📥 Tabloyu Excel Olarak İndir",
            data=buffer.getvalue(),
            file_name="man_siparis_listesi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Oluşturulan sipariş iş emirlerini tedarik birimine iletmek üzere gerçek bir .xlsx dosyası olarak indirir."
        )

with tab_ai:
    if not st.session_state.is_processed:
        st.warning("Lütfen veri yükleyip analizi başlatın.")
    else:
        m = st.session_state.metrics
        
        mase = m.get("MASE", 0)
        mase_col = "#10b981" if mase < 1.0 else "#ef4444"
        mase_txt = "YZ Başarılı" if mase < 1.0 else "Standart Model"
        
        mae = m.get("MAE", 0)
        rmse = m.get("RMSE", 0)
        
        st.markdown(f"""
<div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap; margin-top: 15px;">

<div class="glass-metric" style="flex: 1; min-width: 220px;">
<div class="m-title">⚖️ MASE (ÖLÇEKLİ) <span class="tooltip-icon" title="Hibrit makine öğrenmesi algoritmasının standart tahmine göre ne kadar üstün olduğunu gösterir sistemin temel performans göstergesidir.">?</span></div>
<div class="m-val" style="color: #D4AF37;">{mase}</div>
<div class="m-sub" style="color: {mase_col}; border: 1px solid {mase_col}50; background: {mase_col}15;">Referans: < 1.0 ({mase_txt})</div>
</div>

<div class="glass-metric" style="flex: 1; min-width: 220px;">
<div class="m-title">📦 MAE (ADET) <span class="tooltip-icon" title="Haftalık bazda tahmindeki ortalama adet sapmasıdır. Değerin küçük olup olmadığına, parçanın ortalama sipariş adedine (Mean Demand) bakılarak karar verilmelidir.">?</span></div>
<div class="m-val" style="color: #D4AF37;">{mae}</div>
<div class="m-sub" style="color: #94a3b8; border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.05);">Tüketim Hacmine Oranla Düşük Olmalı</div>
</div>

<div class="glass-metric" style="flex: 1; min-width: 220px;">
<div class="m-title">🚨 RMSE (RİSK) <span class="tooltip-icon" title="Büyük sapmaları ve hataları sert cezalandıran (karesini alan) risk ölçeğidir. Yüzdelik değil, adet cinsindendir. Stoksuz kalma (Stock-out) riskini yönetmek için izlenir.">?</span></div>
<div class="m-val" style="color: #D4AF37;">{rmse}</div>
<div class="m-sub" style="color: #94a3b8; border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.05);">Tüketim Hacmine Oranla Düşük Olmalı</div>
</div>

</div>
""", unsafe_allow_html=True)

        st.markdown("<h4 style='color: #D4AF37; margin-top: 40px;'>🔍 Açıklanabilir Makine Öğrenmesi: Model Karar Ağırlıkları</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 0.85rem;'> Hibrit makine öğrenmesi algoritmasının siparişleri hesaplarken hangi verilere ne oranda (% olarak) güvendiğini gösterir. Bu sayede model bir 'Kara Kutu' olmaktan çıkar.</p>", unsafe_allow_html=True)
        
        fi_df = st.session_state.feat_imp
        if not fi_df.empty:
            top_fi = fi_df.head(10).sort_values(by="Importances", ascending=True)
            fig = px.bar(top_fi, x="Importances", y="Feature Id", orientation='h', 
                         color="Importances", color_continuous_scale="YlOrRd",
                         text_auto='.1f')
            fig.update_layout(template="plotly_dark", showlegend=False, height=400,
                              xaxis_title="Karar Etkisi (%)", yaxis_title="Veri Tipi (Öznitelik)")
            st.plotly_chart(fig, use_container_width=True)
