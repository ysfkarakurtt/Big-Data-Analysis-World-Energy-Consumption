import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dünya Enerji Dashboard", layout="wide")

st.title(" Dünya Enerji Tüketimi ve Karbon Riski Analizi")
st.markdown("""
Bu dashboard, Yusuf Samet Karakurt, Fatih Şişmanoğlu ve Onur Gökkaya tarafından **World Energy Consumption** dataseti kullanılarak geliştirilen 
**Büyük Veri Analizi** projesinin sonuçlarını göstermektedir.
""")
st.divider()

@st.cache_data
def load_data():
    df = pd.read_csv("data/gold_dashboard.csv")
    results = pd.read_csv("data/model_results.csv")
    return df, results

try:
    df, results_df = load_data()
except Exception as e:
    st.error(f"Veri dosyaları yüklenemedi! Lütfen 'data' klasöründe CSV dosyalarının olduğundan emin olun. Hata: {e}")
    st.stop()

st.sidebar.header(" Analiz Filtreleri")

min_year, max_year = int(df['year'].min()), int(df['year'].max())
selected_years = st.sidebar.slider("Yıl Aralığı", min_year, max_year, (min_year, max_year))

countries = st.sidebar.multiselect("Ülkeleri Seçin", options=sorted(df['country'].unique()), default=["Turkey", "Germany", "United States"])

filtered_df = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
if countries:
    filtered_df = filtered_df[filtered_df['country'].isin(countries)]

tab1, tab2, tab3 = st.tabs([" Enerji Trendleri", " Model Performansı", " Ham Veri"])

with tab1:
    st.subheader("Keşifsel Veri Analizi")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_trend = px.line(filtered_df, x="year", y="carbon_intensity_elec", color="country",
                            title="Yıllara Göre Karbon Yoğunluğu Trendi", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col2:
        fig_scatter = px.scatter(filtered_df, x="gdp_per_capita", y="energy_efficiency_score", 
                                 color="country", size="population", hover_name="country",
                                 title="Gelir (GDP) vs Enerji Verimliliği")
        st.plotly_chart(fig_scatter, use_container_width=True)

    fig_hist = px.histogram(filtered_df, x="renewable_ratio", color="country", barmode="overlay",
                            title="Yenilenebilir Enerji Kullanım Oranı Dağılımı")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.subheader("Makine Öğrenmesi Sonuçları")
    
    col_m1, col_m2 = st.columns([2, 1])
    
    with col_m1:
        results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig_models = px.bar(results_melted[results_melted['Metric'] != 'R2'], 
                            x="Model", y="Score", color="Metric", barmode="group",
                            title="Modellerin Hata Skorları (RMSE & MAE) - Düşük olan daha iyidir")
        st.plotly_chart(fig_models, use_container_width=True)
        
    with col_m2:
        st.markdown("**Model Doğruluk Oranları (R2)**")
        st.dataframe(results_df[["Model", "R2"]].sort_values(by="R2", ascending=False), hide_index=True)

with tab3:
    st.subheader("Filtrelenmiş Altın Katman Verisi")
    st.write(f"Şu anki seçimde {len(filtered_df)} satır veri gösteriliyor.")
    st.dataframe(filtered_df, use_container_width=True)

st.sidebar.info("Dashboard başarıyla güncellendi.")