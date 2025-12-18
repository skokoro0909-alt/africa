import pandas as pd
import numpy as np
from prophet import Prophet
import logging

# Prophetのログ（途中経過の文字）がうるさいので静かにさせる設定
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# --- 1. データの読み込みと下準備（お店の開店準備） ---
# ここはサーバー起動時に1回だけ実行されます

# GitHubのRawデータURL
url_elec = "https://raw.githubusercontent.com/skokoro0909-alt/africa/main/API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_254301.csv"
url_pop = "https://raw.githubusercontent.com/skokoro0909-alt/africa/main/API_EN.POP.DNST_DS2_en_csv_v2_130211.csv"

print("データをダウンロード中...少々お待ちください...")

# CSVを読み込む
df_elec_raw = pd.read_csv(url_elec, skiprows=4)
df_pop_raw = pd.read_csv(url_pop, skiprows=4)

# アフリカ諸国のリスト
africa_countries = [
    "Angola", "Benin", "Botswana", "Burkina Faso", "Cabo Verde", "Cameroon", 
    "Central African Republic", "Chad", "Comoros", "Congo, Dem. Rep.", "Congo, Rep.", 
    "Cote d'Ivoire", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", 
    "Gabon", "Gambia, The", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", 
    "Liberia", "Lesotho", "Madagascar", "Malawi", "Mali", "Mauritania", 
    "Mauritius", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", 
    "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", 
    "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", 
    "Togo", "Uganda", "Zambia", "Zimbabwe"
]
# ※元のリストのスペルミス（Angla, Afirica等）をいくつか修正しておきました

# データを整形する関数（下処理担当）
def clean_data(df_raw):
    id_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    year_cols = [col for col in df_raw.columns if col.isdigit() and int(col) >= 1960]
    
    df_long = df_raw.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="ds",
        value_name="y"
    )
    df_long["ds"] = pd.to_datetime(df_long["ds"], format="%Y")
    df_long.dropna(subset=["y"], inplace=True)
    return df_long

# 準備完了したデータ
df_elec_clean = clean_data(df_elec_raw)
df_pop_clean = clean_data(df_pop_raw)

print("データの準備が完了しました！")


# --- 2. 予測機能（注文を受けて調理するシェフ） ---

def get_prediction(country_name):
    """
    国名を受け取り、将来の人口密度と電力アクセス率を予測して
    テキストで返す関数
    """
    
    # --- A. 人口密度の予測 ---
    df_country_pop = df_pop_clean[df_pop_clean["Country Name"] == country_name].copy()
    
    if df_country_pop.empty:
        return f"エラー: {country_name} のデータが見つかりませんでした。国名の英語スペルを確認してください。"

    # Prophetで予測（人口）
    m_pop = Prophet(growth="linear", changepoint_prior_scale=0.05)
    m_pop.fit(df_country_pop)
    future_pop = m_pop.make_future_dataframe(periods=5, freq="Y") # 5年後まで
    forecast_pop = m_pop.predict(future_pop)
    
    # 5年後の値を取得
    pop_result = forecast_pop.iloc[-1]['yhat'] # 最後の行の予測値
    pop_year = forecast_pop.iloc[-1]['ds'].year

    # --- B. 電力アクセス率の予測 ---
    df_country_elec = df_elec_clean[df_elec_clean["Country Name"] == country_name].copy()
    
    if len(df_country_elec) < 5:
        elec_text = "データ不足のため予測できませんでした"
    else:
        df_country_elec["cap"] = 100.0
        m_elec = Prophet(growth="logistic")
        try:
            m_elec.fit(df_country_elec)
            future_elec = m_elec.make_future_dataframe(periods=5, freq="Y")
            future_elec["cap"] = 100.0
            forecast_elec = m_elec.predict(future_elec)
            
            elec_result = forecast_elec.iloc[-1]['yhat']
            # 100%を超えないように調整
            elec_result = min(elec_result, 100.0)
            elec_text = f"約 {elec_result:.1f}%"
        except Exception:
            elec_text = "計算エラー"

    # --- C. 結果を文章にまとめる ---
    result_text = (
        f"【{country_name}のAI予測データ】\n"
        f"■ {pop_year}年の人口密度予測: 1平方kmあたり約 {pop_result:.0f} 人\n"
        f"■ {pop_year}年の電力アクセス率予測: {elec_text}\n"
    )
    
    return result_text