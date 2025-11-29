# 以下のライブラリがインストールされているか確認してください
# pip install pandas numpy matplotlib prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet


# データポイントの数 (2000年〜2024年までの25年間)
num_years = 25
start_year = 2000

# 1. 日付 (ds) 列の作成
dates = pd.to_datetime([f'{start_year + i}-01-01' for i in range(num_years)])

# 2. 人口密度 (y_pop) 列の作成
# 基本的な増加トレンドを想定し、ノイズを加える
base_pop_density = np.linspace(3500, 5800, num_years) # 3500から5800への増加
noise_pop = np.random.normal(0, 100, num_years) # 平均0、標準偏差100のランダムノイズ
y_pop_density = base_pop_density + noise_pop

# 3. 電力アクセス率 (y_elec) 列の作成
# 緩やかな増加から急増し、100%に近づくS字カーブを想定（シグモイド関数的な動き）
years = np.arange(num_years)
# ロジスティック曲線 (S字カーブ) の作成
# シグモイド関数: 1 / (1 + exp(-x))
logistic_growth = 1 / (1 + np.exp(-(years - 15) / 3)) * 100
# 実際のアクセス率は80%付近を上限として、ノイズを加える
y_elec_access = np.clip(logistic_growth * 0.85 + np.random.normal(0, 2, num_years), 0, 95)
# 0%から95%の間にクリップ（制限）する

# データフレームの結合
dummy_data = pd.DataFrame({
    'ds': dates,
    'y_pop': y_pop_density.round(0),
    'y_elec': y_elec_access.round(2)
})



# 最初の5行を表示
print("--- 模擬データセット（人口密度と電力アクセス率） ---")
print(dummy_data.head())

# 統計情報を確認
# print("\n--- 統計情報 ---")
# print(dummy_data.describe())

# データの可視化 (トレンドの確認)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(dummy_data['ds'], dummy_data['y_pop'], label='Population Density')
plt.title('Simulated Population Density Trend')
plt.xlabel('Year')
plt.ylabel('Density (persons/sq km)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(dummy_data['ds'], dummy_data['y_elec'], label='Electricity Access Rate', color='red')
plt.title('Simulated Electricity Access Rate Trend')
plt.xlabel('Year')
plt.ylabel('Access Rate (%)')
plt.grid(True)

plt.tight_layout()
plt.show()


# Prophet用にデータセットを準備（y_popを使用）
df_pop = dummy_data[['ds', 'y_pop']].rename(columns={'y_pop': 'y'})

# モデルの初期化と学習
m_pop = Prophet()
m_pop.fit(df_pop)

# 将来の期間を定義 (2025年から5年間 = 5 periods)
future_pop = m_pop.make_future_dataframe(periods=5, freq='Y')
forecast_pop = m_pop.predict(future_pop)

# 予測結果のプロット
m_pop.plot(forecast_pop)
plt.title('Population Density Forecast (Prophet)')
plt.xlabel('Year')
plt.ylabel('Density (persons/sq km)')
plt.show()



# Prophet用にデータセットを準備（y_elecを使用）
df_elec = dummy_data[['ds', 'y_elec']].rename(columns={'y_elec': 'y'})

# 最大容量 (cap) を設定
df_elec['cap'] = 100.0

# モデルの初期化と学習 (成長モデルを指定)
m_elec = Prophet(growth='logistic')
m_elec.fit(df_elec)

# 将来の期間を定義し、capを設定
future_elec = m_elec.make_future_dataframe(periods=5, freq='Y')
future_elec['cap'] = 100.0 # 未来の予測期間にも cap が必要
forecast_elec = m_elec.predict(future_elec)

# 予測結果のプロット
m_elec.plot(forecast_elec)
plt.title('Electricity Access Rate Forecast (Logistic Growth)')
plt.xlabel('Year')
plt.ylabel('Access Rate (%)')
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

africa_countries = [
    "Angla",
    "Benin",
    "Botswana",
    "Burkina Faso",
    "Cabo Verde",
    "Cameroon",
    "Central Afurican Republic",
    "Chad",
    "Comoros",
    "Congo,Dem.rep.",
    "Congo,Rep.",
    "Cote d'lvoire",
    "Equatorial Guinea",
    "Eritrea",
    "Eswatini",
    "Ethiopia",
    "Gabon",
    "Gambia,The",
    "Ghana",
    "Guinea",
    "Guinea-Bissau",
    "Kenya",
    "Liberia",
    "Lesotho",
    "Madagascar",
    "Malawi",
    "Mali",
    "Mauritania",
    "Mauritius",
    "Mozambique",
    "Namibia",
    "Niger",
    "Nigeria",
    "Rwanda",
    "Sao Tome and Principe",
    "Senegal",
    "Seychelles",
    "Sierra Leone",
    "Somalia,Fed.Rep",
    "South Afirica",
    "South Sudan",
    "Sudan",
    "Tanzania",
    "Togo",
    "Uganda",
    "Zambia",
    "Zimbabwe",
]

df_el_accese = pd.read_csv("API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_254301.csv", skiprows=4)
id_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
year_cols = [col for col in df_el_accese.columns if col.isdigit() and int(col) >= 1960]

df_el_accese_long = df_el_accese.melt(
    id_vars=id_cols,
    value_vars=year_cols,
    var_name="ds",  # 年が入る列を 'ds' に
    value_name="y",  # 値が入る列を 'y' に
)
df_el_accese_long["ds"] = pd.to_datetime(df_el_accese_long["ds"], format="%Y")
df_el_accese_long.dropna(subset=["y"], inplace=True)
df_el_accese_long.reset_index(drop=True, inplace=True)
df_el_accese_africa = df_el_accese_long[
    df_el_accese_long["Country Name"].isin(africa_countries)
].copy()
print("\n--- 最終整形後のデータ (先頭10行) ---")
print(df_el_accese_africa.head(10))

df_density = pd.read_csv("API_EN.POP.DNST_DS2_en_csv_v2_130211.csv", skiprows=4)
id_cols_d = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
year_cols_d = [col for col in df_density.columns if col.isdigit() and int(col) >= 1960]

df_density_long = df_density.melt(
    id_vars=id_cols_d,
    value_vars=year_cols_d,
    var_name="ds",  # 年が入る列を 'ds' に
    value_name="y",  # 値が入る列を 'y' に
)
df_density_long["ds"] = pd.to_datetime(df_density_long["ds"], format="%Y")
df_density_long.dropna(subset=["y"], inplace=True)
df_density_long.reset_index(drop=True, inplace=True)
df_density_africa = df_density_long[
    df_density_long["Country Name"].isin(africa_countries)
].copy()
print("\n--- 最終整形後のデータ (先頭10行) ---")
print(df_density_africa.head(10))


all_pop_forecasts = []
prediction_periods = 12

for country_name in df_density_africa["Country Name"].unique():
    print(f"\n--- {country_name} の人口密度予測を開始 ---")

    df_country = df_density_long[df_density_long["Country Name"] == country_name].copy()

    # 🌟 変更点: ロジスティックではなく線形成長モデル（デフォルト）を使用し、cap は設定しない
    m = Prophet(growth="linear", changepoint_prior_scale=0.05)

    # モデルの学習
    m.fit(df_country)

    # 将来のデータフレームを作成
    future = m.make_future_dataframe(periods=prediction_periods, freq="Y")

    # 予測の実行 (futureに cap を設定する必要もありません)
    forecast = m.predict(future)

    # 結果の整形と集約
    forecast["Country Name"] = country_name
    forecast["actual_y"] = df_country["y"].combine_first(
        pd.Series([np.nan] * len(forecast))
    )
    all_pop_forecasts.append(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "Country Name", "actual_y"]]
    )
df_pop_forecasts = pd.concat(all_pop_forecasts, ignore_index=True)
print(df_pop_forecasts.tail(100))


print("\n✅ 人口密度予測が完了し、df_pop_forecasts が生成されました。")

# 必要な列に絞り込み、予測値を丸める
df_pop_forecasts_clean = (
    df_pop_forecasts[["Country Name", "ds", "yhat"]]
    .rename(columns={"yhat": "pop_density_hat"})
    .copy()
)
df_pop_forecasts_clean["pop_density_hat"] = df_pop_forecasts_clean[
    "pop_density_hat"
].round(2)

print("--- 年ごとの予測データ（最新の過去年以降）---")
print(df_pop_forecasts_clean.head(10))
print(f"\n合計 {len(df_pop_forecasts_clean)} 行の予測データが作成されました。")

# 予測結果のプロット
m.plot(forecast)
plt.title(f"Population Density Forecast for {country_name} (Prophet)")
plt.xlabel("Year")
plt.ylabel("Density (persons/sq km)")
plt.show()


all_elec_forecasts = []
prediction_periods = 5  # 5年間予測

# df_el_accese_africa から、予測対象のアフリカ諸国のリストを取得
countries_to_forecast = df_el_accese_africa["Country Name"].unique()

for country_name in countries_to_forecast:
    print(f"\n--- {country_name} の電気アクセス率予測を開始 ---")

    # 1. 特定の国のデータを抽出
    df_elec_country = df_el_accese_africa[
        df_el_accese_africa["Country Name"] == country_name
    ].copy()

    # データが少なすぎる場合はスキップ
    if len(df_elec_country) < 5:
        print(
            f"警告: {country_name} のデータポイントが少なすぎます ({len(df_elec_country)})。スキップします。"
        )
        continue

    # 2. Prophet用にデータセットを準備（dsとyのみ）
    df_elec = df_elec_country[["ds", "y"]].copy()

    # 3. 最大容量 (cap) を設定 (アクセス率は最大 100%)
    df_elec["cap"] = 100.0

    # モデルの初期化と学習 (ロジスティック成長モデルを使用)
    m_elec = Prophet(growth="logistic")
    try:
        m_elec.fit(df_elec)
    except Exception as e:
        print(
            f"エラー: {country_name} のモデル学習中にエラーが発生しました: {e}。スキップします。"
        )
        continue

    # 将来の期間を定義し、capを設定
    future_elec = m_elec.make_future_dataframe(periods=prediction_periods, freq="Y")
    future_elec["cap"] = 100.0

    # 予測の実行
    forecast_elec = m_elec.predict(future_elec)

    # 結果の整形と集約
    forecast_elec["Country Name"] = country_name
    all_elec_forecasts.append(
        forecast_elec[["ds", "yhat", "yhat_lower", "yhat_upper", "Country Name"]]
    )

# すべての予測結果を結合
df_elec_forecasts = pd.concat(all_elec_forecasts, ignore_index=True)

print(
    "\n✅ 全アフリカ諸国の電気アクセス率予測が完了し、df_elec_forecasts が生成されました。"
)
print("\n--- 結合された電気アクセス率予測データ (最新の10行) ---")
print(df_elec_forecasts.head(10))

# 最初の国の予測結果をプロット (例: Angola)
first_country_name = df_elec_forecasts["Country Name"].iloc[0]
df_plot = df_elec_forecasts[df_elec_forecasts["Country Name"] == first_country_name]
m_elec.plot(df_plot)
plt.title(f"{first_country_name} Electricity Access Rate Forecast (Logistic Growth)")
plt.xlabel("Year")
plt.ylabel("Access Rate (%)")
plt.show()
