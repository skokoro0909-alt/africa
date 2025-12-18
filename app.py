import os
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from project import search_literature_function 
import model 

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

gemini_pro = genai.GenerativeModel("gemini-2.0-flash")

app = Flask(__name__)

# --- 模擬データ (図書室の本棚のようなもの) ---
DATA = [
    {"title": "Energy Access and Development in Kenya", "year": 2021, "author": "A. Smith", "country": "Kenya", "publisher": "World Bank"},
    {"title": "Off-grid Solar Expansion in West Africa", "year": 2019, "author": "M. Okoye", "country": "Nigeria", "publisher": "IRENA"},
    {"title": "Mini-grid Finance Models in Africa", "year": 2023, "author": "S. Chen", "country": "Ghana", "publisher": "UNDP"},
    {"title": "Rwanda Electrification Pathways", "year": 2020, "author": "J. Kim", "country": "Rwanda", "publisher": "IEA"},
    {"title": "Grid Reliability and Urban Growth", "year": 2022, "author": "A. Smith", "country": "South Africa", "publisher": "OECD"},
]

# 文字が含まれているかチェックする便利な道具
def contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()

@app.route('/api/stats')
def get_stats():
    # 地図（JavaScript）から送られてきた「年」と「種類」を受け取ります
    year = request.args.get('year', type=int)
    layer = request.args.get('layer') # 'population' か 'energy'

    # model.py の中のデータ（掃除済みのデータ）を使います
    if layer == 'population':
        df = model.df_pop_clean
    else:
        df = model.df_elec_clean

    # 指定された「年」のデータだけを抜き出します
    target_data = df[df['ds'].dt.year == year]

    # { "国名": 数値 } という形の辞書（リストのようなもの）を作ります
    result_data = {}
    for _, row in target_data.iterrows():
        result_data[row['Country Name']] = row['y']

    # JavaScriptが読める形式（JSON）にして返信します
    return jsonify(result_data)


# --- 1. トップページ（お家の玄関） ---
@app.route('/') 
def home():
    # index.html を表示します
    return render_template('index.html')

# --- 2. プロジェクト検索ページ（図書室） ---
app.route('/project', methods=['GET', 'POST'])
def project_page():
    # 料理（検索結果）を入れるお皿を準備
    data = {
        "results": [],
        "error_message": "",
        "raw_json": ""
    }
    
    # 検索ボタンが押されたとき（POST）の処理
    if request.method == 'POST':
        # 画面の入力欄 <input name="keyword"> から言葉を受け取る
        user_input = request.form.get("keyword")

        if user_input:
            search_result = search_literature_function(user_input)
            data = search_result
        else:
            data["error_message"] = "検索ワードを入力してください。"

    # 完成した料理（data）を project.html というお部屋に届けます
    return render_template('project.html', **data)

# --- 3. AI生成用の窓口 ---
@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('input')
    response = gemini_pro.generate_content(user_input)
    return jsonify({'response': response.text})

if __name__ == "__main__":
    app.run(debug=True)