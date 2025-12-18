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

# --- 1. 地図にデータを届ける「専用の窓口」 ---
# index.html の地図が動いたときに、ここへデータを取りに来ます
@app.route('/api/stats')
def get_stats():
    # 地図から「何年」の「何のデータ」が欲しいか受け取ります
    year = request.args.get('year', type=int)
    layer = request.args.get('layer') # 'population' か 'energy'

    # model.py で準備したデータ（df_pop_clean など）を使います
    if layer == 'population':
        df = model.df_pop_clean
    else:
        df = model.df_elec_clean

    # 指定された「年」のデータだけを絞り込みます
    # ds列（日付）から年を取り出して比較します
    target_data = df[df['ds'].dt.year == year]

    # JavaScriptが使いやすいように { "国名": 数値 } という形に変換します
    result_data = {}
    for _, row in target_data.iterrows():
        result_data[row['Country Name']] = row['y']

    return jsonify(result_data)

# --- 2. トップページ（地図の表示） ---
@app.route('/') 
def home():
    return render_template('index.html')

# --- 3. プロジェクト検索ページ ---
@app.route('/project', methods=['GET', 'POST'])
def project_page():
    data = {
        "results": [],
        "error_message": "",
        "raw_json": ""
    }
    
    if request.method == 'POST':
        user_input = request.form.get("keyword")
        if user_input:
            search_result = search_literature_function(user_input)
            data = search_result
        else:
            data["error_message"] = "検索ワードを入力してください。"

    return render_template('project.html', **data)

# --- 4. AI生成用の窓口 ---
@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('input')
    response = gemini_pro.generate_content(user_input)
    return jsonify({'response': response.text})

if __name__ == "__main__":
    app.run(debug=True)