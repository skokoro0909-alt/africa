import os
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from project import search_literature_function

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

gemini_pro = genai.GenerativeModel("gemini-2.0-flash")




app = Flask(__name__)
@app.route('/') 
def home():
    return render_template('index.html')

@app.route('/project')
def Project():
   return render_template("project.html")


@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('input')
    response = gemini_pro.generate_content(user_input)
    return jsonify({'response': response.text})

@app.route('/', methods=['GET', 'POST'])
def index():
    # まずは、お皿（データ）を空の状態で用意します
    # これが project.html の {{ results }} や {{ error_message }} に渡されます
    data = {
        "results": [],
        "error_message": "",
        "raw_json": ""
    }
    
    # --- 「検索ボタン」が押された（POST）ときの処理 ---
    if request.method == 'POST':
        # お客さんの注文（キーワード）を聞き取る
        # project.html の <input name="keyword"> の中身を取得
        user_input = request.form.get("keyword")

        # 注文があれば、裏方のシェフ（別ファイルの関数）に仕事を依頼！
        if user_input:
            # ★ここで search_service.py の機能を使います
            # 戻り値として、結果が入った辞書（お弁当）を受け取ります
            search_result = search_literature_function(user_input)

            # 受け取ったお弁当を、今回テーブルに出すデータとして採用
            data = search_result
            
        else:
            # もしキーワードが空っぽだったら
            data["error_message"] = "検索ワードを入力してください。"

    # --- 3. 料理の提供 ---
    # データを project.html に渡して表示させる
    # **data と書くことで、辞書の中身（resultsなど）を個別の変数として渡せます
    return render_template('project.html', **data)



DATA = [
    {"title": "Energy Access and Development in Kenya", "year": 2021, "author": "A. Smith", "country": "Kenya", "publisher": "World Bank"},
    {"title": "Off-grid Solar Expansion in West Africa", "year": 2019, "author": "M. Okoye", "country": "Nigeria", "publisher": "IRENA"},
    {"title": "Mini-grid Finance Models in Africa", "year": 2023, "author": "S. Chen", "country": "Ghana", "publisher": "UNDP"},
    {"title": "Rwanda Electrification Pathways", "year": 2020, "author": "J. Kim", "country": "Rwanda", "publisher": "IEA"},
    {"title": "Grid Reliability and Urban Growth", "year": 2022, "author": "A. Smith", "country": "South Africa", "publisher": "OECD"},
]

def contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()

@app.route("/projects", methods=["GET"])
def projects():
    # クエリ取得
    q = (request.args.get("q") or "").strip()
    country = (request.args.get("country") or "").strip()
    author = (request.args.get("author") or "").strip()
    year_raw = (request.args.get("year") or "").strip()

    year = None
    if year_raw:
        try:
            year = int(year_raw)
        except ValueError:
            year = None

    # 検索したかどうか（フォームから何か入力があったら True）
    searched = any([q, country, author, year_raw])

    results = DATA

    # フィルタ：Keyword（title/publisher/author/countryに対して）
    if q:
        results = [
            item for item in results
            if contains(item["title"], q)
            or contains(item["publisher"], q)
            or contains(item["author"], q)
            or contains(item["country"], q)
        ]

    # フィルタ：Country
    if country:
        results = [item for item in results if contains(item["country"], country)]

    # フィルタ：Author
    if author:
        results = [item for item in results if contains(item["author"], author)]

    # フィルタ：Year
    if year is not None:
        results = [item for item in results if item["year"] == year]

    return render_template(
        "project.html",
        q=q,
        country=country,
        author=author,
        year=year_raw,
        results=results,
        searched=searched,
    )

# トップ（地図）から Finder に飛ばす用
@app.route("/finder")
def finder_redirect():
    return render_template("project.html", results=None, searched=False)

if __name__ == "__main__":
    app.run(debug=True)

if __name__ == '__main__':
    app.run()