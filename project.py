import time
import re
import json
import google.generativeai as genai
# --- 内部用：一番いいモデル（シェフ）を選ぶ関数 ---
def get_best_model():
    """
    どのAIモデルを使うか決める関数です。
    基本は最新の 'gemini-2.5-flash' を選びます。
    """
    return 'gemini-2.5-flash', True
# --- 外部用：文献検索を実行するメインの関数 ---
def search_literature_function(user_input):
    """
    キーワードを受け取り、AIに検索させて結果を返します。
    返り値は、app.pyで使いやすいように辞書形式にします。
    """
    # 結果を入れておく空のお弁当箱を用意
    result_data = {
        "results": [],
        "error_message": "",
        "raw_json": ""
    }
    # キーワードが空っぽなら、何もせず帰す
    if not user_input:
        result_data["error_message"] = "検索ワードを入力してください。"
        return result_data
    # シェフ（AI）への指示書（プロンプト）を作成
    prompt = f"""
あなたは、アフリカのエネルギー問題に関する専門リサーチAIです。
以下の検索条件に基づき、信頼性の高い文献を検索し、**必ず以下のJSONフォーマット**で出力してください。
【検索条件】
ユーザー入力: {user_input}
- 対象: 査読済み論文、国際機関レポート（World Bank, IEA, AfDBなど）
- 年代: 2010-2024
- 言語: 英語の文献を日本語で要約
- 件数: 3〜5件程度（デモ表示のため）
【出力するJSONフォーマット】
以下のキーを持つオブジェクトのリスト（配列）を作成してください。
[
  {{
    "title": "文献のタイトル（原題）",
    "author": "著者名",
    "year": "発行年",
    "source_url": "DOIまたはURL",
    "summary": "日本語での要約（200文字程度）",
    "tags": {{
      "country": "国名",
      "energy_type": "エネルギーの種類",
      "topic": "分野"
    }}
  }}
]
【注意】
- Markdownのコードブロック（```json ... ```）は含めず、純粋なJSONデータのみを出力してください。
- 捏造せず、実在する文献のみを挙げてください。
"""
    # --- 粘り強いリトライ処理（最大3回） ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 1. モデルと設定を決める
            model_name, use_json_mode = get_best_model()
            generation_config = {}
            if use_json_mode:
                generation_config = {"response_mime_type": "application/json"}
            # 2. シェフを呼び出す
            # (app.pyですでにAPIキーの設定が終わっている前提で動きます)
            model = genai.GenerativeModel(
                model_name,
                generation_config=generation_config
            )
            # 3. 料理（生成）開始
            response = model.generate_content(prompt)
            text_response = response.text
            # 4. お弁当箱を開けて中身を整理（不要な記号を削除）
            clean_text = re.sub(r"```json|```", "", text_response).strip()
            # 生のデータも保存しておく（デバッグ用）
            result_data["raw_json"] = clean_text
            # JSONとして読み込んで、結果リストに入れる
            parsed_data = json.loads(clean_text)
            result_data["results"] = parsed_data
            # 成功したらエラーなしとしてループを抜ける
            result_data["error_message"] = ""
            break
        except Exception as e:
            error_str = str(e)
            # 混雑している場合 (429やQuota)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 10
                    print(f"混雑中... {wait_time}秒待機します ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
            # モデルが見つからない場合 (404) -> 予備の1.5モデルに切り替え
            elif "404" in error_str or "not found" in error_str.lower():
                try:
                    print("2.5 Flashが見つからないため、1.5 Flashで再試行します...")
                    model_fallback = genai.GenerativeModel('gemini-1.5-flash')
                    response = model_fallback.generate_content(prompt)
                    text_response = response.text
                    clean_text = re.sub(r"```json|```", "", text_response).strip()
                    result_data["raw_json"] = clean_text
                    result_data["results"] = json.loads(clean_text)
                    result_data["error_message"] = ""
                    break
                except Exception as e2:
                    result_data["error_message"] = f"予備モデルでもエラー: {e2}"
                    break
            # それ以外のエラー
            else:
                result_data["error_message"] = f"エラーが発生しました: {e}"
                break
    # 最終的な結果セットを返す
    return result_data