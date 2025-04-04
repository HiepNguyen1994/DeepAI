### Query kết hợp BM25 

import os
import json
import faiss
import numpy as np
import openai
import re
from rank_bm25 import BM25Okapi
from datetime import datetime
from datetime import datetime, timedelta
import streamlit as st

openai.api_key = st.secrets["api"]["openai_key"]




# === FAISS News Handler ===
# def get_faiss_news_answer(query_text: str, start_date=None, end_date=None):
#     # === Setup ===
#     base_folder = os.getcwd()
#     articles_folder = os.path.join(base_folder, "articles_by_week")
#     faiss_index_path = os.path.join(base_folder, "faiss_banking_news.bin")
#     index = faiss.read_index(faiss_index_path)

#     # === Tự động tạo ngày nếu chưa truyền ===
#     if not start_date or not end_date:
#         today = datetime.today()
#         start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
#         end_date = today.strftime("%Y-%m-%d")

#     start_dt = datetime.strptime(start_date, "%Y-%m-%d")
#     end_dt = datetime.strptime(end_date, "%Y-%m-%d")

#     doc_name = query_text.strip()
def get_faiss_news_answer(query_text: str, start_date=None, end_date=None):
    # === Setup ===
    base_folder = os.getcwd()
    # Gỡ articles_folder – KHÔNG DÙNG THƯ MỤC NỮA
    # articles_folder = os.path.join(base_folder, "articles_by_week")

    faiss_index_path = os.path.join(base_folder, "faiss_banking_news.bin")
    index = faiss.read_index(faiss_index_path)

    # === Tự động tạo ngày nếu chưa truyền ===
    if not start_date or not end_date:
        today = datetime.today()
        start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    doc_name = query_text.strip()

    # === Load dữ liệu từ file articles (gộp sẵn)
    articles_path = os.path.join(base_folder, "articles_2025-03-17.json")  # hoặc đổi tên lại thành articles.json
    with open(articles_path, "r", encoding="utf-8") as f:
        articles_data = json.load(f)

    # === Lọc theo ngày
    filtered_articles = {
        url: art for url, art in articles_data.items()
        if start_dt <= datetime.strptime(art["date"], "%Y-%m-%d") <= end_dt
    }

    # === Phần xử lý tiếp (nối vào phần index.search, get_embedding, v.v.)
    return filtered_articles


    # === Dùng GPT để trích keyword cho BM25 ===
    def extract_keywords_by_gpt(prompt):
        try:
            # Nếu truy vấn ngắn, dùng luôn
            if len(prompt.strip().split()) <= 2:
                return [prompt.strip().lower()]
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Bạn là hệ thống phân tích tin tức ngân hàng. Hãy trích ra các từ khóa chính xác từ câu hỏi người dùng để dùng cho tìm kiếm tin tức (dạng BM25), chỉ giữ lại 3–5 cụm từ quan trọng nhất. Trả về dạng list Python."},
                    {"role": "user", "content": f"Truy vấn: {prompt}"}
                ],
                max_tokens=100,
                temperature=0.2
            )
            raw = response.choices[0].message.content.strip()
            print(f"[DEBUG] GPT trả về: {raw}")
            keywords = eval(raw)
            if not isinstance(keywords, list) or not keywords:
                raise ValueError("Invalid keyword list")
            return keywords
        except Exception as e:
            print(f"[ERROR] Không parse được keyword: {e}")
            return [prompt.strip().lower()]  # fallback về nguyên câu

    keyword = extract_keywords_by_gpt(query_text)

    # === Load tất cả file JSON ===
    all_json_files = [
        os.path.join(articles_folder, week_folder, f)
        for week_folder in os.listdir(articles_folder)
        if os.path.isdir(os.path.join(articles_folder, week_folder))
        for f in os.listdir(os.path.join(articles_folder, week_folder))
        if f.endswith(".json")
    ]

    if not all_json_files:
        return "❌ Không tìm thấy file tin tức nào."

    def extract_date(article):
        try:
            return datetime.strptime(article["date"], "%Y-%m-%d")
        except:
            return datetime.min

    filtered_documents = {}
    documents_for_bm25 = []
    article_ids = []

    for json_file in all_json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for url, content in data.items():
            article_date = extract_date(content)
            if start_dt <= article_date <= end_dt:
                text = content.get("title", "") + " " + content.get("content", "")
                documents_for_bm25.append(text.split())
                article_ids.append(url)
                filtered_documents[url] = content

    if not documents_for_bm25:
        return "❌ Không có bài báo nào trong khoảng thời gian đã chọn."

    bm25 = BM25Okapi(documents_for_bm25)
    bm25_scores = bm25.get_scores(" ".join(keyword).split())
    top_n = 200
    bm25_results = sorted(zip(article_ids, bm25_scores), key=lambda x: x[1], reverse=True)[:top_n]

    final_bm25_results = []
    for url, score in bm25_results:
        content = filtered_documents[url]["content"]
        if any(re.search(r"\b" + re.escape(kw) + r"\b", content, re.IGNORECASE) for kw in keyword):
            final_bm25_results.append((url, score))

    if not final_bm25_results:
        return f"❌ Không có bài nào chứa từ khóa: {keyword}"

    filtered_documents = {url: filtered_documents[url] for url, score in final_bm25_results}
    bm25_article_ids = list(filtered_documents.keys())

    def search_faiss_news(keywords, bm25_article_ids, k=100):
        query_vector = np.array(
            openai.embeddings.create(model="text-embedding-3-large", input=" ".join(keywords)).data[0].embedding,
            dtype=np.float32
        ).reshape(1, -1)
        D, I = index.search(query_vector, k=k)

        results = []
        for idx in I[0]:
            if idx >= len(filtered_documents):
                continue
            article_url = list(filtered_documents.keys())[idx]
            if article_url in bm25_article_ids:
                article = filtered_documents[article_url]
                results.append({
                    "url": article_url,
                    "content": article["content"],
                    "date": article["date"]
                })
        return sorted(results, key=lambda x: extract_date(x), reverse=True)[:20]

    faiss_results = search_faiss_news(doc_name, bm25_article_ids, k=100)
    # faiss_results = search_faiss_news(keyword, bm25_article_ids, k=70)
    if not faiss_results:
        faiss_results = [
            {"url": url, "content": filtered_documents[url]["content"], "date": filtered_documents[url]["date"]}
            for url in bm25_article_ids[:5]
        ]

    context = "\n\n".join([
        f"📌 [{res['date']}] {res['url']}\n{res['content'][:10000]}..."
        for res in faiss_results
    ])

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia phân tích ngân hàng, phân tích chi tiết nhất có thể, dài khoảng 500 chữ. "
                    "Hãy trả lời đúng vào truy vấn của người dùng bên dưới, chỉ dựa trên nội dung được cung cấp. "
                    "Nếu không đủ dữ liệu, hãy nói rõ thay vì bịa đặt. Câu trả lời nên logic, có thể trích dẫn mốc thời gian hoặc số liệu nếu có."
                ),
            },
            {
                "role": "user",
                "content": f"""📌 **Truy vấn:** {query_text.strip()}

📖 **Dữ liệu thu thập được (tin tức gần đây):**

{context}
"""
            },
        ],
        max_tokens=4000,
    )

    return response.choices[0].message.content





