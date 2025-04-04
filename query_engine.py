import os, json, faiss, numpy as np, openai, re
from datetime import datetime
import os, json, faiss, numpy as np, openai, re
from datetime import datetime
from rank_bm25 import BM25Okapi
from collections import defaultdict

openai.api_key = "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"

# query_engine.py - Đã fix fallback BM25 + FAISS + lọc thời gian


# ==== CẤU HÌNH ====
BASE_FOLDER = os.getcwd()
FAISS_INDEX_PATH = os.path.join(BASE_FOLDER, "faiss_banking_index.bin")
KEY_MAP_PATH = os.path.join(BASE_FOLDER, "faiss_key_map.json")
EMBEDDING_DIM = 3072  # openai text-embedding-3-large

# ==== LOAD DATA ====
index = faiss.read_index(FAISS_INDEX_PATH)

with open(KEY_MAP_PATH, "r", encoding="utf-8") as f:
    key_map = json.load(f)

def extract_date_from_filename(file_name):
    if not file_name:
        return None
    try:
        clean = file_name.strip().lower().replace("_", "").replace(" ", "")

        # Trường hợp: YYYYMMDD
        match = re.search(r'(\d{8})', clean)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d")

        # Trường hợp: Q1-2024 hoặc Q12024
        match = re.search(r'q([1-4])[-/]?(\d{4})', clean)
        if match:
            q, y = int(match.group(1)), int(match.group(2))
            return datetime(y, (q - 1) * 3 + 1, 1)

        # Trường hợp chỉ có năm YYYY
        match = re.search(r'(\d{4})', clean)
        if match:
            return datetime(int(match.group(1)), 1, 1)

    except Exception as e:
        print(f"[❌ EXTRACT ERROR] {file_name} → {e}")
    
    return None  # rõ ràng return None nếu không match gì cả

def filter_by_time(results, time_filter):
    if not time_filter or time_filter.lower() == "tất cả":
        return results

    # Lọc mới nhất theo ngày có thể xác định được
    if time_filter.lower() == "mới nhất":
        dated_results = [r for r in results if extract_date_from_filename(r["file_name"])]
        dated_results.sort(key=lambda x: extract_date_from_filename(x["file_name"]), reverse=True)
        return dated_results[:1] if dated_results else results[:20]

    # Lọc theo quý (Q1-2024...)
    match = re.search(r'Q([1-4])[/-](\d{4})', time_filter)
    if match:
        q, y = int(match[1]), int(match[2])
        start = datetime(y, (q - 1) * 3 + 1, 1)
        end = datetime(y, (q - 1) * 3 + 3, 28)
        filtered = [
            r for r in results 
            if (d := extract_date_from_filename(r["file_name"])) and start <= d <= end
        ]
        print(f"[DEBUG] 🎯 Lọc theo quý {time_filter} còn {len(filtered)} đoạn")
        return filtered

    # Lọc theo năm
    match = re.search(r'(\d{4})', time_filter)
    if match:
        y = int(match[1])
        filtered = [
            r for r in results 
            if (d := extract_date_from_filename(r["file_name"])) and d.year == y
        ]
        print(f"[DEBUG] 🎯 Lọc theo năm {y} còn {len(filtered)} đoạn")
        return filtered

    # Lọc theo ngày cụ thể YYYY-MM-DD
    try:
        target = datetime.strptime(time_filter, "%Y-%m-%d")
        filtered = [
            r for r in results 
            if extract_date_from_filename(r["file_name"]) == target
        ]
        print(f"[DEBUG] 🎯 Lọc theo ngày {target.strftime('%Y-%m-%d')} còn {len(filtered)} đoạn")
        return filtered
    except:
        print("[DEBUG] ⚠️ Không nhận diện được thời gian, giữ nguyên toàn bộ")
        return results


def get_bm25_indices(doc_name, sources, top_n=2000):
    contents = []
    valid_indices = []

    for i, item in enumerate(key_map):
        file_name = item["file_name"].lower()
        source_match = (
            "tất cả" in sources
            or any(src.lower() in file_name for src in sources)
        )

        if source_match:
            contents.append((item["content"] + " " + file_name).lower())
            valid_indices.append(i)

    if not contents:
        print("[WARN] ❗ BM25 không tìm được đoạn phù hợp.")
        return []

    tokenized = [c.split() for c in contents]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(doc_name.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [valid_indices[i] for i in top_indices]



def search_faiss(query, valid_indices, k=800):
    query_vec = openai.embeddings.create(model="text-embedding-3-large", input=[query]).data[0].embedding
    query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)
    D, I = index.search(query_vec, k)

    valid_set = set(valid_indices)
    return [key_map[idx] for idx in I[0] if idx in valid_set and idx < len(key_map)]

def smart_filter_faiss(faiss_results, doc_name, time_filter, sources):
    def match(item, doc=None, year=None, srcs=None):
        full_text = (item["file_name"] + " " + item["content"]).lower()
        doc_ok = doc.lower() in full_text if doc else True
        year_ok = year in full_text if year else True
        src_ok = any(src.lower() in full_text for src in srcs) if srcs else True
        return doc_ok and year_ok and src_ok

    match_year = re.search(r'Q[1-4][/-](\d{4})', time_filter) or re.search(r'(\d{4})', time_filter)
    year = match_year.group(1) if match_year else None
    srcs = [s for s in sources if s.lower() != "tất cả"]

    # Gom từng tầng → tránh mất HSC khi VDSC trúng trước
    results = []
    seen_files = set()

    filter_levels = [
        ("🎯 VPB + năm + nguồn", dict(doc=doc_name, year=year, srcs=srcs)),
        ("🔄 VPB + năm", dict(doc=doc_name, year=year, srcs=None)),
        ("🔄 VPB + nguồn", dict(doc=doc_name, year=None, srcs=srcs)),
        ("🔄 Nguồn + năm", dict(doc=None, year=year, srcs=srcs)),  # <-- tầng thêm vào
        ("🔄 Chỉ VPB", dict(doc=doc_name, year=None, srcs=None)),
    ]

    for label, cond in filter_levels:
        batch = [
            r for r in faiss_results
            if match(r, **cond) and r["file_name"] not in seen_files
        ]
        if batch:
            print(f"[DEBUG] {label} → {len(batch)} đoạn")
            results.extend(batch)
            seen_files.update(r["file_name"] for r in batch)

    return results


# === GPT SYSTEM PROMPT ===
SYSTEM_PROMPT = """
Bạn là chuyên viên phân tích cao cấp ngành ngân hàng.

Yêu cầu:
- Chỉ được trả lời các con số hoặc tỷ lệ nếu chúng **xuất hiện nguyên văn trong tài liệu truy xuất được từ FAISS**. 
Nếu không có số cụ thể, hãy nói rằng: "Không thấy con số cụ thể trong tài liệu".
Tuyệt đối không phỏng đoán hoặc ước lượng.
- Giọng văn xéo xắt, thông minh, cực kỳ logic, đừng để sai chính tả và không bịa nội dung. 
- Trích xuất các luận điểm rõ ràng, có dẫn chứng và số liệu cụ thể.
- Viết phong cách phân tích tài chính khách quan, mạch lạc.
- Nếu thông tin không đủ, trình bày trung thực thay vì suy đoán.
- Nếu có thể, hãy ghi rõ tên tổ chức phát hành tài liệu (ví dụ: VDSC, BSC...).
"""

# === MAIN ===
def get_answer(doc_name: str, time_filter: str, sources: list, user_prompt: str):
    print("[INFO] 🔎 Đang lọc BM25...")
    filtered_indices = get_bm25_indices(doc_name, sources)
    print(f"[DEBUG] BM25 lấy {len(filtered_indices)} đoạn")

    query_text = f"{doc_name}. {user_prompt}"
    faiss_results = search_faiss(query_text, filtered_indices)
    print(f"[DEBUG] FAISS trả về {len(faiss_results)} đoạn")

    filtered_results = smart_filter_faiss(faiss_results, doc_name, time_filter, sources)

    if not filtered_results:
        print("[WARN] ❗ smart_filter_faiss không có kết quả, fallback faiss_results")
        filtered_results = faiss_results[:10]

    # Kết hợp cả 2: đảm bảo mỗi nguồn xuất hiện ít nhất 2 đoạn (nếu có)
    filtered_results_by_source = []
    seen_files = set()


    for source in sources:
        source_filtered = [
            item for item in filtered_results
            if source.lower() in item['file_name'].lower() and item['file_name'] not in seen_files
        ]
        if source_filtered:
            selected = source_filtered[:15]  # lấy tối đa 15 đoạn từ mỗi nguồn
            filtered_results_by_source.extend(selected)
            seen_files.update(item['file_name'] for item in selected)

    # Nếu không đủ 5 đoạn, lấy thêm để đủ context
    if len(filtered_results_by_source) < 5:
        additional_results = [item for item in filtered_results if item['file_name'] not in seen_files]
        filtered_results_by_source.extend(additional_results[:5 - len(filtered_results_by_source)])

    context = "\n\n".join(
        f"Nguồn: [{next((src for src in sources if src.lower() in item['file_name'].lower()), 'Không rõ')}] - {item['file_name']}\n"
        f"Nội dung: {item['content']}"
        for item in filtered_results_by_source
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": f"""📖 **Dữ liệu FAISS:**\n\n{context}\n\n📌 **Yêu cầu:**\n{user_prompt}"""}
        ],
        max_tokens=3000,
    )

    return response.choices[0].message.content







#
#
# def get_answer(doc_name: str, time_filter: str, user_prompt: str):
#     filtered_indices = get_bm25_indices(doc_name)
#     print("[DEBUG] BM25 filtered:", len(filtered_indices))
#
#     query_text = f"{doc_name}. {user_prompt}"
#     faiss_results = search_faiss(query_text, filtered_indices, k=70)
#     print("[DEBUG] FAISS raw:", len(faiss_results))
#
#     filtered_results = force_year_filter(faiss_results, time_filter)
#     print("[DEBUG] FAISS sau năm:", len(filtered_results))
#
#     if not filtered_results:
#         print("[⚠️] FAISS lọc rỗng sau khi áp dụng thời gian, fallback all FAISS")
#         filtered_results = faiss_results
#
#     if not filtered_results:
#         return f"❌ Không tìm thấy nội dung phù hợp với truy vấn '{doc_name}'!"
#
#     context = "\n\n".join([res["content"] for res in filtered_results])[:3000]
#
#     response = openai.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT.strip()},
#             {"role": "user", "content": f"""📖 **Dữ liệu FAISS:**\n\n{context}\n\n📌 **Yêu cầu:**\n{user_prompt}"""}
#         ],
#         max_tokens=3000,
#     )
#
#     return response.choices[0].message.content





