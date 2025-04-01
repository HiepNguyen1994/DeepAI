import streamlit as st
from query_engine import get_answer

st.set_page_config(page_title="🧠 Trợ lý ngân hàng FAISS", page_icon="📊")

st.title("💬 Bot hỗ trợ phân tích ngành ngân hàng")
st.markdown("Nhập câu hỏi để truy vấn vào tài liệu ngân hàng đã indexing:")

# Inputs thân thiện hơn
doc_name = st.text_input("📁 Chủ đề (keywords chính, ví dụ: CTG, VPB, Fulbright...)")

time_filter = st.selectbox(
    "📅 Bộ lọc thời gian",
    ["mới nhất", "tất cả", "Q1-2024", "Q4-2023", "2024", "2023"]
)

source_options = ["tất cả", "VDSC", "HSC", "Vietcap", "Fulbright"]
sources = st.multiselect("🏦 Nguồn tài liệu", options=source_options, default=["tất cả"])

user_prompt = st.text_area("📝 Yêu cầu chi tiết (prompt):")

# Nút tìm kiếm
if st.button("🔍 Tìm kiếm"):
    if not doc_name or not user_prompt:
        st.warning("⚠️ Vui lòng nhập đủ thông tin chủ đề và prompt.")
    else:
        with st.spinner("⏳ Đang tìm kiếm và tổng hợp..."):
            try:
                print("[DEBUG] Query:", doc_name, time_filter, sources, user_prompt)
                answer = get_answer(doc_name, time_filter, sources, user_prompt)

                st.markdown(f"### 📌 Kết quả truy vấn cho **{doc_name}** ({', '.join(sources)} - {time_filter}):")
                st.markdown(answer)

            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi: {e}")
