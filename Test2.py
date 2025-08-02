import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

# --- Cấu hình cơ sở dữ liệu ---
DB_HOST = "localhost"
DB_NAME = "mydatabase"
DB_USER = "myuser"
DB_PASSWORD = "2836"
VECTOR_DIM = 384 # Kích thước vector của mô hình Sentence-Transformer mặc định

# --- Cấu hình mô hình embedding ---
@st.cache_resource
def load_embedding_model():
    """Tải mô hình Sentence-Transformer để tạo vector embedding."""
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

# --- Các hàm tương tác với cơ sở dữ liệu ---
def get_db_connection():
    """Thiết lập kết nối với PostgreSQL."""
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    return conn

def create_database():
    """Tạo database nếu nó chưa tồn tại."""
    try:
        # Connect to the default 'postgres' database to create a new one
        conn = psycopg2.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f'CREATE DATABASE {DB_NAME}')
            st.success(f"Database '{DB_NAME}' đã được tạo thành công.")
        else:
            st.info(f"Database '{DB_NAME}' đã tồn tại.")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Lỗi khi tạo database: {e}")
        return False

def create_table():
    """Tạo bảng 'cocktails' nếu chưa tồn tại."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cocktails (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding VECTOR(%s)
            );
        """, (VECTOR_DIM,))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Lỗi khi tạo bảng: {e}")
        return False

def insert_cocktails(cocktails: List[Dict[str, str]]):
    """Chèn dữ liệu cocktail đã được nhúng vào cơ sở dữ liệu."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        for cocktail in cocktails:
            embedding = model.encode(cocktail['description'])
            cur.execute(
                "INSERT INTO cocktails (name, description, embedding) VALUES (%s, %s, %s)",
                (cocktail['name'], cocktail['description'], embedding.tolist())
            )
        conn.commit()
        cur.close()
        conn.close()
        st.success(f"Đã chèn thành công {len(cocktails)} cocktail vào cơ sở dữ liệu.")
        return True
    except Exception as e:
        st.error(f"Lỗi khi chèn dữ liệu: {e}")
        return False

def insert_cocktails_from_csv(csv_file_path: str):
    """Đọc dữ liệu từ file CSV và chèn vào database."""
    if not os.path.exists(csv_file_path):
        st.error(f"Lỗi: Không tìm thấy file '{csv_file_path}'. Vui lòng đảm bảo file tồn tại trong cùng thư mục với script.")
        return False

    try:
        df = pd.read_csv(csv_file_path)
        cocktails = df.to_dict('records') # Chuyển DataFrame thành list of dictionaries
        return insert_cocktails(cocktails)
    except Exception as e:
        st.error(f"Lỗi khi đọc file CSV: {e}")
        return False

def find_similar_cocktails(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Tìm kiếm cocktail tương tự dựa trên mô tả truy vấn của người dùng."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query_embedding = model.encode(query)
        cur.execute(
            "SELECT name, description, 1 - (embedding <=> %s) AS similarity FROM cocktails ORDER BY similarity DESC LIMIT %s",
            (query_embedding.tolist(), k)
        )
        results = cur.fetchall()
        cur.close()
        conn.close()
        return [{"name": res[0], "description": res[1], "similarity": res[2]} for res in results]
    except Exception as e:
        st.error(f"Lỗi khi tìm kiếm cocktail: {e}")
        return []

# --- Giao diện Streamlit ---
st.title("Hệ thống gợi ý Cocktail")

if create_database():
    if create_table():
        st.success("Cơ sở dữ liệu và bảng đã sẵn sàng.")

        st.header("Thêm dữ liệu từ file CSV")
        csv_file = 'final_cocktails.csv'
        if st.button(f"Tải dữ liệu từ file '{csv_file}' và thêm vào database"):
            insert_cocktails_from_csv(csv_file)

        st.header("Gợi ý Cocktail cho bạn")
        user_query = st.text_input("Hãy mô tả hương vị bạn muốn (ví dụ: 'ngọt, ít cồn, có vị trái cây'):")
        if user_query:
            similar_cocktails = find_similar_cocktails(user_query, k=3)
            if similar_cocktails:
                st.subheader("Những loại cocktail được gợi ý:")
                for cocktail in similar_cocktails:
                    st.markdown(f"**{cocktail['name']}** (Mức độ tương đồng: {cocktail['similarity']:.2f})")
                    st.write(f"Mô tả: {cocktail['description']}")
                    st.markdown("---")
            else:
                st.warning("Không tìm thấy cocktail phù hợp. Hãy thử mô tả khác.")