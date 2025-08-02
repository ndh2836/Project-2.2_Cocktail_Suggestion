import psycopg2

DB_HOST = "localhost"
DB_NAME = "mydatabase"
DB_USER = "myuser"
DB_PASSWORD = "2836"

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    print("Kết nối database thành công!")
    conn.close()
except psycopg2.Error as e:
    print(f"Lỗi khi kết nối database: {e}")