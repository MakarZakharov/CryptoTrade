import psycopg2

postgres_config = {
    "dbname": "trading",
    "user": "postgres",
    "password": "ваш_пароль",  # ← сюда введи свой пароль!
    "host": "localhost",
    "port": 5432
}

try:
    conn = psycopg2.connect(**postgres_config)
    print("✅ Успешное подключение к PostgreSQL")
    conn.close()
except Exception as e:
    print("❌ Ошибка подключения:", e)
