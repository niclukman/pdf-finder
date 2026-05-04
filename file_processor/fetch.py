import psycopg, os

from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from pgvector.psycopg import register_vector

load_dotenv()

def main():
    DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
    conn = psycopg.connect(DB_CONNECTION_STRING)
    register_vector(conn)

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    query = ["test"]
    query_embedding = model.encode(query)["dense_vecs"][0]

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, embedding FROM test ORDER BY embedding <=> %s LIMIT 5",
            (query_embedding,)
        )

        results = cur.fetchall()
        for row in results:
            print(row)

    conn.close()

main()
