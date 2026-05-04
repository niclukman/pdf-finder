import fitz, psycopg, os
from dotenv import load_dotenv
from glmocr import GlmOcr
from FlagEmbedding import BGEM3FlagModel
from pgvector.psycopg import register_vector

load_dotenv()

def main():
    DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
    conn = psycopg.connect(DB_CONNECTION_STRING)
    register_vector(conn)

    ocr = GlmOcr(mode="selfhosted", model="zai-org/GLM-OCR")
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    sentences = []

    pdf = fitz.open("test.pdf")
    for i in range(pdf.page_count):
        page = pdf.load_page(i)
        pix = page.get_pixmap()
        bytestream = pix.tobytes("jpg")

        result = ocr.parse(bytestream)
        sentences.append(result.to_dict()["markdown_result"])

    embeddings = model.encode(sentences)["dense_vecs"]
    for embedding in embeddings:
        conn.execute("INSERT INTO test (embedding) VALUES (%s)", (embedding,))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
