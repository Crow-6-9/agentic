# rag_filter.py

from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from sklearn.metrics.pairwise import cosine_similarity
import os

class PythonRAGFilter:
    def __init__(self, docs_path="python_docs.txt", threshold=0.4):
        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.threshold = threshold

        if os.path.exists("faiss_index"):
            self.db = FAISS.load_local("faiss_index", self.embedding_model)
        else:
            raw_docs = TextLoader(docs_path).load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(raw_docs)
            self.db = FAISS.from_documents(docs, self.embedding_model)
            self.db.save_local("faiss_index")

        self.retriever = self.db.as_retriever()

    def is_python_question(self, prompt: str) -> bool:
        query_embedding = self.embedding_model.embed_query(prompt)
        docs = self.retriever.get_relevant_documents(prompt)

        if not docs:
            return False

        top_doc_embedding = self.embedding_model.embed_query(docs[0].page_content)
        similarity = cosine_similarity(
            [query_embedding], [top_doc_embedding]
        )[0][0]

        return similarity >= self.threshold

    def retrieve_context(self, prompt: str, top_k=4) -> str:
        docs = self.retriever.get_relevant_documents(prompt)
        return "\n\n".join(doc.page_content for doc in docs[:top_k])
