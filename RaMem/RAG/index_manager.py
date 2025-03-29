from llama_index.core import QueryBundle
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    load_index_from_storage
)
from llama_index.core import Settings
import os

class IndexManager:
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        if os.path.exists(persist_dir):
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            self.index = VectorStoreIndex([])
            os.makedirs(persist_dir, exist_ok=True)

    def get_relevant_context(self, query: str, top_k=2) -> str:
        query_embedding = Settings.embed_model.get_text_embedding(query)
        query = QueryBundle(query_str=query, embedding=query_embedding)
        results = self.index.as_retriever().retrieve(query)[:top_k]
        context = [res.metadata for res in results if res.score > 0.53]
        return context

    def add_qa_to_index(self, question: str, answer: str):
        qa_text = f"{question}\n:{answer}"
        document = Document(text=qa_text, metadata= {"question": question, "answer": answer})
        self.index.insert(document)
        self.index.storage_context.persist(self.persist_dir)