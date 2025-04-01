from semantic_router import Route
from googlesearch import search
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import _Settings
from llama_index.core import Settings

Settings.chunk_size = 256

def web_search_handler(query: str, settings:_Settings) -> str:
    """Web search for find information about the query

    :param query: The query to search for web
    :type query: str
    :return: A string with the result of the search."""
    results = list(search(query, num_results=3))
    
    if not results:
        return ""
    
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=results)
    index = VectorStoreIndex.from_documents(documents=documents)
    retriever = index.as_retriever(similarity_top_k=1)
    retrieved_nodes = retriever.retrieve(query)
    context = "\n".join(node.text for node in retrieved_nodes if node.score > 0.5)
    return context

web_search = Route(
    name="web_search",
    utterances=[
        "Dame información actualizada sobre lo que pasó en la semana",
        "que me puedes decir sobre este tema?",
        "Quiero saber sobre este servicio que ",
        "hableme sobre el proyecto",
        "que me dices sobre esta noticia",
        "¿Puedes buscar información sobre",
        "¿Puedes encontrar información sobre",
        "¿Puedes buscar algo sobre",
        "Quiero que busques",
        "¿Podrías buscar información acerca de",
        "Encuentra datos sobre",
        "Busca detalles relacionados con",
        "Me gustaría saber más sobre",
        "Investiga sobre ",
        "Proporciona información acerca de ",
        "¿Qué puedes decirme sobre el tema",
        "Necesito información referente a",
    ],
    metadata={ "func": 'web_search_handler' },
)

