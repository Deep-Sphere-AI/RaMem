from semantic_router import Route

def web_search_handler(query: str):
    """Web search for find information about the query

    :param query: The query to search for web
    :type query: str
    :return: A string with the result of the search."""
    return f"Buscando información sobre: {query}"

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

