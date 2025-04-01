from semantic_router import Route

def code_search_handler(query: str, settings):
    return f"Buscando código sobre: {query}"

code_search = Route(
    name="code_search",
    utterances=[
        "necesito un código",
        "necesito un orograma que haga",
        "quiero escribri un código",
        "quiero escribir un programa",
    ],
)

