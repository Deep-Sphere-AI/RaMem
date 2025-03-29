from .web_router import web_search, web_search_handler
from .code_router import code_search, code_search_handler

from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder as SemanticEncoder
# from semantic_router.layer import RouteLayer

routes = [web_search, code_search]

mapping_handlers = {
    "web_search": web_search_handler,
    "code_search": code_search_handler,
}

class RouterModel():
    def __init__(self, generate_func, default_func): # model_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
        encoder = SemanticEncoder()
        self.ramem_router = SemanticRouter(encoder=encoder, routes=routes, auto_sync='local')
        self.generate_func = generate_func
        self.default_func = default_func

    def __call__(self, prompt, **kwargs):
        name = self.ramem_router(prompt).name
        
        if not name:
            return self.default_func(prompt, **kwargs)
        
        handler = mapping_handlers.get(f"{name}_handler")
        if handler:
            context = handler(prompt)
            messages = [{
                'role': 'user',
                'content': f'{context}\n{prompt}'
            }]
            return self.generate_func(messages, **kwargs)
        

        
