from unsloth import FastLanguageModel
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .RAG.index_manager import IndexManager
from .RAG.routes.routes import RouterModel
from llama_index.llms.huggingface import HuggingFaceLLM

class RaMemIntegratedModel:
    def __init__(self, model_name: str = "DeepSphere-AI/base-RaMem-LoRA", max_seq_length=4096, persist_dir: str = "./memory"):
        # Cargar el modelo y el tokenizador
        self.max_seq_length = max_seq_length
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            output_hidden_states=True,
        )
        # Configurar el modelo para inferencia rápida
        FastLanguageModel.for_inference(self.model)
        
        # Inicializar el modelo de embeddings para el índice
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") #sentence-transformers/paraphrase-multilingual-mpnet-base-v2
        
        Settings.embed_model = self.embed_model
        Settings.llm = HuggingFaceLLM(
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
        )

        # Configurar el IndexManager
        self.index_manager = IndexManager(persist_dir=persist_dir)
        # Historial inicial del sistema
        self._history = [
            {"role": "system", "content": "Eres un asistente en español llamado RaMem y ayudas respondiendo con la mayor exactitud posible."}
        ]

        self.router = RouterModel(generate_func=self.chat_completion, default_func=self._generate_response, setting=Settings)

    def generate(self, prompt: str, **kwargs) -> str:
        # Añadir el prompt del usuario al historial
        self._history.append({"role": "user", "content": prompt})

        response = self.router(prompt, **kwargs)
        self._update_index_and_hsitory(prompt, response)
        return response

    def chat_completion(self, messages: str, **kwargs) -> str:
        """No usa el historial para generar la respuesta
        """
        if len(messages) > 10:
            messages = messages[-9:]
        
        messages = [{
            'role': 'system',
            'content': 'Eres un asistente en español llamado RaMem y ayudas respondiendo con la mayor exactitud posible.'
        }] + messages

        message = self._convert_history_to_message(messages)
        response = self._generate(message, **kwargs)
        
        return response

    def _generate_response(self, prompt: str, **kwargs) -> str:
        """Funcion que toma el texto ingresado por el usuario y genera una respuesta
            en conjunto con el contexto relevante recuperado. El prompt inicial es pasado al chat_template.
            Usa el historial para generar la respuesta en caso no haya contexto relevante

        :params prompt: prompt del usaurio
        :params **kwargs: otros parametros para la generacion de respuesta
        :type prompt: str
        :return: Respuesta generada por el modelo."""

        # Obtener contexto relevante del índice
        context = self.index_manager.get_relevant_context(prompt)

        # Construir el input para el modelo
        if not context:
            input_prompt = self._convert_history_to_message(self._history)
        else:
            input_prompt = self._convert_to_chat_with_context(prompt, context)

        return self._generate(input_prompt, **kwargs)
    
    def _update_index_and_hsitory(self, prompt:str, response:str):
        """Funcion que actualiza el indice con la query del usuario y el historial con la respuesta del modelo.

        :params prompt: prompt del usuario
        :params response: respuesta del modelo
        :type prompt: str
        :type response: str
        :return: None
        """
        # Actualizar el índice con la pregunta y respuesta
        self.index_manager.add_qa_to_index(prompt, response)
        
        # Añadir la respuesta al historial
        self._history.append({"role": "assistant", "content": response})
        
    def _generate(self, format_prompt, **kwargs) -> str:
        """Funcion que genera una respuesta en base al prompt formateado con el chat_template

        :params format_prompt: prompt formateado con chat_template aplicado  
        :type format_prompt: str
        :return: respuesta generada por el modelo"""

        # Tokenizar el input
        inputs = self.tokenizer(format_prompt, return_tensors="pt").to("cuda")

        # Generar la respuesta
        output = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 1024),
            use_cache=True,
            temperature=kwargs.get("temperature", 1.0),
            min_p=kwargs.get("min_p", 0.1)
        )

        # Decodificar la respuesta
        response:str = self.tokenizer.decode(output[0], skip_special_tokens=True)
        only_response = response.split("assistant")[-1].strip().rstrip("<|eot_id|>").lstrip('<|end_header_id|>\n\n')
        
        return only_response

    def _convert_history_to_message(self, messages) -> str:
        if messages[0]['role'] == 'system':
            messages.insert(0, {
                'role': 'system',
                'content': 'Eres un asistente en español llamado RaMem y ayudas respondiendo con la mayor exactitud posible.'
            })
        return self.tokenizer.apply_chat_template(messages[-10:], tokenize=False)

    def _convert_to_chat_with_context(self, input_text:str, context:str) -> str:
        conv_context = [{
            'role': 'system',
            'content': 'Eres un asistente en español llamado RaMem y ayudas respondiendo con la mayor exactitud posible.'
        }]

        for ctx in context:
            conv_user = { 'role': 'user', 'content': ctx.get('question') }
            conv_asis = { 'role': 'assitant', 'content': ctx.get('answer') }
            conv_context.extend([conv_user, conv_asis])

        conv_context.append({
            'role': 'user',
            'content': input_text
        })
        
        return self.tokenizer.apply_chat_template(conv_context, tokenize=False)