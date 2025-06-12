from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, Type, Union
import os
import requests

try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None
try:
    from mistralai import Mistral as MistralClient
except ImportError:
    MistralClient = None
    ChatMessage = None
try:
    from cerebras.cloud.sdk import Cerebras 
except ImportError:
    Cerebras = None 

try:
    import cohere
except ImportError:
    cohere = None
try:
    from groq import Groq as GroqClient
except ImportError:
    GroqClient = None
try:
    from together import Together as TogetherClient
except ImportError:
    TogetherClient = None
try:
    from PIL import Image 
except ImportError:
    Image = None

class Model(ABC):
    def __init__(self,
                 model_id: str,
                 name: str,
                 provider_name: str,
                 api_key: str,
                 modalities: List[str],
                 params: Optional[Dict[str, Any]] = None):
        self.model_id = model_id
        self.name = name
        self.provider_name = provider_name
        self.api_key = api_key
        self.modalities = modalities
        self.params = params if params is not None else {}
        self._client: Any = None
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        pass

    @abstractmethod
    def inference(self, prompt: Any, **kwargs) -> Any:
        pass

    @abstractmethod
    def _list_api_models(self) -> List[Dict[str, Any]]:
        pass

    def __str__(self):
        key_display = "Provided"
        nim_like_providers = ["nvidia_nim", "nvidia_api"] 
        if self.provider_name not in nim_like_providers and self.api_key and len(self.api_key) > 4:
            key_display = f"...{self.api_key[-4:]}"
        elif self.api_key and self.provider_name in nim_like_providers:
            key_display = self.api_key if len(self.api_key) < 30 else f"{self.api_key[:15]}..."

        return (f"Model(Name='{self.name}', API_ID='{self.model_id}', Provider='{self.provider_name}', "
                f"Modalities={self.modalities}, Key/URL='{key_display}', Defaults={self.params})")


class OpenAIModel(Model):
    def _initialize_client(self):
        if OpenAIClient is None: self._client = None; return
        try: self._client = OpenAIClient(api_key=self.api_key)
        except Exception as e: print(f"Error initializing OpenAI client for '{self.name}': {e}"); self._client = None

    def inference(self, prompt: Any, **kwargs) -> Any:
        if not self._client: raise RuntimeError(f"OpenAI client for '{self.name}' not initialized.")
        api_call_params = {**self.params, **kwargs}
        if "dall-e" in self.model_id:
            if not isinstance(prompt, str): raise ValueError("DALL-E prompt must be a string.")
            dalle_params = {k: v for k, v in api_call_params.items() if k in ['n', 'size', 'response_format', 'quality', 'style']}
            dalle_params.setdefault('n', 1); dalle_params.setdefault('size', "1024x1024")
            try:
                response = self._client.images.generate(model=self.model_id, prompt=prompt, **dalle_params)
                return [img.url for img in response.data if hasattr(img, 'url')] if response.data else response.data
            except Exception as e: return f"Error DALL-E ({self.name}): {e}"
        
        messages: List[Dict[str, Any]] = []
        if isinstance(prompt, str): messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt): messages = prompt
        elif isinstance(prompt, dict) and "role" in prompt and "content" in prompt: messages = [prompt]
        elif isinstance(prompt, dict) and "text" in prompt and "image_url" in prompt and \
             any(m in self.modalities for m in ["image", "vision"]) and \
             ("vision" in self.model_id or "gpt-4" in self.model_id or "o1" in self.model_id or "o4" in self.model_id):
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt["text"]}, {"type": "image_url", "image_url": {"url": prompt["image_url"]}}]}]
        else: raise ValueError(f"Unsupported prompt for OpenAI {self.name}: {type(prompt)}")
        
        gpt_call_params = {k:v for k,v in api_call_params.items() if k not in ['size', 'quality', 'style', 'response_format']}
        try:
            completion = self._client.chat.completions.create(model=self.model_id, messages=messages, **gpt_call_params)
            return completion.choices[0].message.content
        except Exception as e: return f"Error OpenAI GPT ({self.name}): {e}"

    def _list_api_models(self) -> List[Dict[str, Any]]:
        if OpenAIClient is None: raise RuntimeError("OpenAI SDK not installed.")
        if not self._client: self._initialize_client()
        if not self._client: raise RuntimeError("OpenAI client failed to initialize for listing models.")
        try:
            models_list = self._client.models.list()
            return [{"id": model.id, "owned_by": model.owned_by, "created_at": model.created} for model in models_list.data]
        except Exception as e: print(f"Error listing OpenAI models: {e}"); return []


class GeminiModel(Model):
    def _initialize_client(self):
        if genai is None: self._client = None; return
        try: self._client = genai.GenerativeModel(self.model_id)
        except Exception as e: print(f"Error initializing Gemini client for '{self.name}': {e}"); self._client = None

    def inference(self, prompt: Any, **kwargs) -> Any:
        if not self._client: raise RuntimeError(f"Gemini client for '{self.name}' not initialized.")
        all_params = {**self.params, **kwargs}
        gen_config_keys = ["candidate_count", "stop_sequences", "max_output_tokens", "temperature", "top_p", "top_k"]
        gen_config_params = {k: v for k, v in all_params.items() if k in gen_config_keys}
        safety_settings = all_params.get("safety_settings")
        try:
            gen_config = genai.types.GenerationConfig(**gen_config_params) if gen_config_params else None
            response = self._client.generate_content(prompt, generation_config=gen_config, safety_settings=safety_settings)
            if hasattr(response, 'text'): return response.text
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return f"Blocked: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, "text"))
            return response
        except Exception as e: return f"Error Gemini ({self.name}): {getattr(e, 'message', str(e))}"

    def _list_api_models(self) -> List[Dict[str, Any]]:
        if genai is None: raise RuntimeError("Google GenAI SDK not installed.")
        try:
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods or 'createTunedModel' in m.supported_generation_methods:
                    models.append({"id": m.name, "display_name": m.display_name, "description": m.description,
                                   "version": m.version, "supported_generation_methods": m.supported_generation_methods})
            return models
        except Exception as e: print(f"Error listing Gemini models: {e}"); return []


class OpenRouterModel(Model):
    def _initialize_client(self):
        if not self.api_key: raise ValueError("OpenRouter API key required.")
        self._client = requests.Session()
        self._client.headers.update({"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})

    def inference(self, prompt: Any, **kwargs) -> Any:
        if not self._client: raise RuntimeError(f"OpenRouter client for '{self.name}' not initialized.")
        messages: List[Dict[str, Any]] = []
        if isinstance(prompt, str): messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt): messages = prompt
        else: raise ValueError(f"Unsupported prompt for OpenRouter {self.name}: {type(prompt)}")
        payload = {"model": self.model_id, "messages": messages, **{**self.params, **kwargs}}
        try:
            response = self._client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, timeout=payload.get("timeout", 120))
            response.raise_for_status()
            result = response.json()
            if result.get('choices') and result['choices'][0].get('message'): return result['choices'][0]['message']['content']
            if result.get('error'): return f"OpenRouter API Error: {result['error'].get('message', str(result['error']))}"
            return f"Unexpected OpenRouter response: {result}"
        except requests.exceptions.HTTPError as e:
            try: error_msg = e.response.json().get("error", {}).get("message", e.response.text)
            except ValueError: error_msg = e.response.text
            return f"HTTP Error OpenRouter ({self.name}): {e.response.status_code} - {error_msg}"
        except Exception as e: return f"Error OpenRouter ({self.name}): {e}"

    def _list_api_models(self) -> List[Dict[str, Any]]:
        if not self._client: self._initialize_client()
        if not self._client: raise RuntimeError("OpenRouter client failed to initialize for listing.")
        try:
            response = self._client.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e: print(f"Error listing OpenRouter models: {e}"); return []


class NvidiaNimModel(Model):
    def _initialize_client(self):
        if not self.api_key: raise ValueError("NVIDIA NIM base URL (as api_key) required.")
        self._client = requests.Session()
        self._client.headers.update({"Content-Type": "application/json"})

    def inference(self, prompt: Any, **kwargs) -> Any:
        if not self._client: raise RuntimeError(f"NVIDIA NIM client for '{self.name}' not initialized.")
        base_url = self.api_key
        chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"
        messages: List[Dict[str, Any]] = []
        if isinstance(prompt, str): messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt): messages = prompt
        else: raise ValueError(f"Unsupported prompt for NVIDIA NIM {self.name}: {type(prompt)}")
        payload = {"model": self.model_id, "messages": messages, **{**self.params, **kwargs}}
        try:
            response = self._client.post(chat_url, json=payload, timeout=payload.get("timeout", 180))
            response.raise_for_status()
            result = response.json()
            if result.get('choices') and result['choices'][0].get('message'): return result['choices'][0]['message']['content']
            if result.get('error'): return f"NIM API Error: {result['error'].get('message', str(result['error']))}"
            return f"Unexpected NIM response: {result}"
        except requests.exceptions.HTTPError as e:
            try: error_msg = e.response.json().get("error", {}).get("message", e.response.text)
            except ValueError: error_msg = e.response.text
            return f"HTTP Error NIM ({self.name}): {e.response.status_code} - {error_msg}"
        except Exception as e: return f"Error NIM ({self.name}): {e}"

    def _list_api_models(self) -> List[Dict[str, Any]]:
        if not self._client: self._initialize_client()
        if not self._client: raise RuntimeError("NVIDIA NIM client failed to initialize for listing.")
        base_url = self.api_key
        models_url = f"{base_url.rstrip('/')}/v1/models"
        try:
            response = self._client.get(models_url)
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e: print(f"Error listing NVIDIA NIM models from {models_url}: {e}"); return []


class MistralModel(Model): 
    def _initialize_client(self):
        if MistralClient is None: self._client = None; return
        try: self._client = MistralClient(api_key=self.api_key)
        except Exception as e: print(f"Error initializing Mistral client for '{self.name}': {e}"); self._client = None

    def inference(self, prompt: Any, **kwargs) -> Any:
        if not self._client: raise RuntimeError(f"Mistral client for '{self.name}' not initialized.")
        messages_for_api: List[Dict[str, str]]
        if isinstance(prompt, str):
            messages_for_api = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt):
            messages_for_api = [{"role": str(p_msg["role"]), "content": str(p_msg["content"])} for p_msg in prompt]
        else:
            raise ValueError(f"Unsupported prompt type for Mistral {self.name}. Expected str or List[Dict[str, Any]].")
        all_params = {**self.params, **kwargs}
        if 'model' in all_params: del all_params['model']
        try:
            response_obj = self._client.chat.complete( 
                model=self.model_id,
                messages=messages_for_api,
                **all_params
            )

            return response_obj.choices[0].message.content
        except Exception as e: return f"Error Mistral ({self.name}): {e}"

    def _list_api_models(self) -> List[Dict[str, Any]]:
        if MistralClient is None: raise RuntimeError("Mistral SDK (Mistral class) not imported/defined.")
        if not self._client: self._initialize_client()
        if not self._client: raise RuntimeError("Mistral client failed to initialize for listing models.")
        try:
            models_list_response = self._client.list_models()
            if hasattr(models_list_response, 'data'):
                return [
                    {"id": model.id, "owned_by": model.owned_by, "created_at": model.created}
                    for model in models_list_response.data
                    if hasattr(model, 'id') and hasattr(model, 'owned_by') and hasattr(model, 'created')
                ]
            elif isinstance(models_list_response, list):
                 return [
                    {"id": model.id if hasattr(model, 'id') else model.get('id'),
                     "owned_by": model.owned_by if hasattr(model, 'owned_by') else model.get('owned_by'),
                     "created_at": model.created if hasattr(model, 'created') else model.get('created')}
                    for model in models_list_response # type: ignore
                ] # type: ignore
            else: return []
        except Exception as e:
            print(f"Error listing Mistral models: {e}")
            return []


class CohereModel(Model):
    def _initialize_client(self):
        if cohere is None: self._client = None; return
        try: self._client = cohere.Client(self.api_key, client_name="multimodel_interface")
        except Exception as e: print(f"Error initializing Cohere client for '{self.name}': {e}"); self._client = None

    def inference(self, prompt: Any, **kwargs) -> Any:
        if not self._client: raise RuntimeError(f"Cohere client for '{self.name}' not initialized.")
        message: str = ""; chat_history: Optional[List[Dict[str, str]]] = None
        if isinstance(prompt, str): message = prompt
        elif isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt):
            cohere_history = [{"role": "USER" if msg["role"].lower() == "user" else "CHATBOT", "message": msg["content"]} for msg in prompt[:-1]]
            chat_history = cohere_history if cohere_history else None
            message = prompt[-1]["content"] if prompt else ""
            if prompt and prompt[-1]["role"].lower() != "user": print(f"Warning: Last message for Cohere should be 'user'. Found: {prompt[-1]['role']}")
        else: raise ValueError(f"Unsupported prompt type for Cohere {self.name}")
        all_params = {**self.params, **kwargs}
        if "temperature" in all_params and all_params["temperature"] is None: del all_params["temperature"]
        try:
            response = self._client.chat(model=self.model_id, message=message, chat_history=chat_history, **all_params)
            return response.text
        except Exception as e: return f"Error Cohere ({self.name}): {e}"

    def _list_api_models(self) -> List[Dict[str, Any]]:
        if cohere is None: raise RuntimeError("Cohere SDK not installed.")
        print("Cohere API does not provide a simple list of all base chat models via SDK. Returning known chat models. Check Cohere documentation.")
        known_cohere_chat_models = [
            {"id": "command-r", "name": "Command R", "context_length": 128000, "endpoints": ["chat"]},
            {"id": "command-r-plus", "name": "Command R+", "context_length": 128000, "endpoints": ["chat"]},
            {"id": "command", "name": "Command", "context_length": 4096, "endpoints": ["chat"]},
            {"id": "command-light", "name": "Command Light", "context_length": 4096, "endpoints": ["chat"]},
        ]
        return [m for m in known_cohere_chat_models if "chat" in m.get("endpoints", [])]


class NvidiaApiModel(Model):
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    def _initialize_client(self):
        if not self.api_key: raise ValueError("NVIDIA API Key required.")
        self._client = requests.Session()
        self._client.headers.update({"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"})

    def inference(self, prompt: Any, **kwargs) -> Any:
        if not self._client: raise RuntimeError(f"NVIDIA API client for '{self.name}' not initialized.")
        messages: List[Dict[str, Any]] = []
        if isinstance(prompt, str): messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt): messages = prompt
        else: raise ValueError(f"Unsupported prompt for NVIDIA API {self.name}")
        all_params = {**self.params, **kwargs}
        if "max_output_tokens" in all_params:
            all_params["max_tokens"] = all_params.pop("max_output_tokens")
        
        payload = {"model": self.model_id, "messages": messages, **all_params}
        chat_url = f"{self.BASE_URL}/chat/completions"
        try:
            response = self._client.post(chat_url, json=payload, timeout=payload.get("timeout", 180))
            response.raise_for_status()
            result = response.json()
            if result.get('choices') and result['choices'][0].get('message'): return result['choices'][0]['message']['content']
            if result.get('error'): return f"NVIDIA API Error: {result['error'].get('message', str(result['error']))}"
            return f"Unexpected NVIDIA API response: {result}"
        except requests.exceptions.HTTPError as e:
            try: error_msg = e.response.json().get("error", {}).get("message", e.response.text)
            except ValueError: error_msg = e.response.text
            return f"HTTP Error NVIDIA API ({self.name} @ {chat_url}): {e.response.status_code} - {error_msg}"
        except Exception as e: return f"Error NVIDIA API ({self.name}): {e}"

    def _list_api_models(self) -> List[Dict[str, Any]]:
        if not self._client: self._initialize_client()
        if not self._client: raise RuntimeError("NVIDIA API client failed to initialize for listing.")
        models_url = f"{self.BASE_URL}/models"
        try:
            response = self._client.get(models_url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list): return data["data"]
            elif isinstance(data, list): return data
            else: print(f"Unexpected format from NVIDIA API {models_url}: {data}"); return []
        except Exception as e: print(f"Error listing NVIDIA API models from {models_url}: {e}"); return []


class CerebrasModel(Model):

    def _initialize_client(self):
        if Cerebras is None:
            self._client = None
            print(f"Warning: Cerebras SDK (cerebras-cloud-sdk) not installed. Cannot initialize Cerebras client for '{self.name}'. Please install with 'pip install cerebras-cloud-sdk'.")
            return
        try:
            self._client = Cerebras(api_key=self.api_key)
        except Exception as e:
            print(f"Error initializing Cerebras client for '{self.name}': {e}")
            self._client = None

    def inference(self, prompt: Any, **kwargs) -> Any:
        if not self._client:
            raise RuntimeError(f"Cerebras client for '{self.name}' not initialized.")

        messages: List[Dict[str, Any]] = []
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt):
            messages = prompt
        elif isinstance(prompt, dict) and "role" in prompt and "content" in prompt:
            messages = [prompt] # Allow single message dict as prompt
        else:
            raise ValueError(f"Unsupported prompt for Cerebras {self.name}: {type(prompt)}")

        api_call_params = {**self.params, **kwargs}
        if "timeout" in api_call_params: 
            del api_call_params["timeout"] 

        try:
            chat_completion = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **api_call_params
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error Cerebras ({self.name}): {e}"

    def _list_api_models(self) -> List[Dict[str, Any]]:
        if Cerebras is None:
            raise RuntimeError("Cerebras SDK not installed. Cannot list models.")
        if not self._client:
            self._initialize_client()
        if not self._client:
            raise RuntimeError("Cerebras client failed to initialize for listing models.")
        try:
            models_list_response = self._client.models.list()
            
            if hasattr(models_list_response, 'data') and isinstance(models_list_response.data, list):
                return [
                    {"id": model.id, "owned_by": model.owned_by, "created_at": model.created}
                    for model in models_list_response.data 
                    if hasattr(model, 'id') and hasattr(model, 'owned_by') and hasattr(model, 'created')
                ]
            elif isinstance(models_list_response, list):
                return [
                    {"id": model.id if hasattr(model, 'id') else model.get('id'),
                     "owned_by": model.owned_by if hasattr(model, 'owned_by') else model.get('owned_by'),
                     "created_at": model.created if hasattr(model, 'created') else model.get('created')}
                    for model in models_list_response 
                ]
            else:
                print(f"Unexpected format from Cerebras models.list(): {models_list_response}")
                return []
        except Exception as e:
            print(f"Error listing Cerebras models: {e}")
            return []


class GroqModel(Model):
    def _initialize_client(self):
        if GroqClient is None: self._client = None; return
        try: self._client = GroqClient(api_key=self.api_key)
        except Exception as e: print(f"Error initializing Groq client for '{self.name}': {e}"); self._client = None

    def inference(self, prompt: Any, **kwargs) -> Any:
        if not self._client: raise RuntimeError(f"Groq client for '{self.name}' not initialized.")
        messages: List[Dict[str, str]] = []
        if isinstance(prompt, str): messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt):
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in prompt]
        else: raise ValueError(f"Unsupported prompt type for Groq {self.name}")
        all_params = {**self.params, **kwargs}
        try:
            response = self._client.chat.completions.create(model=self.model_id, messages=messages, **all_params)
            return response.choices[0].message.content
        except Exception as e: return f"Error Groq ({self.name}): {e}"

    def _list_api_models(self) -> List[Dict[str, Any]]:
        if GroqClient is None: raise RuntimeError("Groq SDK not installed.")
        if not self._client: self._initialize_client()
        if not self._client: raise RuntimeError("Groq client failed to initialize for listing models.")
        try:
            models_list = self._client.models.list()
            return [{"id": model.id, "owned_by": model.owned_by, "created_at": model.created, "active": getattr(model, "active", None)} for model in models_list.data]
        except Exception as e: print(f"Error listing Groq models: {e}"); return []


class TogetherModel(Model):
    def _initialize_client(self):
        if TogetherClient is None: self._client = None; return
        try: self._client = TogetherClient(api_key=self.api_key)
        except Exception as e: print(f"Error initializing Together client for '{self.name}': {e}"); self._client = None

    def inference(self, prompt: Any, **kwargs) -> Any:
        if not self._client: raise RuntimeError(f"Together client for '{self.name}' not initialized.")
        messages: List[Dict[str, str]] = []
        if isinstance(prompt, str): messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt):
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in prompt]
        else: raise ValueError(f"Unsupported prompt type for Together {self.name}")
        all_params = {**self.params, **kwargs}
        try:
            response = self._client.chat.completions.create(model=self.model_id, messages=messages, **all_params)
            return response.choices[0].message.content
        except Exception as e: return f"Error Together ({self.name}): {e}"

    def _list_api_models(self) -> List[Dict[str, Any]]:
        if TogetherClient is None: raise RuntimeError("Together SDK not installed.")
        if not self._client: self._initialize_client()
        if not self._client: raise RuntimeError("Together client failed to initialize for listing models.")
        try:
            models_raw = self._client.models.list()
            models_list = []
            for model_data_item in models_raw:
                model_data = {}
                if hasattr(model_data_item, 'model_dump'): model_data = model_data_item.model_dump()
                elif isinstance(model_data_item, dict): model_data = model_data_item
                else: model_data = {"name": str(model_data_item)}
                models_list.append({"id": model_data.get("name"), "display_name": model_data.get("display_name"), "type": model_data.get("type"), "context_length": model_data.get("context_length"), "raw_api_details": model_data})
            return models_list
        except Exception as e: print(f"Error listing Together models: {e}"); return []


class CloudflareModel(Model):

    BASE_URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai"

    def _initialize_client(self):

        self.account_id = self.params.get("account_id")
        if not self.account_id:

            for k in self.params:
                if "cloudflare_account_id" in k.lower() or "cf_account_id" in k.lower():
                    self.account_id = self.params[k]
                    break
        if not self.account_id:

            raise ValueError("Cloudflare Account ID required for Cloudflare models. Set 'CLOUDFLARE_ACCOUNT_ID' environment variable or pass 'account_id' in model parameters.")

        self._client = requests.Session()

        self._client.headers.update({"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})

    def inference(self, prompt: Any, **kwargs) -> Any:
        if not self._client:
            raise RuntimeError(f"Cloudflare client for '{self.name}' not initialized.")
        if not self.account_id:
            raise RuntimeError(f"Cloudflare account ID not set for '{self.name}'. Cannot perform inference.")

        messages: List[Dict[str, Any]] = []
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt):
            messages = prompt
        elif isinstance(prompt, dict) and "role" in prompt and "content" in prompt:
            messages = [prompt] 
        else:
            raise ValueError(f"Unsupported prompt for Cloudflare {self.name}")

        inference_url = f"{self.BASE_URL_TEMPLATE.format(account_id=self.account_id)}/run/{self.model_id}"

        payload_params = {k:v for k,v in {**self.params, **kwargs}.items() if k not in ["account_id"]}
        payload = {"messages": messages, **payload_params}

        try:

            response = self._client.post(inference_url, json=payload, timeout=payload.get("timeout", 120))
            response.raise_for_status() 
            result = response.json()

            if result.get("success") and "result" in result and "response" in result["result"]:
                return result["result"]["response"]

            if not result.get("success") or result.get("errors"):
                error_msgs = [err.get("message", "Unknown error") for err in result.get("errors", [])]
                return f"Cloudflare API Error ({self.name}): {'; '.join(error_msgs) or 'Request not successful.'} - Full Response: {result}"

            return f"Unexpected Cloudflare response structure: {result}"

        except requests.exceptions.HTTPError as e:
            try:

                error_msg = e.response.json().get("errors", [{}])[0].get("message", e.response.text)
            except (ValueError, IndexError, TypeError):
                error_msg = e.response.text 
            return f"HTTP Error Cloudflare ({self.name}): {e.response.status_code} - {error_msg}"
        except Exception as e:
            return f"Error Cloudflare ({self.name}): {e}"

    def _list_api_models(self) -> List[Dict[str, Any]]:
        if not self._client:
            self._initialize_client() 
        if not self._client:
            raise RuntimeError("Cloudflare client failed to initialize for listing models.")
        if not self.account_id:
            raise RuntimeError(f"Cloudflare account ID not set for '{self.name}'. Cannot list models.")

        models_url = f"{self.BASE_URL_TEMPLATE.format(account_id=self.account_id)}/models"

        try:
            response = self._client.get(models_url)
            response.raise_for_status()
            data = response.json()

            if data.get("success") and "result" in data and isinstance(data["result"], list):

                return [{"id": m.get("id"), "name": m.get("name"), "task": m.get("task"), "raw_api_details": m} for m in data["result"]]

            print(f"Unexpected format from Cloudflare {models_url}: {data}"); return []
        except Exception as e:
            print(f"Error listing Cloudflare models from {models_url}: {e}"); return []



PROVIDER_CLASS_MAP: Dict[str, Type[Model]] = {
    "openai": OpenAIModel, "gemini": GeminiModel, "google": GeminiModel,
    "openrouter": OpenRouterModel, "nvidia": NvidiaNimModel,
    "mistral": MistralModel, "cohere": CohereModel, "nvidia_api": NvidiaApiModel,
    "cerebras": CerebrasModel, "groq": GroqModel, "together": TogetherModel,
    "cloudflare": CloudflareModel,
}

def _generate_friendly_name_for_openrouter(api_id: str) -> str:
    parts = api_id.split('/')
    name_part = parts[-1].replace('-', ' ').replace('_', ' ').title()
    provider_prefix = parts[0].replace('-', ' ').replace('_', ' ').title() if len(parts) > 1 else ""
    if provider_prefix and not name_part.lower().startswith(provider_prefix.lower().split(' ')[0]):
        return f"{provider_prefix} {name_part}".strip()
    return name_part.strip()

_openrouter_model_api_ids_raw = """
"""
_openrouter_models_processed_dict = {}
if _openrouter_model_api_ids_raw.strip() and _openrouter_model_api_ids_raw.strip() != "# PASTE YOUR LONG LIST OF OPENROUTER MODEL IDs HERE IF YOU WANT STATIC FALLBACKS":
    for _api_id_line in _openrouter_model_api_ids_raw.strip().split('\n'):
        _api_id = _api_id_line.strip()
        if not _api_id: continue
        _modalities = ["text"]
        if any(tag in _api_id.lower() for tag in ["vision", "-vl-", "pixtral", "grok-vision", "kimi-vl"]): _modalities.append("image")
        _openrouter_models_processed_dict[_api_id] = {"user_friendly_name": _generate_friendly_name_for_openrouter(_api_id), "modalities": _modalities, "default_params": {}}

KNOWN_MODELS_INFO: Dict[str, Dict[str, Dict[str, Any]]] = {
    "gemini": {
        "models/gemini-1.5-pro-latest": {"user_friendly_name": "Gemini 1.5 Pro", "modalities": ["text", "image", "video", "audio", "pdf"], "default_params": {"temperature": 0.7}},
        "models/gemini-1.5-flash-latest": {"user_friendly_name": "Gemini 1.5 Flash", "modalities": ["text", "image", "video", "audio", "pdf"], "default_params": {}},
        "gemini-pro": {"user_friendly_name": "Gemini 1.0 Pro (Text)", "modalities": ["text"], "default_params": {}},
    },"openai": {
        "gpt-4o": {"user_friendly_name": "GPT-4o", "modalities": ["text", "image"], "default_params": {"temperature": 0.5}},
        "gpt-4-turbo": {"user_friendly_name": "GPT-4 Turbo", "modalities": ["text", "image"], "default_params": {"temperature": 0.5}},
        "gpt-3.5-turbo": {"user_friendly_name": "GPT-3.5 Turbo", "modalities": ["text"], "default_params": {}},
        "dall-e-3": {"user_friendly_name": "DALL-E 3", "modalities": ["image_generation"], "default_params": {"size": "1024x1024", "n": 1}},
    },"openrouter": _openrouter_models_processed_dict, "nvidia_nim": {},
    "mistral": {
        "mistral-tiny": {"user_friendly_name": "Mistral Tiny", "modalities": ["text"]}, "mistral-small-latest": {"user_friendly_name": "Mistral Small", "modalities": ["text"]},
        "mistral-medium-latest": {"user_friendly_name": "Mistral Medium", "modalities": ["text"]}, "mistral-large-latest": {"user_friendly_name": "Mistral Large", "modalities": ["text"]},
        "open-mistral-7b": {"user_friendly_name": "Open Mistral 7B", "modalities": ["text"]}, "open-mixtral-8x7b": {"user_friendly_name": "Open Mixtral 8x7B", "modalities": ["text"]},
    },"cohere": {
        "command-r": {"user_friendly_name": "Cohere Command R", "modalities": ["text", "tool_use"], "default_params": {"temperature": 0.3}},
        "command-r-plus": {"user_friendly_name": "Cohere Command R+", "modalities": ["text", "tool_use", "image"], "default_params": {"temperature": 0.3}},
        "command": {"user_friendly_name": "Cohere Command", "modalities": ["text"]}, "command-light": {"user_friendly_name": "Cohere Command Light", "modalities": ["text"]},
    },"nvidia_api": {
        "playground_llama2_13b": {"user_friendly_name": "NVIDIA Playground Llama2 13B", "modalities": ["text"]},
    },"cerebras": {
        "llama-3.1-8b": {"user_friendly_name": "Cerebras Llama 3.1 8B", "modalities": ["text"]},
        "llama-3.3-70b": {"user_friendly_name": "Cerebras Llama 3.3 70B", "modalities": ["text"]},
        "llama-4-scout-17b-16e-instruct": {"user_friendly_name": "Cerebras Llama 4 16B", "modalities": ["text"]},
        "qwen-3-32b": {"user_friendly_name": "Cerebras Qwen 32B", "modalities": ["text"]},
    },"groq": {
        "llama3-8b-8192": {"user_friendly_name": "Groq Llama3 8B", "modalities": ["text"]}, "llama3-70b-8192": {"user_friendly_name": "Groq Llama3 70B", "modalities": ["text"]},
        "mixtral-8x7b-32768": {"user_friendly_name": "Groq Mixtral 8x7B", "modalities": ["text"]}, "gemma-7b-it": {"user_friendly_name": "Groq Gemma 7B IT", "modalities": ["text"]},
    },"together": {
        "togethercomputer/llama-2-7b-chat": {"user_friendly_name": "Together Llama-2 7B Chat", "modalities": ["text"]},
        "mistralai/Mixtral-8x7B-Instruct-v0.1": {"user_friendly_name": "Together Mixtral 8x7B Instruct", "modalities": ["text"]},
        "Qwen/Qwen1.5-72B-Chat": {"user_friendly_name": "Together Qwen1.5 72B Chat", "modalities": ["text"]},
    },"cloudflare": {
        "@cf/meta/llama-2-7b-chat-fp16": {"user_friendly_name": "Cloudflare Llama-2 7B Chat FP16", "modalities": ["text"]},
        "@cf/mistral/mistral-7b-instruct-v0.1": {"user_friendly_name": "Cloudflare Mistral 7B Instruct", "modalities": ["text"]},
        "@cf/google/gemma-7b-it": {"user_friendly_name": "Cloudflare Google Gemma 7B IT", "modalities": ["text"]},
    }
}

def _get_api_key_for_provider(provider_name: str) -> str:
    p_name_lower = provider_name.lower()
    env_vars = {
        "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "google": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY", "nvidia_nim": "NVIDIA_NIM_BASE_URL",
        "mistral": "MISTRAL_API_KEY", "cohere": "COHERE_API_KEY", "nvidia_api": "NVIDIA_API_KEY",
        "cerebras": "CEREBRAS_API_KEY", "groq": "GROQ_API_KEY", "together": "TOGETHER_API_KEY",
        "cloudflare": "CLOUDFLARE_API_KEY",
    }
    key_env_var = env_vars.get(p_name_lower)
    api_key_val = os.getenv(key_env_var) if key_env_var else None
    if not api_key_val and p_name_lower in ["gemini", "google"]:
        api_key_val = os.getenv("GOOGLE_API_KEY")
        if api_key_val: key_env_var = "GOOGLE_API_KEY"
    if not api_key_val:
        msg = f"API Key/Base URL for '{provider_name}' not found. "
        msg += f"Set env var '{key_env_var}' or pass directly." if key_env_var else "Provider not configured for API key env var."
        raise ValueError(msg)
    return api_key_val

def _get_additional_config_for_provider(provider_name: str) -> Dict[str, Any]:
    config = {}; p_name_lower = provider_name.lower()
    if p_name_lower == "cloudflare":
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if account_id: config["account_id"] = account_id
        else: print(f"Warning: CLOUDFLARE_ACCOUNT_ID env var not set. Required for Cloudflare.")
    return config

def _create_specific_model_instance( provider_name: str, model_api_id: str, api_key: str,
    user_friendly_name_override: Optional[str] = None, modalities_override: Optional[List[str]] = None,
    params_override: Optional[Dict[str, Any]] = None) -> Model:
    p_name_lower = provider_name.lower()
    model_class = PROVIDER_CLASS_MAP.get(p_name_lower)
    if not model_class: raise ValueError(f"Unsupported provider: {provider_name}")
    known_info_key = "gemini" if p_name_lower == "google" else p_name_lower
    model_defaults = KNOWN_MODELS_INFO.get(known_info_key, {}).get(model_api_id, {})
    name = user_friendly_name_override or model_defaults.get("user_friendly_name")
    if not name:
        if p_name_lower == "openrouter": name = _generate_friendly_name_for_openrouter(model_api_id)
        elif p_name_lower == "nvidia_nim": name = f"NIM {model_api_id}"
        elif p_name_lower == "cloudflare": name = f"Cloudflare {model_api_id}"
        else: name = model_api_id
    modalities = modalities_override or model_defaults.get("modalities", ["text"])
    final_constructor_params = {**model_defaults.get("default_params", {}), **(params_override or {})}
    return model_class(model_id=model_api_id, name=name, provider_name=p_name_lower, api_key=api_key, modalities=modalities, params=final_constructor_params)


class ProviderInterface:
    def __init__(self, provider_name: str, api_key_or_url: str, additional_config: Optional[Dict[str, Any]] = None):
        self.provider_name = provider_name.lower()
        self.api_key_or_url = api_key_or_url
        self.additional_config = additional_config if additional_config is not None else {}
        self._model_class: Type[Model] = PROVIDER_CLASS_MAP.get(self.provider_name)
        if not self._model_class: raise ValueError(f"Unsupported provider: '{self.provider_name}'")
        if self.provider_name in ["gemini", "google"] and genai:
            try: genai.configure(api_key=self.api_key_or_url)
            except Exception as e: print(f"Warning: Could not configure Google GenAI SDK: {e}")
        try:
            lister_params = {**self.additional_config}
            self._lister_instance: Model = self._model_class(model_id="api-lister-placeholder", name="APILister", provider_name=self.provider_name, api_key=self.api_key_or_url, modalities=[], params=lister_params)
            if self._lister_instance._client is None and self.provider_name not in ["gemini", "google", "nvidia_nim", "cohere"]:
                print(f"Warning: Lister client for {self.provider_name} is None. API listing may fail.")
        except Exception as e: print(f"Error creating API lister instance for {self.provider_name}: {e}"); self._lister_instance = None

    def list_models(self) -> List[Dict[str, Any]]:
        if not self._lister_instance: print(f"Cannot list models for {self.provider_name}: Lister not created."); return []
        try: api_models_raw = self._lister_instance._list_api_models()
        except Exception as e: print(f"Failed to list models from {self.provider_name} API: {e}"); api_models_raw = []
        augmented = []; known_info_key = "gemini" if self.provider_name == "google" else self.provider_name
        provider_known_info = KNOWN_MODELS_INFO.get(known_info_key, {})
        for raw_info in api_models_raw:
            api_id = raw_info.get("id")
            if not api_id: continue
            known_specific = provider_known_info.get(api_id, {})
            name = known_specific.get("user_friendly_name")
            if not name:
                if self.provider_name == "openrouter": name = _generate_friendly_name_for_openrouter(api_id)
                elif self.provider_name == "nvidia_nim": name = f"NIM {raw_info.get('name', api_id)}"
                elif self.provider_name == "nvidia_api": name = f"NVIDIA API {raw_info.get('name', api_id)}"
                elif self.provider_name == "cloudflare": name = raw_info.get("name", api_id)
                elif self.provider_name == "cohere": name = raw_info.get("name", api_id)
                elif self.provider_name in ["gemini", "google"]: name = raw_info.get("display_name", api_id)
                elif raw_info.get("display_name"): name = raw_info.get("display_name")
                else: name = api_id
            modalities = known_specific.get("modalities")
            if not modalities:
                modalities = ["text"]
                if self.provider_name == "openrouter" and any(tag in api_id.lower() for tag in ["vision", "-vl-", "pixtral"]): modalities.append("image")
                if self.provider_name == "cloudflare" and "vision" in raw_info.get("task", {}).get("type", ""): modalities.append("image")
                if self.provider_name == "cohere" and "command-r-plus" in api_id: modalities.append("image")
            current_default_params = {**self.additional_config, **known_specific.get("default_params", {})}
            augmented.append({"provider": self.provider_name, "api_model_id": api_id, "user_friendly_name": name, "modalities": modalities, "default_params": current_default_params, "raw_api_details": raw_info})
        augmented.sort(key=lambda x: x["user_friendly_name"])
        return augmented

    def inference(self, model_name: str, prompt: Any, api_key_override: Optional[str] = None, additional_config_override: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        current_api_key = api_key_override if api_key_override is not None else self.api_key_or_url
        final_additional_config = {**self.additional_config, **(additional_config_override or {})}
        known_info_key = "gemini" if self.provider_name == "google" else self.provider_name
        model_defaults = KNOWN_MODELS_INFO.get(known_info_key, {}).get(model_name, {})
        constructor_params = {**model_defaults.get("default_params", {}), **final_additional_config}
        model_instance = _create_specific_model_instance(provider_name=self.provider_name, model_api_id=model_name, api_key=current_api_key, params_override=constructor_params)
        return model_instance.inference(prompt, **kwargs)


def get_provider(provider_name: str, api_key_or_url: Optional[str] = None, additional_config_override: Optional[Dict[str, Any]] = None) -> ProviderInterface:
    key_val = api_key_or_url if api_key_or_url is not None else _get_api_key_for_provider(provider_name)
    final_additional_config = _get_additional_config_for_provider(provider_name)
    if additional_config_override is not None: final_additional_config.update(additional_config_override)
    return ProviderInterface(provider_name, key_val, additional_config=final_additional_config)


def list_known_models_static(provider_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    models_list = []; target_providers = KNOWN_MODELS_INFO
    if provider_filter:
        key = "gemini" if provider_filter.lower() == "google" else provider_filter.lower()
        if key in KNOWN_MODELS_INFO: target_providers = {key: KNOWN_MODELS_INFO[key]}
        else: print(f"Provider filter '{provider_filter}' not in KNOWN_MODELS_INFO."); return []
    for p_key, p_models in target_providers.items():
        for api_id, details in p_models.items():
            models_list.append({"provider": p_key, "api_model_id": api_id, "user_friendly_name": details.get("user_friendly_name", api_id), "modalities": details.get("modalities", ["unknown"]), "default_params": details.get("default_params", {})})
    return models_list


class MultiModelInterface:
    def __init__(self, provider_configs: Dict[str, Dict[str, Any]]):
        self.providers: Dict[str, ProviderInterface] = {}
        self.all_available_models_cache: Optional[List[Dict[str, Any]]] = None
        for provider_name, config_dict in provider_configs.items():
            p_name_lower = provider_name.lower()
            api_key_val = config_dict.get("api_key")
            additional_conf_override = {k: v for k, v in config_dict.items() if k != "api_key"}
            try:
                self.providers[p_name_lower] = get_provider(p_name_lower, api_key_or_url=api_key_val, additional_config_override=additional_conf_override)
                print(f"Successfully configured provider: {p_name_lower}")
            except Exception as e: print(f"Warning: Could not configure provider '{p_name_lower}': {e}")

    def list_all_models(self, refresh: bool = False, provider_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if not refresh and self.all_available_models_cache is not None:
            if provider_filter: return [m for m in self.all_available_models_cache if m['provider'] == provider_filter.lower()]
            return self.all_available_models_cache
        all_models = []; providers_to_query = {}
        if provider_filter:
            p_name_lower = provider_filter.lower()
            if p_name_lower in self.providers: providers_to_query = {p_name_lower: self.providers[p_name_lower]}
            else: print(f"Provider filter '{provider_filter}' not configured or found."); return []
        else: providers_to_query = self.providers
        for provider_name, provider_interface in providers_to_query.items():
            try:
                print(f"Listing models for {provider_name}...")
                models = provider_interface.list_models()
                all_models.extend(models)
            except Exception as e: print(f"Error listing models for {provider_name}: {e}")
        all_models.sort(key=lambda x: (x["provider"], x["user_friendly_name"]))
        if not provider_filter: self.all_available_models_cache = all_models
        return all_models

    def find_models(self, modality: Optional[str] = None, provider: Optional[str] = None, name_contains: Optional[str] = None, refresh_list: bool = False) -> List[Dict[str, Any]]:
        models = self.list_all_models(refresh=refresh_list, provider_filter=provider)
        if modality: models = [m for m in models if modality.lower() in [mod.lower() for mod in m.get("modalities", [])]]
        if name_contains: models = [m for m in models if name_contains.lower() in m.get("user_friendly_name", "").lower()]
        return models

    def get_model_instance(self, provider_name: str, model_api_id: str, api_key_override: Optional[str] = None, additional_config_override: Optional[Dict[str, Any]] = None, **default_params_override) -> Optional[Model]:
        p_name_lower = provider_name.lower()
        if p_name_lower not in self.providers: print(f"Provider '{provider_name}' not configured."); return None
        provider_interface = self.providers[p_name_lower]
        current_api_key = api_key_override if api_key_override is not None else provider_interface.api_key_or_url
        final_additional_config = {**provider_interface.additional_config, **(additional_config_override or {})}
        params_for_constructor = {**final_additional_config, **default_params_override}
        try:
            return _create_specific_model_instance(provider_name=p_name_lower, model_api_id=model_api_id, api_key=current_api_key, params_override=params_for_constructor)
        except Exception as e: print(f"Error creating model instance for {p_name_lower}/{model_api_id}: {e}"); return None

    def inference(self, provider_name: str, model_api_id: str, prompt: Any, api_key: Optional[str] = None, additional_provider_config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        p_name_lower = provider_name.lower()
        if p_name_lower not in self.providers: return f"Error: Provider '{provider_name}' not configured."
        provider_interface = self.providers[p_name_lower]
        try:
            return provider_interface.inference(model_api_id, prompt, api_key_override=api_key, additional_config_override=additional_provider_config, **kwargs)
        except Exception as e: return f"Error during inference with {p_name_lower}/{model_api_id}: {e}"

    def __str__(self):
        return f"MultiModelInterface(Configured Providers: {list(self.providers.keys())})"


if __name__ == "__main__":
    print("--- Initializing MultiModelInterface ---")
    initial_provider_setups = {
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")}, "gemini": {"api_key": os.getenv("GEMINI_API_KEY")},
        "openrouter": {"api_key": os.getenv("OPENROUTER_API_KEY")}, "mistral": {"api_key": os.getenv("MISTRAL_API_KEY")},
        "cohere": {"api_key": os.getenv("COHERE_API_KEY")}, "groq": {"api_key": os.getenv("GROQ_API_KEY")},
        "together": {"api_key": os.getenv("TOGETHER_API_KEY")},
        "cloudflare": {"api_key": os.getenv("CLOUDFLARE_API_KEY"), "account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID")},
        "nvidia_api": {"api_key": os.getenv("NVIDIA_API_KEY")},
        # "cerebras": {"api_key": os.getenv("CEREBRAS_API_KEY")},
        # "nvidia_nim": {"api_key": "http://localhost:8000"}
    }
    active_provider_configs = {}
    for p_name, p_config in initial_provider_setups.items():
        key_present = False
        if p_config.get("api_key"): key_present = True
        else:
            try: _get_api_key_for_provider(p_name); key_present = True
            except ValueError: key_present = False
        if p_name == "cloudflare" and key_present:
            if not p_config.get("account_id") and not os.getenv("CLOUDFLARE_ACCOUNT_ID"):
                print(f"Skipping Cloudflare: API key found but CLOUDFLARE_ACCOUNT_ID not set/provided."); key_present = False
        if key_present: active_provider_configs[p_name] = p_config
        else: print(f"Skipping provider '{p_name}': API key/URL (and account_id for Cloudflare) not found.")

    if not active_provider_configs:
        print("\nNo provider API keys/URLs found. Set environment variables. Skipping examples.")
    else:
        print(f"\nAttempting to configure MultiModelInterface with: {list(active_provider_configs.keys())}")
        multi_interface = MultiModelInterface(active_provider_configs)
        print(multi_interface)

        print("\n--- Listing all available models (refresh=True to force API call) ---")
        all_models = multi_interface.list_all_models(refresh=False)
        if all_models:
            print(f"Total models found across configured providers: {len(all_models)}")
            models_by_provider = {}
            for m in all_models: models_by_provider.setdefault(m['provider'], []).append(m)
            for prov, prov_models in models_by_provider.items():
                print(f"\n--- Models from {prov} ({len(prov_models)} found) ---")
                for i, model_data in enumerate(prov_models[:3]): print(f"  Name: {model_data['user_friendly_name']}, API ID: {model_data['api_model_id']}, Modalities: {model_data['modalities']}")
                if len(prov_models) > 3: print(f"  ... and {len(prov_models)-3} more.")
        else: print("No models found.")

        if "openai" in multi_interface.providers:
            print("\n--- OpenAI GPT-3.5 Turbo Inference ---")
            gpt_models = multi_interface.find_models(provider="openai", name_contains="GPT-3.5 Turbo")
            if gpt_models:
                print(f"OpenAI Response: {multi_interface.inference('openai', gpt_models[0]['api_model_id'], 'Translate hello to French.')}")
            else: print("GPT-3.5 Turbo model not found.")

        if "mistral" in multi_interface.providers:
            print("\n--- Mistral Inference (Small) ---")
            mistral_models = multi_interface.find_models(provider="mistral", name_contains="Small")
            if mistral_models:
                print(f"Mistral Response: {multi_interface.inference('mistral', mistral_models[0]['api_model_id'], 'What is the Mistral wind?')}")
            else: print("Mistral Small model not found.")

        if "groq" in multi_interface.providers:
            print("\n--- Groq Inference (Llama3 8B) ---")
            groq_models = multi_interface.find_models(provider="groq", name_contains="Llama3 8B")
            if groq_models:
                # Example of passing API key directly (though usually configured at init)
                # custom_groq_key = "gsk_YOUR_GROQ_KEY_HERE" # Replace with a real key if testing this
                # print(f"Groq Response: {multi_interface.inference('groq', groq_models[0]['api_model_id'], 'How fast is Groq?', api_key=custom_groq_key)}")
                print(f"Groq Response: {multi_interface.inference('groq', groq_models[0]['api_model_id'], 'How fast is Groq?')}")

            else: print("Groq Llama3 8B model not found.")

        if "cloudflare" in multi_interface.providers:
            print("\n--- Cloudflare Inference (Llama-2 7B) ---")
            cf_models = multi_interface.find_models(provider="cloudflare", name_contains="Llama-2-7B")
            if cf_models:
                # Example of passing additional config directly
                # custom_cf_config = {"account_id": "YOUR_CLOUDFLARE_ACCOUNT_ID_HERE"} # Replace if testing this
                # print(f"Cloudflare Response: {multi_interface.inference('cloudflare', cf_models[0]['api_model_id'], 'What is Cloudflare Workers AI?', additional_provider_config=custom_cf_config)}")
                print(f"Cloudflare Response: {multi_interface.inference('cloudflare', cf_models[0]['api_model_id'], 'What is Cloudflare Workers AI?')}")
            else: print("Cloudflare Llama-2 7B model not found.")
        
        if "cohere" in multi_interface.providers:
            print("\n--- Cohere Command R+ Instance Example ---")
            cohere_r_plus_models = multi_interface.find_models(provider="cohere", name_contains="Command R+")
            if cohere_r_plus_models:
                cohere_instance = multi_interface.get_model_instance("cohere", cohere_r_plus_models[0]['api_model_id'], temperature=0.1)
                if cohere_instance:
                    print(f"Got instance: {cohere_instance}")
                    print(f"Cohere Instance Response: {cohere_instance.inference('Explain RAG in LLMs.', max_tokens=150)}")
                else: print("Could not get Cohere R+ instance.")
            else: print("Cohere Command R+ model not found.")