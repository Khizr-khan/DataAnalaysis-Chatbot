import requests
from pandasai.llm.base import LLM

class OllamaLLM(LLM):
    def __init__(self, model="mistral", api_base="http://localhost:11434"):
        self.model = model
        self.api_base = api_base
        super().__init__()

    @property
    def type(self) -> str:
        return "ollama"  # ✅ required by PandasAI

    def call(self, instruction, suffix=""):
        prompt = instruction.to_string() + suffix
        response = requests.post(
            f"{self.api_base}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json()["response"]
