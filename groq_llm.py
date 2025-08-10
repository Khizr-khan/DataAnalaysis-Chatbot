import os
import requests
from pandasai.llm.base import LLM

def _resolve_groq_key(explicit_key=None):
    # 1) explicit
    if explicit_key and str(explicit_key).strip():
        return str(explicit_key).strip()

    # 2) Streamlit secrets (if available)
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            k = st.secrets["GROQ_API_KEY"]
            if k and str(k).strip():
                return str(k).strip()
    except Exception:
        pass

    # 3) Environment variable
    k = os.getenv("GROQ_API_KEY")
    if k and str(k).strip():
        return str(k).strip()

    return None

class GroqLLM(LLM):
    """
    Minimal PandasAI-compatible wrapper for Groq's Chat Completions API
    (OpenAI-compatible format).
    """

    def __init__(self, model="llama-3.1-8b-instant", api_key=None, base_url="https://api.groq.com/openai/v1"):
        self.model = model
        self.api_key = _resolve_groq_key(api_key)
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it in Streamlit Secrets or export it as an env var."
            )
        super().__init__()

    @property
    def type(self) -> str:
        return "groq"

    def call(self, instruction, suffix=""):
        # PandasAI gives an Instruction; turn it into a single user message
        prompt = instruction.to_string() + suffix

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": False,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
