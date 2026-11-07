"""
IBM Granite LLM Integration via watsonx.ai
Uses the modern chat completions API (/ml/v1/text/chat).
"""

import os
from dotenv import load_dotenv

load_dotenv()


class GraniteClient:
    """
    Client for IBM Granite (and compatible) LLMs via watsonx.ai.
    Uses ModelInference.chat() — the current non-deprecated API.
    """

    def __init__(self):
        self.api_key    = os.getenv("WATSONX_API_KEY", "")
        self.project_id = os.getenv("WATSONX_PROJECT_ID", "")
        self.url        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self.model_id   = os.getenv("GRANITE_MODEL_ID", "ibm/granite-4-h-small")
        self.max_tokens = int(os.getenv("MAX_NEW_TOKENS", "800"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))

        self._model       = None
        self._initialized = False
        self._init_client()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_client(self):
        """Connect to IBM watsonx.ai using the ibm-watsonx-ai SDK."""
        if not self.api_key or self.api_key == "your_ibm_cloud_api_key_here":
            print("[GraniteClient] WATSONX_API_KEY not set — running in demo mode.")
            return
        if not self.project_id or self.project_id == "your_watsonx_project_id_here":
            print("[GraniteClient] WATSONX_PROJECT_ID not set — running in demo mode.")
            return

        try:
            from ibm_watsonx_ai import APIClient, Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference

            creds = Credentials(url=self.url, api_key=self.api_key)
            client = APIClient(creds)

            self._model = ModelInference(
                model_id=self.model_id,
                api_client=client,
                project_id=self.project_id,
            )

            # Quick connectivity test using the chat endpoint
            test = self._chat("Say OK")
            if test:
                self._initialized = True
                print(f"[GraniteClient] Connected to IBM Granite: {self.model_id}")
            else:
                print(f"[GraniteClient] Model returned empty response — demo mode.")

        except ImportError:
            print("[GraniteClient] ibm-watsonx-ai not installed — demo mode.")
        except Exception as e:
            print(f"[GraniteClient] Connection error: {e} — demo mode.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _chat(self, user_message: str) -> str:
        """Call the watsonx.ai chat completions endpoint."""
        messages = [{"role": "user", "content": user_message}]
        response = self._model.chat(
            messages=messages,
            params={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        # Extract text from response
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""

    def generate(self, prompt: str) -> str:
        """
        Generate a response using IBM Granite via the chat API.
        Falls back to demo placeholder if not connected.
        """
        if self._initialized and self._model:
            try:
                result = self._chat(prompt)
                if result:
                    return result
                print("[GraniteClient] Empty response — falling back to demo.")
            except Exception as e:
                print(f"[GraniteClient] Generation error: {type(e).__name__}: {e}")

        return self._demo_response(prompt)

    # ------------------------------------------------------------------
    # Demo fallback
    # ------------------------------------------------------------------

    def _demo_response(self, prompt: str) -> str:
        """Structured placeholder shown when Granite is not connected."""
        if "substitut" in prompt.lower():
            return (
                "**Substitution Suggestions (Demo Mode):**\n"
                "• Swap the missing ingredient with a similar pantry item.\n"
                "• Maintain the same flavor profile (e.g., one acid for another).\n\n"
                "To get real AI-powered substitutions, add your IBM watsonx.ai credentials to .env."
            )
        return (
            "**Recipe Assistant — Demo Mode**\n\n"
            "RAG retrieval is working — relevant recipes have been found above.\n"
            "To enable full IBM Granite AI responses, add your credentials to .env:\n\n"
            "  WATSONX_API_KEY=your_key\n"
            "  WATSONX_PROJECT_ID=your_project_id\n\n"
            "Then restart the server with: python app.py"
        )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._initialized

    def get_model_info(self) -> dict:
        return {
            "model_id":       self.model_id,
            "url":            self.url,
            "connected":      self._initialized,
            "max_new_tokens": self.max_tokens,
            "temperature":    self.temperature,
        }
