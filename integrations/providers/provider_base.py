from __future__ import annotations
from typing import Any, Dict, List, Optional

class ProviderMixin:
    """
    Lightweight optional capabilities for embedding providers.
    - is_available(): report whether the SDK/client is present
    - _set_client_for_tests(): inject a fake client to avoid network
    - provider_name(), model_name(), stats(): small, consistent metadata
    """
    _client: Any = None
    model: Optional[str] = None

    # ---- test/mocking support ----
    def _set_client_for_tests(self, client: Any) -> None:
        """Inject a fake client; used by unit tests to avoid network."""
        self._client = client

    # ---- availability & metadata ----
    def _sdk_available(self) -> bool:
        """Implement in subclass to signal whether its SDK is importable."""
        return False

    def is_available(self) -> bool:
        """True if a client is injected or the SDK is importable."""
        return self._client is not None or self._sdk_available()

    def provider_name(self) -> str:
        n = self.__class__.__name__
        return n.replace("Embeddings", "").lower()

    def model_name(self) -> str:
        return self.model or ""

    def stats(self, texts: List[str]) -> Dict[str, Any]:
        """Return a minimal, stable shape without calling remote APIs."""
        return {
            "provider": self.provider_name(),
            "model": self.model_name(),
            "batch_size": len(texts),
            "dim": None,  # left None to avoid remote calls; tests assert keys not values
        }