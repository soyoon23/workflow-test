"""Registry for observability providers."""

import logging
from typing import Optional

from .base import ObservabilityProvider

logger = logging.getLogger(__name__)


class ObservabilityRegistry:
    """Registry for managing observability providers.

    Follows the same dict-based registry pattern as ToolRegistry,
    SkillRegistry, and PromptRegistry.
    """

    def __init__(self):
        self._providers: dict[str, ObservabilityProvider] = {}
        self._active_provider: Optional[str] = None
        self._load_default_providers()

    def _load_default_providers(self) -> None:
        """Load built-in providers with graceful import handling."""
        from .providers.noop_provider import NoopProvider

        self.register(NoopProvider())

        try:
            from .providers.langfuse_provider import LangfuseProvider

            self.register(LangfuseProvider())
        except ImportError:
            logger.debug("Langfuse provider not available (SDK not installed)")

    def register(self, provider: ObservabilityProvider) -> None:
        """Register a provider."""
        self._providers[provider.name] = provider
        logger.debug("Registered observability provider: %s", provider.name)

    def unregister(self, name: str) -> None:
        """Unregister a provider."""
        self._providers.pop(name, None)
        if self._active_provider == name:
            self._active_provider = None

    def get(self, name: str) -> Optional[ObservabilityProvider]:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_available(self) -> list[str]:
        """Get names of all providers whose SDKs are installed."""
        return [name for name, p in self._providers.items() if p.is_available()]

    def set_active(self, name: str) -> None:
        """Set the active provider by name."""
        if name not in self._providers:
            raise ValueError(f"Unknown observability provider: {name}")
        self._active_provider = name

    def get_active(self) -> Optional[ObservabilityProvider]:
        """Get the currently active provider (or None)."""
        if self._active_provider:
            return self._providers.get(self._active_provider)
        return None

    def resolve_provider(self, config: dict) -> Optional[ObservabilityProvider]:
        """Resolve the provider to use from config.

        Reads config['observability']['provider'] first, then falls back
        to legacy config['langfuse']['enabled'] for backward compatibility.
        Returns None if observability is disabled.
        """
        obs_cfg = config.get("observability", {})
        provider_name = obs_cfg.get("provider")

        if provider_name:
            if provider_name == "noop":
                return None
            provider = self.get(provider_name)
            if provider and provider.is_available():
                return provider
            logger.warning(
                "Requested provider '%s' not available, skipping observability",
                provider_name,
            )
            return None

        # Backward compatible: check legacy langfuse config
        langfuse_cfg = config.get("langfuse", {})
        if langfuse_cfg.get("enabled", False):
            provider = self.get("langfuse")
            if provider and provider.is_available():
                return provider

        return None
