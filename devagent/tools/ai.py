from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Protocol
from urllib import error, request
import json

from devagent.config.settings import ConfigManager, ProviderModelConfig

GENERATION_CAPABILITY = "generate"
EMBED_CAPABILITY = "embed"
PROVIDER_ORDER = ("gemini", "groq", "xai", "openrouter")
PROVIDER_LABELS = {"gemini": "Gemini", "groq": "Groq", "xai": "xAI", "openrouter": "OpenRouter"}
TRANSIENT_BACKOFF_SECONDS = (3.0, 5.0)
GEMINI_DEFAULT_FAST_MODEL = "gemini-2.5-flash"
GEMINI_DEFAULT_DEEP_MODEL = "gemini-2.5-pro"
GEMINI_DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
GROQ_DEFAULT_FAST_MODEL = "llama-3.1-8b-instant"
GROQ_DEFAULT_DEEP_MODEL = "llama-3.1-8b-instant"
XAI_DEFAULT_FAST_MODEL = "grok-3-mini"
OPENROUTER_DEFAULT_FAST_MODEL = "openai/gpt-4o-mini"
OPENROUTER_DEFAULT_DEEP_MODEL = "openai/gpt-4o-mini"
TRANSIENT_SERVER_ERROR = "transient_server"
QUOTA_EXHAUSTED_ERROR = "quota_exhausted"
MODEL_UNAVAILABLE_ERROR = "model_unavailable"
PROVIDER_UNAVAILABLE_ERROR = "provider_unavailable"
AUTH_OR_PERMISSION_ERROR = "auth_or_permission"
UNKNOWN_ERROR = "unknown"

_CLIENT_CACHE: dict[tuple[str, str, str | None], object] = {}
_MODEL_CACHE: dict[tuple[str, str], tuple["DiscoveredModel", ...]] = {}


@dataclass(frozen=True)
class ProviderCredentials:
    provider: str
    api_key: str
    api_source: str


@dataclass(frozen=True)
class DiscoveredModel:
    provider: str
    id: str
    label: str
    capabilities: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


@dataclass(frozen=True)
class AIProviderStatus:
    provider: str
    api_source: str | None
    selected: bool
    generation_models: int = 0
    embedding_models: int = 0
    error: str | None = None

    @property
    def label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, self.provider)


@dataclass(frozen=True)
class ProviderModelListing:
    provider: str
    models: tuple[DiscoveredModel, ...] = ()
    error: str | None = None

    @property
    def label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, self.provider)


@dataclass(frozen=True)
class AIStatusSnapshot:
    selected_provider: str | None
    fast_model: str | None
    deep_model: str | None
    embedding_model: str | None
    providers: tuple[AIProviderStatus, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedAIProfile:
    provider: str | None
    api_key: str | None
    api_source: str | None
    fast_model: str | None
    deep_model: str | None
    embedding_model: str | None
    available_providers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    text: str | None
    provider: str | None
    model: str | None
    used_deep_mode: bool
    attempts: int
    fallback_notes: tuple[str, ...] = ()
    error_kind: str | None = None
    final_error: str | None = None

    @property
    def succeeded(self) -> bool:
        return bool(self.text)


GenerationProgressCallback = Callable[[str], None]


class ProviderAdapter(Protocol):
    provider: str
    credentials: ProviderCredentials

    def list_models(self, *, refresh: bool = False) -> list[DiscoveredModel]:
        ...

    def complete(self, prompt: str, *, model: str, system_instruction: str | None = None) -> str | None:
        ...

    def embed(self, texts: Iterable[str], *, model: str) -> list[list[float]] | None:
        ...


@dataclass(frozen=True)
class AIClient:
    provider: str | None
    api_key: str | None
    api_source: str | None = None
    fast_model: str | None = GEMINI_DEFAULT_FAST_MODEL
    deep_model: str | None = GEMINI_DEFAULT_DEEP_MODEL
    embedding_model: str | None = GEMINI_DEFAULT_EMBEDDING_MODEL
    available_providers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "AIClient":
        load_dotenv_if_available()
        config = ConfigManager.load()
        credentials = resolve_available_credentials()
        profile = resolve_profile(credentials, config.ai_settings)
        return cls(
            provider=profile.provider,
            api_key=profile.api_key,
            api_source=profile.api_source,
            fast_model=profile.fast_model,
            deep_model=profile.deep_model,
            embedding_model=profile.embedding_model,
            available_providers=profile.available_providers,
            warnings=profile.warnings,
        )

    @property
    def available(self) -> bool:
        return bool(self.provider and self.api_key)

    @property
    def provider_label(self) -> str:
        if not self.provider:
            return "AI"
        return PROVIDER_LABELS.get(self.provider, self.provider)

    def selected_model(self, *, deep: bool = False) -> str | None:
        if deep and self.deep_model:
            return self.deep_model
        return self.fast_model

    def complete(self, prompt: str, *, deep: bool = False, system_instruction: str | None = None) -> str | None:
        return self.generate(
            prompt,
            deep=deep,
            system_instruction=system_instruction,
        ).text

    def generate(
        self,
        prompt: str,
        *,
        deep: bool = False,
        system_instruction: str | None = None,
        progress_callback: GenerationProgressCallback | None = None,
    ) -> GenerationResult:
        if not self.available or not self.provider:
            return GenerationResult(
                text=None,
                provider=None,
                model=None,
                used_deep_mode=deep,
                attempts=0,
                error_kind=PROVIDER_UNAVAILABLE_ERROR,
                final_error="No AI provider is configured.",
            )

        emit_generation_progress(progress_callback, "Thinking...")
        candidates = self._generation_candidates(deep=deep)
        if not candidates:
            return GenerationResult(
                text=None,
                provider=self.provider,
                model=None,
                used_deep_mode=deep,
                attempts=0,
                error_kind=MODEL_UNAVAILABLE_ERROR,
                final_error=f"No generation model is configured for {self.provider_label}.",
            )

        attempts = 0
        fallback_notes: list[str] = []
        previous_failure_kind: str | None = None
        previous_provider: str | None = None
        previous_model: str | None = None
        last_error_kind: str | None = None
        last_error: str | None = None

        for index, (provider, model, candidate_deep) in enumerate(candidates):
            if index:
                note = describe_generation_fallback(
                    from_provider=previous_provider,
                    from_model=previous_model,
                    to_provider=provider,
                    to_model=model,
                    error_kind=previous_failure_kind,
                )
                if note:
                    fallback_notes.append(note)
                    emit_generation_progress(progress_callback, note)

            for retry_index in range(len(TRANSIENT_BACKOFF_SECONDS) + 1):
                if retry_index:
                    delay = TRANSIENT_BACKOFF_SECONDS[retry_index - 1]
                    retry_note = f"{provider_label(provider)} is busy. Retrying in {int(delay)}s..."
                    emit_generation_progress(progress_callback, retry_note)
                    time.sleep(delay)
                attempts += 1
                try:
                    text = self._adapter_for(provider).complete(
                        prompt,
                        model=model,
                        system_instruction=system_instruction,
                    )
                except Exception as exc:
                    error_kind = classify_generation_error(exc)
                    error_message = humanize_provider_error(provider, exc)
                    last_error_kind = error_kind
                    last_error = error_message
                    previous_failure_kind = error_kind
                    previous_provider = provider
                    previous_model = model
                    if error_kind == TRANSIENT_SERVER_ERROR and retry_index < len(TRANSIENT_BACKOFF_SECONDS):
                        continue
                    break

                if text:
                    return GenerationResult(
                        text=text,
                        provider=provider,
                        model=model,
                        used_deep_mode=candidate_deep,
                        attempts=attempts,
                        fallback_notes=tuple(unique_preserving_order(fallback_notes)),
                    )

                last_error_kind = UNKNOWN_ERROR
                last_error = f"{provider_label(provider)} returned an empty response."
                previous_failure_kind = UNKNOWN_ERROR
                previous_provider = provider
                previous_model = model
                break

        return GenerationResult(
            text=None,
            provider=previous_provider or self.provider,
            model=previous_model,
            used_deep_mode=deep,
            attempts=attempts,
            fallback_notes=tuple(unique_preserving_order(fallback_notes)),
            error_kind=last_error_kind or UNKNOWN_ERROR,
            final_error=last_error or f"{self.provider_label} could not generate a response right now.",
        )

    def embed(self, texts: Iterable[str]) -> list[list[float]] | None:
        if not self.available or not self.provider or not self.embedding_model:
            return None
        text_list = list(texts)
        if not text_list:
            return []
        if not self.supports_embeddings(self.provider):
            return None
        model = self._resolved_embedding_model()
        if not model:
            return None
        try:
            return self._adapter_for(self.provider).embed(text_list, model=model)
        except Exception:
            return None

    def supports_embeddings(self, provider: str | None = None) -> bool:
        target = provider or self.provider
        return target == "gemini"

    def provider_status(self, *, refresh: bool = False) -> AIStatusSnapshot:
        warnings = list(self.warnings)
        provider_rows: list[AIProviderStatus] = []

        for provider in self.available_providers:
            listing = self.list_models_safe(provider=provider, refresh=refresh)
            models = list(listing.models)
            if listing.error:
                warnings.append(f"{listing.label}: {listing.error}")
            provider_rows.append(
                AIProviderStatus(
                    provider=provider,
                    api_source=self._credentials_for(provider).api_source if self._credentials_for(provider) else None,
                    selected=provider == self.provider,
                    generation_models=sum(1 for model in models if model.supports(GENERATION_CAPABILITY)),
                    embedding_models=sum(1 for model in models if model.supports(EMBED_CAPABILITY)),
                    error=listing.error,
                )
            )

        fast_model = self.fast_model
        deep_model = self.deep_model
        embedding_model = self.embedding_model

        if self.provider:
            generation_listing = self.list_models_safe(
                provider=self.provider,
                refresh=refresh,
                capability=GENERATION_CAPABILITY,
            )
            if generation_listing.error:
                warnings.append(
                    f"{self.provider_label}: could not verify visible generation models right now. "
                    f"Keeping the saved/default chat and deep models. {generation_listing.error}"
                )
            else:
                generation_ids = {model.id for model in generation_listing.models}
                if fast_model and generation_ids and fast_model not in generation_ids:
                    warnings.append(f"Saved chat model `{fast_model}` is not currently available for {self.provider_label}; DevAgent will fall back automatically.")
                    fast_model = self._fallback_model_for(self.provider, GENERATION_CAPABILITY, deep=False)
                if deep_model and generation_ids and deep_model not in generation_ids:
                    warnings.append(f"Saved deep model `{deep_model}` is not currently available for {self.provider_label}; DevAgent will fall back automatically.")
                    deep_model = self._fallback_model_for(self.provider, GENERATION_CAPABILITY, deep=True)

            if self.supports_embeddings(self.provider):
                embedding_listing = self.list_models_safe(
                    provider=self.provider,
                    refresh=refresh,
                    capability=EMBED_CAPABILITY,
                )
                if embedding_listing.error:
                    warnings.append(
                        f"{self.provider_label}: could not verify visible embedding models right now. "
                        f"Keeping the saved/default embedding model. {embedding_listing.error}"
                    )
                else:
                    embedding_ids = {model.id for model in embedding_listing.models}
                    if embedding_model and embedding_ids and embedding_model not in embedding_ids:
                        warnings.append(f"Saved embedding model `{embedding_model}` is not currently available for {self.provider_label}; DevAgent will fall back automatically.")
                        embedding_model = self._fallback_model_for(self.provider, EMBED_CAPABILITY, deep=False)
            else:
                embedding_model = None

        return AIStatusSnapshot(
            selected_provider=self.provider,
            fast_model=fast_model,
            deep_model=deep_model,
            embedding_model=embedding_model,
            providers=tuple(provider_rows),
            warnings=tuple(unique_preserving_order(warnings)),
        )

    def list_models(
        self,
        *,
        provider: str | None = None,
        refresh: bool = False,
        capability: str | None = None,
    ) -> list[DiscoveredModel]:
        target = provider or self.provider
        if not target:
            return []
        adapter = self._adapter_for(target)
        models = adapter.list_models(refresh=refresh)
        if capability:
            return [model for model in models if model.supports(capability)]
        return models

    def list_models_safe(
        self,
        *,
        provider: str | None = None,
        refresh: bool = False,
        capability: str | None = None,
    ) -> ProviderModelListing:
        target = provider or self.provider
        if not target:
            return ProviderModelListing(provider=provider or "unknown", models=(), error="No AI provider is selected.")
        try:
            models = self.list_models(provider=target, refresh=refresh, capability=capability)
            return ProviderModelListing(provider=target, models=tuple(models))
        except Exception as exc:
            return ProviderModelListing(provider=target, models=(), error=humanize_provider_error(target, exc))

    def _resolved_generation_model(self, *, deep: bool) -> str | None:
        if not self.provider:
            return None
        selected = self.selected_model(deep=deep)
        try:
            available = {model.id for model in self.list_models(provider=self.provider, capability=GENERATION_CAPABILITY)}
        except Exception:
            available = set()
        if selected and (not available or selected in available):
            return selected
        return self._fallback_model_for(self.provider, GENERATION_CAPABILITY, deep=deep)

    def _generation_candidates(self, *, deep: bool) -> list[tuple[str, str, bool]]:
        if not self.provider:
            return []
        candidates: list[tuple[str, str, bool]] = []
        selected_model = self._resolved_generation_model(deep=deep)
        if selected_model:
            candidates.append((self.provider, selected_model, deep))
        fast_model = self._resolved_generation_model(deep=False)
        if deep and fast_model and fast_model != selected_model:
            candidates.append((self.provider, fast_model, False))
        for provider in self.available_providers:
            if provider == self.provider:
                continue
            other_model = resolved_generation_model_for_provider(provider, deep=False)
            if other_model:
                candidates.append((provider, other_model, False))

        unique: list[tuple[str, str, bool]] = []
        seen: set[tuple[str, str]] = set()
        for provider, model, candidate_deep in candidates:
            key = (provider, model)
            if key in seen:
                continue
            seen.add(key)
            unique.append((provider, model, candidate_deep))
        return unique

    def _resolved_embedding_model(self) -> str | None:
        if not self.provider or not self.supports_embeddings(self.provider):
            return None
        selected = self.embedding_model
        try:
            available = {model.id for model in self.list_models(provider=self.provider, capability=EMBED_CAPABILITY)}
        except Exception:
            available = set()
        if selected and (not available or selected in available):
            return selected
        return self._fallback_model_for(self.provider, EMBED_CAPABILITY, deep=False)

    def _fallback_model_for(self, provider: str, capability: str, *, deep: bool) -> str | None:
        try:
            discovered = self.list_models(provider=provider, capability=capability)
        except Exception:
            discovered = []
        if discovered:
            chosen = choose_default_discovered_model(provider, discovered, capability=capability, deep=deep)
            if chosen:
                return chosen.id
        if provider == "gemini":
            if capability == EMBED_CAPABILITY:
                return GEMINI_DEFAULT_EMBEDDING_MODEL
            return GEMINI_DEFAULT_DEEP_MODEL if deep else GEMINI_DEFAULT_FAST_MODEL
        if provider == "groq" and capability == GENERATION_CAPABILITY:
            return GROQ_DEFAULT_DEEP_MODEL if deep else GROQ_DEFAULT_FAST_MODEL
        if provider == "xai" and capability == GENERATION_CAPABILITY:
            return XAI_DEFAULT_FAST_MODEL
        if provider == "openrouter" and capability == GENERATION_CAPABILITY:
            return OPENROUTER_DEFAULT_DEEP_MODEL if deep else OPENROUTER_DEFAULT_FAST_MODEL
        return None

    def _credentials_for(self, provider: str) -> ProviderCredentials | None:
        if provider == self.provider and self.api_key:
            return ProviderCredentials(provider=provider, api_key=self.api_key, api_source=self.api_source or "")
        return resolve_available_credentials().get(provider)

    def _adapter_for(self, provider: str) -> ProviderAdapter:
        credentials = self._credentials_for(provider)
        if credentials is None:
            raise RuntimeError(f"No API key is configured for provider `{provider}`.")
        if provider == "gemini":
            return GeminiAdapter(credentials)
        if provider == "groq":
            return GroqAdapter(credentials)
        if provider == "xai":
            return XAIAdapter(credentials)
        if provider == "openrouter":
            return OpenRouterAdapter(credentials)
        raise RuntimeError(f"Unsupported AI provider `{provider}`.")


def resolve_profile(
    credentials: dict[str, ProviderCredentials],
    ai_settings,
) -> ResolvedAIProfile:
    available = tuple(provider for provider in PROVIDER_ORDER if provider in credentials)
    warnings: list[str] = []
    selected_provider = ai_settings.selected_provider if ai_settings.selected_provider in credentials else None
    if ai_settings.selected_provider and selected_provider is None:
        warnings.append(f"Saved provider `{ai_settings.selected_provider}` is not currently configured; DevAgent is using the next available provider.")
    if selected_provider is None:
        selected_provider = "gemini" if "gemini" in credentials else next(iter(available), None)
    if not selected_provider:
        return ResolvedAIProfile(
            provider=None,
            api_key=None,
            api_source=None,
            fast_model=None,
            deep_model=None,
            embedding_model=None,
            available_providers=available,
            warnings=tuple(warnings),
        )

    provider_config = ai_settings.providers.get(selected_provider, ProviderModelConfig())
    provider_credentials = credentials[selected_provider]
    fast_model = first_non_empty(
        provider_config.model,
        provider_env_model(selected_provider, "model"),
        default_model_for_provider(selected_provider, capability=GENERATION_CAPABILITY, deep=False),
    )
    deep_model = first_non_empty(
        provider_config.deep_model,
        provider_env_model(selected_provider, "deep_model"),
        default_model_for_provider(selected_provider, capability=GENERATION_CAPABILITY, deep=True),
    )
    embedding_model = first_non_empty(
        provider_config.embedding_model,
        provider_env_model(selected_provider, "embedding_model"),
        default_model_for_provider(selected_provider, capability=EMBED_CAPABILITY, deep=False),
    )
    if selected_provider != "gemini":
        embedding_model = None

    return ResolvedAIProfile(
        provider=selected_provider,
        api_key=provider_credentials.api_key,
        api_source=provider_credentials.api_source,
        fast_model=fast_model,
        deep_model=deep_model,
        embedding_model=embedding_model,
        available_providers=available,
        warnings=tuple(warnings),
    )


def resolve_available_credentials() -> dict[str, ProviderCredentials]:
    credentials: dict[str, ProviderCredentials] = {}
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        credentials["gemini"] = ProviderCredentials("gemini", gemini_key, "GEMINI_API_KEY")
    else:
        google_key = os.environ.get("GOOGLE_API_KEY")
        if google_key:
            credentials["gemini"] = ProviderCredentials("gemini", google_key, "GOOGLE_API_KEY")

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        credentials["groq"] = ProviderCredentials("groq", groq_key, "GROQ_API_KEY")

    xai_key = os.environ.get("XAI_API_KEY")
    if xai_key:
        credentials["xai"] = ProviderCredentials("xai", xai_key, "XAI_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        credentials["openrouter"] = ProviderCredentials("openrouter", openrouter_key, "OPENROUTER_API_KEY")
    return credentials


def resolve_api_credentials(provider: str | None = None) -> tuple[str | None, str | None]:
    credentials = resolve_available_credentials()
    if provider:
        chosen = credentials.get(provider)
        return (chosen.api_key, chosen.api_source) if chosen else (None, None)
    for candidate in PROVIDER_ORDER:
        if candidate in credentials:
            chosen = credentials[candidate]
            return chosen.api_key, chosen.api_source
    return None, None


def provider_env_model(provider: str, kind: str) -> str | None:
    if provider == "gemini":
        if kind == "model":
            return os.environ.get("GEMINI_MODEL_FAST") or os.environ.get("GEMINI_MODEL")
        if kind == "deep_model":
            return os.environ.get("GEMINI_MODEL_DEEP")
        if kind == "embedding_model":
            return os.environ.get("GEMINI_EMBEDDING_MODEL")
    if provider == "xai":
        if kind == "model":
            return os.environ.get("XAI_MODEL_FAST") or os.environ.get("XAI_MODEL")
        if kind == "deep_model":
            return os.environ.get("XAI_MODEL_DEEP")
        if kind == "embedding_model":
            return os.environ.get("XAI_EMBEDDING_MODEL")
    if provider == "openrouter":
        if kind == "model":
            return os.environ.get("OPENROUTER_MODEL_FAST") or os.environ.get("OPENROUTER_MODEL")
        if kind == "deep_model":
            return os.environ.get("OPENROUTER_MODEL_DEEP")
    return None


def resolved_generation_model_for_provider(provider: str, *, deep: bool) -> str | None:
    config = ConfigManager.load().ai_settings.providers.get(provider, ProviderModelConfig())
    return first_non_empty(
        config.deep_model if deep else config.model,
        provider_env_model(provider, "deep_model" if deep else "model"),
        default_model_for_provider(provider, capability=GENERATION_CAPABILITY, deep=deep),
    )


def default_model_for_provider(provider: str, *, capability: str, deep: bool) -> str | None:
    if provider == "gemini":
        if capability == EMBED_CAPABILITY:
            return GEMINI_DEFAULT_EMBEDDING_MODEL
        return GEMINI_DEFAULT_DEEP_MODEL if deep else GEMINI_DEFAULT_FAST_MODEL
    if provider == "groq" and capability == GENERATION_CAPABILITY:
        return GROQ_DEFAULT_DEEP_MODEL if deep else GROQ_DEFAULT_FAST_MODEL
    if provider == "xai" and capability == GENERATION_CAPABILITY:
        return XAI_DEFAULT_FAST_MODEL
    if provider == "openrouter" and capability == GENERATION_CAPABILITY:
        return OPENROUTER_DEFAULT_DEEP_MODEL if deep else OPENROUTER_DEFAULT_FAST_MODEL
    return None


def choose_default_discovered_model(
    provider: str,
    models: list[DiscoveredModel],
    *,
    capability: str,
    deep: bool,
) -> DiscoveredModel | None:
    if not models:
        return None

    ranked = list(models)
    if provider == "gemini":
        preferences = []
        if capability == EMBED_CAPABILITY:
            preferences = ("embedding",)
        elif deep:
            preferences = ("2.5-pro", "pro")
        else:
            preferences = ("2.5-flash", "flash")
        for needle in preferences:
            for model in ranked:
                if needle in model.id.casefold():
                    return model
    if provider == "groq" and capability == GENERATION_CAPABILITY:
        preferences = ("llama-3.3-70b-versatile", "70b-versatile", "70b") if deep else ("llama-3.1-8b-instant", "8b-instant", "instant")
        for needle in preferences:
            for model in ranked:
                if needle in model.id.casefold():
                    return model
    return ranked[0]


def first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def provider_label(provider: str | None) -> str:
    if not provider:
        return "AI"
    return PROVIDER_LABELS.get(provider, provider)


def emit_generation_progress(
    callback: GenerationProgressCallback | None,
    message: str,
) -> None:
    if not callback:
        return
    callback(message)


def classify_generation_error(exc: Exception) -> str:
    message = clean_error_text(str(exc)).casefold()
    if not message:
        return UNKNOWN_ERROR

    transient_markers = (
        "503",
        "unavailable",
        "high demand",
        "spikes in demand",
        "temporarily unavailable",
        "temporary",
        "timed out",
        "timeout",
        "deadline exceeded",
        "connection reset",
        "network error",
        "service unavailable",
    )
    if any(marker in message for marker in transient_markers):
        return TRANSIENT_SERVER_ERROR

    quota_markers = (
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "rate-limit",
        "billing",
        "credits",
        "licenses yet",
        "token limit",
        "exhausted",
        "too many requests",
    )
    if any(marker in message for marker in quota_markers):
        return QUOTA_EXHAUSTED_ERROR

    model_markers = (
        "model not found",
        "unknown model",
        "unsupported model",
        "not available for your plan",
        "does not have access to model",
        "model is not available",
        "not found",
    )
    if any(marker in message for marker in model_markers):
        return MODEL_UNAVAILABLE_ERROR

    auth_markers = (
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "permission",
        "api key",
        "authentication",
        "access denied",
    )
    if any(marker in message for marker in auth_markers):
        return AUTH_OR_PERMISSION_ERROR

    provider_markers = (
        "dns",
        "connection refused",
        "name or service not known",
        "host not found",
        "no api key",
        "not currently configured",
        "provider request failed",
    )
    if any(marker in message for marker in provider_markers):
        return PROVIDER_UNAVAILABLE_ERROR

    return UNKNOWN_ERROR


def describe_generation_fallback(
    *,
    from_provider: str | None,
    from_model: str | None,
    to_provider: str,
    to_model: str,
    error_kind: str | None,
) -> str | None:
    if not from_provider or not from_model:
        return None

    if from_provider == to_provider and from_model == to_model:
        return None

    if error_kind == TRANSIENT_SERVER_ERROR:
        reason = f"{provider_label(from_provider)} stayed busy."
    elif error_kind == QUOTA_EXHAUSTED_ERROR:
        reason = f"{provider_label(from_provider)} is exhausted."
    elif error_kind == MODEL_UNAVAILABLE_ERROR:
        reason = f"{provider_label(from_provider)} could not use `{from_model}`."
    elif error_kind == AUTH_OR_PERMISSION_ERROR:
        reason = f"{provider_label(from_provider)} could not use the current credentials."
    elif error_kind == PROVIDER_UNAVAILABLE_ERROR:
        reason = f"{provider_label(from_provider)} is unavailable."
    else:
        reason = f"{provider_label(from_provider)} could not finish the request."

    if from_provider == to_provider:
        return f"{reason} Falling back to `{to_model}`..."
    return f"{reason} Trying {provider_label(to_provider)} with `{to_model}`..."


def is_transient_ai_error(exc: Exception) -> bool:
    return classify_generation_error(exc) == TRANSIENT_SERVER_ERROR


class _DuplicateGeminiKeyFilter(logging.Filter):
    """Hide google-genai's "both keys are set" notice: the key is always passed explicitly."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "Both GOOGLE_API_KEY and GEMINI_API_KEY are set" not in record.getMessage()


logging.getLogger("google_genai._api_client").addFilter(_DuplicateGeminiKeyFilter())


@contextmanager
def selected_api_environment(api_key: str | None, api_source: str | None) -> Iterator[None]:
    """Kept for call-site compatibility; it no longer touches ``os.environ``.

    It used to rewrite and restore the provider key variables around each Gemini call so
    the SDK would not see two keys. The process environment is shared by every thread, so
    that raced with the file watcher and with other calls. Every client is now built with
    its key passed explicitly, which the SDK prefers over the environment.
    """
    yield


def dotenv_candidates() -> list[Path]:
    """Files DevAgent may read provider keys from.

    ``load_dotenv()`` with no argument searches upward from the installed package, which
    is neither predictable nor something a project directory should be able to influence.
    Only these two locations are read, and a variable already set in the environment
    always wins.
    """
    candidates: list[Path] = []
    explicit = os.environ.get("DEVAGENT_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(ConfigManager.config_dir() / ".env")
    return candidates


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    for candidate in dotenv_candidates():
        try:
            if candidate.is_file():
                load_dotenv(candidate, override=False)
        except Exception:
            continue


class GeminiAdapter:
    provider = "gemini"

    def __init__(self, credentials: ProviderCredentials):
        self.credentials = credentials

    def list_models(self, *, refresh: bool = False) -> list[DiscoveredModel]:
        cache_key = (self.provider, self.credentials.api_key)
        if refresh:
            _MODEL_CACHE.pop(cache_key, None)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)

        client = self._get_client()
        with selected_api_environment(self.credentials.api_key, self.credentials.api_source):
            raw_models = list(client.models.list())
        models = [normalize_gemini_model(item) for item in raw_models]
        filtered = [model for model in models if model.capabilities]
        _MODEL_CACHE[cache_key] = tuple(filtered)
        return filtered

    def complete(self, prompt: str, *, model: str, system_instruction: str | None = None) -> str | None:
        client = self._get_client()
        response = None
        if system_instruction:
            try:
                with selected_api_environment(self.credentials.api_key, self.credentials.api_source):
                    from google.genai import types

                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(system_instruction=system_instruction),
                    )
            except Exception as exc:
                if is_transient_ai_error(exc):
                    raise
                response = None
        if response is not None:
            return getattr(response, "text", None)

        combined = prompt if not system_instruction else f"{system_instruction}\n\n{prompt}"
        with selected_api_environment(self.credentials.api_key, self.credentials.api_source):
            response = client.models.generate_content(model=model, contents=combined)
        return getattr(response, "text", None)

    def embed(self, texts: Iterable[str], *, model: str) -> list[list[float]] | None:
        text_list = list(texts)
        if not text_list:
            return []
        client = self._get_client()
        with selected_api_environment(self.credentials.api_key, self.credentials.api_source):
            response = client.models.embed_content(model=model, contents=text_list)
        return [embedding.values for embedding in response.embeddings]

    def _get_client(self):
        cache_key = (self.provider, self.credentials.api_key, self.credentials.api_source)
        cached = _CLIENT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        with selected_api_environment(self.credentials.api_key, self.credentials.api_source):
            from google import genai

            client = genai.Client(api_key=self.credentials.api_key)
        _CLIENT_CACHE[cache_key] = client
        return client


class XAIAdapter:
    provider = "xai"
    base_url = "https://api.x.ai/v1"

    def __init__(self, credentials: ProviderCredentials):
        self.credentials = credentials

    def list_models(self, *, refresh: bool = False) -> list[DiscoveredModel]:
        cache_key = (self.provider, self.credentials.api_key)
        if refresh:
            _MODEL_CACHE.pop(cache_key, None)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)

        try:
            payload = fetch_json(
                f"{self.base_url}/language-models",
                headers={"Authorization": f"Bearer {self.credentials.api_key}"},
            )
        except Exception:
            payload = fetch_json(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.credentials.api_key}"},
            )

        models = [normalize_xai_model(item) for item in extract_model_items(payload)]
        filtered = [model for model in models if model.capabilities]
        _MODEL_CACHE[cache_key] = tuple(filtered)
        return filtered

    def complete(self, prompt: str, *, model: str, system_instruction: str | None = None) -> str | None:
        client = self._get_client()
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model=model, messages=messages)
        return extract_chat_completion_text(response)

    def embed(self, texts: Iterable[str], *, model: str) -> list[list[float]] | None:
        return None

    def _get_client(self):
        cache_key = (self.provider, self.credentials.api_key, self.base_url)
        cached = _CLIENT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        from openai import OpenAI

        client = OpenAI(api_key=self.credentials.api_key, base_url=self.base_url)
        _CLIENT_CACHE[cache_key] = client
        return client


class OpenRouterAdapter(XAIAdapter):
    """OpenAI-compatible adapter for OpenRouter's unified model gateway."""

    provider = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def list_models(self, *, refresh: bool = False) -> list[DiscoveredModel]:
        cache_key = (self.provider, self.credentials.api_key)
        if refresh:
            _MODEL_CACHE.pop(cache_key, None)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)

        payload = fetch_json(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.credentials.api_key}"},
        )
        models = [normalize_openrouter_model(item) for item in extract_model_items(payload)]
        filtered = [model for model in models if model.capabilities]
        _MODEL_CACHE[cache_key] = tuple(filtered)
        return filtered

    def _get_client(self):
        cache_key = (self.provider, self.credentials.api_key, self.base_url)
        cached = _CLIENT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        from openai import OpenAI

        default_headers = {}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_TITLE")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-OpenRouter-Title"] = title
        client = OpenAI(
            api_key=self.credentials.api_key,
            base_url=self.base_url,
            default_headers=default_headers or None,
        )
        _CLIENT_CACHE[cache_key] = client
        return client


class GroqAdapter(XAIAdapter):
    provider = "groq"
    base_url = "https://api.groq.com/openai/v1"

    def list_models(self, *, refresh: bool = False) -> list[DiscoveredModel]:
        cache_key = (self.provider, self.credentials.api_key)
        if refresh:
            _MODEL_CACHE.pop(cache_key, None)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)

        try:
            raw_models = list(self._get_client().models.list())
            models = [normalize_groq_model(item) for item in raw_models]
        except Exception:
            payload = fetch_json(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.credentials.api_key}"},
            )
            models = [normalize_groq_model(item) for item in extract_model_items(payload)]
        filtered = [model for model in models if model.capabilities]
        _MODEL_CACHE[cache_key] = tuple(filtered)
        return filtered


def normalize_gemini_model(raw) -> DiscoveredModel:
    name = extract_attr(raw, "name") or extract_attr(raw, "model") or ""
    model_id = strip_model_prefix(name) or str(name)
    label = extract_attr(raw, "display_name") or extract_attr(raw, "displayName") or model_id
    methods = [
        value.casefold()
        for value in normalize_string_values(
            extract_attr(raw, "supported_actions")
            or extract_attr(raw, "supportedActions")
            or extract_attr(raw, "supported_generation_methods")
            or extract_attr(raw, "supportedGenerationMethods")
            or extract_attr(raw, "supported_methods")
            or extract_attr(raw, "supportedMethods")
            or []
        )
    ]
    capabilities: list[str] = []
    if any("generatecontent" in method or "streamgeneratecontent" in method for method in methods):
        capabilities.append(GENERATION_CAPABILITY)
    if any("embed" in method for method in methods) or "embedding" in model_id.casefold():
        capabilities.append(EMBED_CAPABILITY)
    aliases = tuple(unique_preserving_order([model_id, name] if name else [model_id]))
    modalities = tuple(normalize_string_values(extract_attr(raw, "supported_modalities") or extract_attr(raw, "inputModalities") or []))
    return DiscoveredModel(
        provider="gemini",
        id=model_id,
        label=str(label),
        capabilities=tuple(capabilities),
        modalities=modalities,
        aliases=aliases,
    )


def normalize_xai_model(raw) -> DiscoveredModel:
    name = extract_attr(raw, "id") or extract_attr(raw, "name") or ""
    model_id = str(name)
    label = extract_attr(raw, "display_name") or extract_attr(raw, "displayName") or extract_attr(raw, "name") or model_id
    aliases = normalize_string_values(extract_attr(raw, "aliases") or [])
    modalities = normalize_string_values(
        extract_attr(raw, "modalities")
        or extract_attr(raw, "input_modalities")
        or extract_attr(raw, "inputModalities")
        or []
    )
    return DiscoveredModel(
        provider="xai",
        id=model_id,
        label=str(label),
        capabilities=(GENERATION_CAPABILITY,),
        modalities=tuple(modalities),
        aliases=tuple(unique_preserving_order([model_id, *aliases])),
    )


def normalize_openrouter_model(raw) -> DiscoveredModel:
    model_id = str(extract_attr(raw, "id") or extract_attr(raw, "name") or "")
    label = extract_attr(raw, "name") or model_id
    architecture = extract_attr(raw, "architecture") or {}
    modalities = normalize_string_values(
        extract_attr(raw, "input_modalities")
        or extract_attr(raw, "inputModalities")
        or (architecture.get("input_modalities") if isinstance(architecture, dict) else [])
    )
    capabilities = () if "embed" in model_id.casefold() else (GENERATION_CAPABILITY,)
    return DiscoveredModel(
        provider="openrouter",
        id=model_id,
        label=str(label),
        capabilities=capabilities,
        modalities=tuple(modalities),
        aliases=(model_id,),
    )


def normalize_groq_model(raw) -> DiscoveredModel:
    name = extract_attr(raw, "id") or extract_attr(raw, "name") or ""
    model_id = str(name)
    label = (
        extract_attr(raw, "display_name")
        or extract_attr(raw, "displayName")
        or extract_attr(raw, "name")
        or model_id
    )
    aliases = normalize_string_values(extract_attr(raw, "aliases") or [])
    modalities = normalize_string_values(
        extract_attr(raw, "modalities")
        or extract_attr(raw, "input_modalities")
        or extract_attr(raw, "inputModalities")
        or []
    )
    active = extract_attr(raw, "active")
    capabilities: tuple[str, ...] = ()
    if active is not False and is_groq_generation_model(model_id, modalities=modalities):
        capabilities = (GENERATION_CAPABILITY,)
    return DiscoveredModel(
        provider="groq",
        id=model_id,
        label=str(label),
        capabilities=capabilities,
        modalities=tuple(modalities),
        aliases=tuple(unique_preserving_order([model_id, *aliases])),
    )


def is_groq_generation_model(model_id: str, *, modalities: Iterable[str] = ()) -> bool:
    identifier = model_id.casefold()
    modality_values = tuple(value.casefold() for value in modalities)
    blocked_markers = (
        "whisper",
        "prompt-guard",
        "safeguard",
        "speech",
        "transcribe",
        "transcription",
        "audio",
        "tts",
    )
    if any(marker in identifier for marker in blocked_markers):
        return False
    if any(any(marker in modality for marker in blocked_markers) for modality in modality_values):
        return False
    return True


def extract_model_items(payload) -> list[object]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "models", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def fetch_json(url: str, *, headers: dict[str, str]) -> object:
    req = request.Request(url, headers={**headers, "Accept": "application/json"})
    try:
        with request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise RuntimeError(extract_http_error_message(exc)) from exc
    except error.URLError as exc:
        reason = clean_error_text(str(getattr(exc, "reason", exc))) or "Network request failed."
        raise RuntimeError(f"Network error: {reason}") from exc


def extract_chat_completion_text(response) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return None
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    if message is None:
        return None
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part).strip() or None
    return None


def extract_attr(value, name: str):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def normalize_string_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return [str(value)]


def strip_model_prefix(name: str) -> str:
    return name[7:] if name.startswith("models/") else name


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.append(value)
    return seen


def humanize_provider_error(provider: str, exc: Exception) -> str:
    label = PROVIDER_LABELS.get(provider, provider)
    message = clean_error_text(str(exc)) or "Provider request failed."
    if message.lower().startswith(label.lower()):
        return message
    return message


def extract_http_error_message(exc: error.HTTPError) -> str:
    status = getattr(exc, "code", None)
    reason = clean_error_text(str(getattr(exc, "reason", "") or "")) or "HTTP error"
    body_text = ""
    try:
        body_text = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body_text = ""
    detail = extract_error_detail_from_body(body_text)
    if detail:
        if reason and reason.casefold() not in detail.casefold():
            return f"HTTP {status} {reason}: {detail}"
        return f"HTTP {status}: {detail}"
    return f"HTTP {status} {reason}"


def extract_error_detail_from_body(body_text: str) -> str | None:
    compact_body = clean_error_text(body_text)
    if not compact_body:
        return None
    try:
        payload = json.loads(body_text)
    except Exception:
        return compact_body

    if isinstance(payload, dict):
        error_value = payload.get("error")
        top_message = clean_error_text(str(payload.get("message", ""))) if payload.get("message") else ""
        code_value = payload.get("code")
        code_message = ""
        if isinstance(code_value, str) and not code_value.isdigit():
            code_message = clean_error_text(code_value)

        if isinstance(error_value, dict):
            nested_message = clean_error_text(str(error_value.get("message", ""))) if error_value.get("message") else ""
            nested_error = clean_error_text(str(error_value.get("error", ""))) if error_value.get("error") else ""
            nested_code = error_value.get("code")
            nested_code_text = ""
            if isinstance(nested_code, str) and not nested_code.isdigit():
                nested_code_text = clean_error_text(nested_code)
            parts = [part for part in (nested_message, nested_error, nested_code_text, top_message, code_message) if part]
            return " ".join(unique_preserving_order(parts)) or compact_body

        if isinstance(error_value, str):
            error_message = clean_error_text(error_value)
            parts = [part for part in (code_message, error_message, top_message) if part]
            return " ".join(unique_preserving_order(parts)) or compact_body

        if top_message or code_message:
            return " ".join(unique_preserving_order([part for part in (top_message, code_message) if part]))

    return compact_body


def clean_error_text(value: str) -> str:
    return " ".join(value.split()).strip()
