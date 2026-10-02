import os
import sys
import types
from pathlib import Path

import devagent.tools.ai as ai_module
from devagent.config.settings import AISettings, ConfigManager, ProviderModelConfig
from devagent.tools.ai import AIClient, resolve_api_credentials, selected_api_environment


def _clear_ai_caches() -> None:
    ai_module._CLIENT_CACHE.clear()
    ai_module._MODEL_CACHE.clear()


def test_resolve_api_credentials_prefers_gemini_over_google_groq_and_xai(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("XAI_API_KEY", "xai-key")

    api_key, source = resolve_api_credentials()

    assert api_key == "gemini-key"
    assert source == "GEMINI_API_KEY"
    assert resolve_api_credentials("groq") == ("groq-key", "GROQ_API_KEY")
    assert resolve_api_credentials("xai") == ("xai-key", "XAI_API_KEY")


def test_resolve_api_credentials_includes_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    assert resolve_api_credentials("openrouter") == ("openrouter-key", "OPENROUTER_API_KEY")


def test_ai_client_uses_openrouter_chat_completion_path(monkeypatch) -> None:
    _clear_ai_caches()
    captured: list[dict[str, object]] = []

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="openrouter says hi"))]
            )

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, default_headers=None):
            assert api_key == "openrouter-key"
            assert base_url == "https://openrouter.ai/api/v1"
            assert default_headers == {"HTTP-Referer": "https://example.test", "X-OpenRouter-Title": "DevAgent"}
            self.chat = types.SimpleNamespace(completions=FakeChatCompletions())

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.test")
    monkeypatch.setenv("OPENROUTER_TITLE", "DevAgent")

    client = AIClient(
        provider="openrouter",
        api_key="openrouter-key",
        api_source="OPENROUTER_API_KEY",
        fast_model="openai/gpt-4o-mini",
        available_providers=("openrouter",),
    )

    assert client.complete("hello", system_instruction="Be helpful") == "openrouter says hi"
    assert captured == [{
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hello"},
        ],
    }]


def test_ai_client_lists_openrouter_models(monkeypatch) -> None:
    _clear_ai_caches()
    monkeypatch.setattr(
        ai_module,
        "fetch_json",
        lambda *args, **kwargs: {"data": [
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "architecture": {"input_modalities": ["text"]}},
            {"id": "openai/text-embedding-3-small", "name": "Embedding", "architecture": {"input_modalities": ["text"]}},
        ]},
    )

    client = AIClient(provider="openrouter", api_key="openrouter-key", api_source="OPENROUTER_API_KEY", available_providers=("openrouter",))
    models = client.list_models(provider="openrouter", refresh=True)

    assert [model.id for model in models] == ["openai/gpt-4o-mini"]
    assert models[0].provider == "openrouter"


def test_selected_api_environment_never_touches_the_process_environment(monkeypatch) -> None:
    # It used to hide the other Google key by rewriting os.environ, which raced with every
    # other thread. The key is passed to the SDK explicitly instead.
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    before = dict(os.environ)

    with selected_api_environment("gemini-key", "GEMINI_API_KEY"):
        assert dict(os.environ) == before

    assert dict(os.environ) == before


def test_selected_api_environment_is_safe_across_threads(monkeypatch) -> None:
    import threading

    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    seen: list[tuple[str | None, str | None]] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        for _ in range(200):
            with selected_api_environment("gemini-key", "GEMINI_API_KEY"):
                seen.append((os.environ.get("GEMINI_API_KEY"), os.environ.get("GOOGLE_API_KEY")))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert set(seen) == {("gemini-key", "google-key")}


def test_duplicate_gemini_key_notice_is_filtered_but_other_warnings_are_not() -> None:
    import logging

    duplicate_filter = next(f for f in logging.getLogger("google_genai._api_client").filters if type(f).__name__ == "_DuplicateGeminiKeyFilter")

    def record(message: str) -> logging.LogRecord:
        return logging.LogRecord("google_genai._api_client", logging.WARNING, __file__, 1, message, None, None)

    assert not duplicate_filter.filter(record("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY."))
    assert duplicate_filter.filter(record("Quota nearly exhausted"))


def test_ai_client_reuses_cached_gemini_client_and_passes_the_key_explicitly(monkeypatch) -> None:
    _clear_ai_caches()
    client_envs: list[tuple[str | None, str | None]] = []
    call_envs: list[tuple[str | None, str | None]] = []
    explicit_keys: list[str | None] = []

    class FakeResponse:
        text = "ok"

    class FakeModels:
        def generate_content(self, **kwargs):
            call_envs.append((os.environ.get("GEMINI_API_KEY"), os.environ.get("GOOGLE_API_KEY")))
            return FakeResponse()

        def embed_content(self, **kwargs):
            call_envs.append((os.environ.get("GEMINI_API_KEY"), os.environ.get("GOOGLE_API_KEY")))
            return types.SimpleNamespace(embeddings=[types.SimpleNamespace(values=[0.1, 0.2])])

    class FakeClient:
        def __init__(self, api_key=None):
            explicit_keys.append(api_key)
            client_envs.append((os.environ.get("GEMINI_API_KEY"), os.environ.get("GOOGLE_API_KEY")))
            self.models = FakeModels()

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=FakeClient)

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    client = AIClient(provider="gemini", api_key="gemini-key", api_source="GEMINI_API_KEY", available_providers=("gemini",))

    assert client.complete("hello") == "ok"
    assert client.complete("hello again") == "ok"
    assert client.embed(["one"]) == [[0.1, 0.2]]
    assert explicit_keys == ["gemini-key"]
    assert client_envs == [("gemini-key", "google-key")]
    assert call_envs == [("gemini-key", "google-key")] * 3


def test_ai_client_retries_transient_503_errors(monkeypatch) -> None:
    _clear_ai_caches()
    attempts: list[int] = []
    sleeps: list[float] = []
    progress: list[str] = []

    class FakeResponse:
        text = "ok after retry"

    class FakeModels:
        def generate_content(self, **kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=FakeClient)

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setattr(ai_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    client = AIClient(provider="gemini", api_key="gemini-key", api_source="GEMINI_API_KEY", available_providers=("gemini",))

    result = client.generate("hello", progress_callback=progress.append)

    assert result.text == "ok after retry"
    assert result.attempts == 3
    assert len(attempts) == 3
    assert sleeps == [3.0, 5.0]
    assert progress[0] == "Thinking..."
    assert "Retrying in 3s" in progress[1]
    assert "Retrying in 5s" in progress[2]


def test_ai_client_returns_error_after_repeated_transient_failures(monkeypatch) -> None:
    _clear_ai_caches()
    attempts: list[int] = []
    sleeps: list[float] = []

    class FakeModels:
        def generate_content(self, **kwargs):
            attempts.append(1)
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=FakeClient)

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setattr(ai_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    client = AIClient(provider="gemini", api_key="gemini-key", api_source="GEMINI_API_KEY", available_providers=("gemini",))

    result = client.generate("hello")

    assert result.succeeded is False
    assert result.error_kind == ai_module.TRANSIENT_SERVER_ERROR
    assert "high demand" in str(result.final_error).lower()
    assert result.attempts == 3
    assert client.complete("hello") is None
    assert len(attempts) == 6
    assert sleeps == [3.0, 5.0, 3.0, 5.0]


def test_ai_client_does_not_retry_non_transient_errors(monkeypatch) -> None:
    _clear_ai_caches()
    attempts: list[int] = []
    sleeps: list[float] = []

    class FakeModels:
        def generate_content(self, **kwargs):
            attempts.append(1)
            raise RuntimeError("400 INVALID_ARGUMENT")

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=FakeClient)

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setattr(ai_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    client = AIClient(provider="gemini", api_key="gemini-key", api_source="GEMINI_API_KEY", available_providers=("gemini",))

    result = client.generate("hello")
    assert result.succeeded is False
    assert result.error_kind == ai_module.UNKNOWN_ERROR
    assert client.complete("hello") is None
    assert len(attempts) == 2
    assert sleeps == []


def test_ai_client_falls_back_from_deep_model_to_same_provider_fast_model(monkeypatch) -> None:
    _clear_ai_caches()

    class FakeAdapter:
        def __init__(self, provider: str):
            self.provider = provider
            self.credentials = types.SimpleNamespace(provider=provider, api_key=f"{provider}-key")

        def list_models(self, *, refresh: bool = False):
            if self.provider == "gemini":
                return [
                    ai_module.DiscoveredModel("gemini", "gemini-2.5-flash", "Gemini Flash", ("generate",)),
                    ai_module.DiscoveredModel("gemini", "gemini-2.5-pro", "Gemini Pro", ("generate",)),
                ]
            return []

        def complete(self, prompt: str, *, model: str, system_instruction: str | None = None):
            if model == "gemini-2.5-pro":
                raise RuntimeError("Model is not available for your plan.")
            return f"answered with {model}"

        def embed(self, texts, *, model: str):
            return None

    monkeypatch.setattr(AIClient, "_adapter_for", lambda self, provider: FakeAdapter(provider))

    client = AIClient(
        provider="gemini",
        api_key="gemini-key",
        api_source="GEMINI_API_KEY",
        fast_model="gemini-2.5-flash",
        deep_model="gemini-2.5-pro",
        available_providers=("gemini",),
    )

    result = client.generate("hello", deep=True)

    assert result.text == "answered with gemini-2.5-flash"
    assert result.model == "gemini-2.5-flash"
    assert any("Falling back to `gemini-2.5-flash`" in note for note in result.fallback_notes)


def test_ai_client_falls_back_to_next_provider_when_selected_provider_is_exhausted(monkeypatch) -> None:
    _clear_ai_caches()

    class FakeAdapter:
        def __init__(self, provider: str):
            self.provider = provider
            self.credentials = types.SimpleNamespace(provider=provider, api_key=f"{provider}-key")

        def list_models(self, *, refresh: bool = False):
            if self.provider == "gemini":
                return [ai_module.DiscoveredModel("gemini", "gemini-2.5-flash", "Gemini Flash", ("generate",))]
            return [ai_module.DiscoveredModel("xai", "grok-3-mini", "Grok 3 Mini", ("generate",))]

        def complete(self, prompt: str, *, model: str, system_instruction: str | None = None):
            if self.provider == "gemini":
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota exhausted")
            return "answered with xai"

        def embed(self, texts, *, model: str):
            return None

    monkeypatch.setattr(AIClient, "_adapter_for", lambda self, provider: FakeAdapter(provider))

    client = AIClient(
        provider="gemini",
        api_key="gemini-key",
        api_source="GEMINI_API_KEY",
        fast_model="gemini-2.5-flash",
        deep_model="gemini-2.5-pro",
        available_providers=("gemini", "xai"),
    )

    result = client.generate("hello")

    assert result.text == "answered with xai"
    assert result.provider == "xai"
    assert any("Trying xAI with `grok-3-mini`" in note for note in result.fallback_notes)


def test_ai_client_prefers_groq_before_xai_during_cross_provider_failover(monkeypatch) -> None:
    _clear_ai_caches()

    class FakeAdapter:
        def __init__(self, provider: str):
            self.provider = provider
            self.credentials = types.SimpleNamespace(provider=provider, api_key=f"{provider}-key")

        def list_models(self, *, refresh: bool = False):
            if self.provider == "gemini":
                return [ai_module.DiscoveredModel("gemini", "gemini-2.5-flash", "Gemini Flash", ("generate",))]
            if self.provider == "groq":
                return [ai_module.DiscoveredModel("groq", "llama-3.1-8b-instant", "Llama 3.1 8B Instant", ("generate",))]
            return [ai_module.DiscoveredModel("xai", "grok-3-mini", "Grok 3 Mini", ("generate",))]

        def complete(self, prompt: str, *, model: str, system_instruction: str | None = None):
            if self.provider == "gemini":
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota exhausted")
            if self.provider == "groq":
                return "answered with groq"
            return "answered with xai"

        def embed(self, texts, *, model: str):
            return None

    monkeypatch.setattr(AIClient, "_adapter_for", lambda self, provider: FakeAdapter(provider))

    client = AIClient(
        provider="gemini",
        api_key="gemini-key",
        api_source="GEMINI_API_KEY",
        fast_model="gemini-2.5-flash",
        deep_model="gemini-2.5-pro",
        available_providers=("gemini", "groq", "xai"),
    )

    result = client.generate("hello")

    assert result.text == "answered with groq"
    assert result.provider == "groq"
    assert any("Trying Groq with `llama-3.1-8b-instant`" in note for note in result.fallback_notes)


def test_ai_client_uses_saved_xai_provider_selection(tmp_path: Path, monkeypatch) -> None:
    _clear_ai_caches()
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    ConfigManager.save_ai_settings(
        AISettings(
            selected_provider="xai",
            providers={"xai": ProviderModelConfig(model="grok-3-mini", deep_model="grok-3-mini")},
        )
    )

    client = AIClient.from_env()

    assert client.provider == "xai"
    assert client.fast_model == "grok-3-mini"
    assert client.deep_model == "grok-3-mini"
    assert client.embedding_model is None


def test_ai_client_uses_saved_groq_provider_selection(tmp_path: Path, monkeypatch) -> None:
    _clear_ai_caches()
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    ConfigManager.save_ai_settings(
        AISettings(
            selected_provider="groq",
            providers={"groq": ProviderModelConfig(model="llama-3.1-8b-instant", deep_model="llama-3.3-70b-versatile")},
        )
    )

    client = AIClient.from_env()

    assert client.provider == "groq"
    assert client.fast_model == "llama-3.1-8b-instant"
    assert client.deep_model == "llama-3.3-70b-versatile"
    assert client.embedding_model is None


def test_bind_workspace_preserves_saved_ai_settings(tmp_path: Path, monkeypatch) -> None:
    _clear_ai_caches()
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    ConfigManager.save_ai_settings(
        AISettings(
            selected_provider="gemini",
            providers={"gemini": ProviderModelConfig(model="gemini-2.5-flash")},
        )
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    ConfigManager.bind_workspace(workspace)
    config = ConfigManager.load()

    assert config.workspace_path == workspace.resolve()
    assert config.ai_settings.selected_provider == "gemini"
    assert config.ai_settings.providers["gemini"].model == "gemini-2.5-flash"


def test_ai_client_filters_gemini_generation_and_embedding_models(monkeypatch) -> None:
    _clear_ai_caches()

    class FakeModels:
        def list(self):
            return [
                types.SimpleNamespace(
                    name="models/gemini-2.5-flash",
                    display_name="Gemini 2.5 Flash",
                    supported_generation_methods=["generateContent", "countTokens"],
                ),
                types.SimpleNamespace(
                    name="models/gemini-embedding-001",
                    display_name="Gemini Embedding",
                    supported_generation_methods=["embedContent"],
                ),
            ]

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = AIClient(provider="gemini", api_key="gemini-key", api_source="GEMINI_API_KEY", available_providers=("gemini",))
    models = client.list_models(provider="gemini", refresh=True)

    assert [(model.id, model.capabilities) for model in models] == [
        ("gemini-2.5-flash", ("generate",)),
        ("gemini-embedding-001", ("embed",)),
    ]


def test_ai_client_filters_gemini_models_from_supported_actions_shape(monkeypatch) -> None:
    _clear_ai_caches()

    class FakeModels:
        def list(self):
            return [
                {
                    "name": "models/gemini-2.5-flash",
                    "display_name": "Gemini 2.5 Flash",
                    "supported_actions": ["generateContent", "countTokens"],
                },
                {
                    "name": "models/gemini-2.5-pro",
                    "display_name": "Gemini 2.5 Pro",
                    "supported_actions": ["generateContent"],
                },
                {
                    "name": "models/gemini-embedding-001",
                    "display_name": "Gemini Embedding 001",
                    "supported_actions": ["embedContent"],
                },
            ]

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = AIClient(provider="gemini", api_key="gemini-key", api_source="GEMINI_API_KEY", available_providers=("gemini",))
    models = client.list_models(provider="gemini", refresh=True)

    assert [(model.id, model.capabilities) for model in models] == [
        ("gemini-2.5-flash", ("generate",)),
        ("gemini-2.5-pro", ("generate",)),
        ("gemini-embedding-001", ("embed",)),
    ]


def test_ai_client_lists_xai_models_from_language_models_endpoint(monkeypatch) -> None:
    _clear_ai_caches()
    monkeypatch.setattr(
        ai_module,
        "fetch_json",
        lambda url, headers: {
            "data": [
                {"id": "grok-3-mini", "name": "Grok 3 Mini"},
                {"id": "grok-3-fast", "display_name": "Grok 3 Fast"},
            ]
        },
    )

    client = AIClient(provider="xai", api_key="xai-key", api_source="XAI_API_KEY", available_providers=("xai",))
    models = client.list_models(provider="xai", refresh=True)

    assert [model.id for model in models] == ["grok-3-mini", "grok-3-fast"]
    assert all(model.capabilities == ("generate",) for model in models)


def test_ai_client_lists_groq_models_and_filters_non_chat_models(monkeypatch) -> None:
    _clear_ai_caches()

    class FakeModels:
        def list(self):
            return [
                types.SimpleNamespace(id="llama-3.1-8b-instant", active=True),
                types.SimpleNamespace(id="llama-3.3-70b-versatile", active=True),
                types.SimpleNamespace(id="whisper-large-v3", active=True),
                types.SimpleNamespace(id="meta-llama/llama-prompt-guard-2-86m", active=True),
                types.SimpleNamespace(id="openai/gpt-oss-20b", active=True),
                types.SimpleNamespace(id="openai/gpt-oss-safeguard-20b", active=True),
            ]

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.models = FakeModels()
            self.chat = types.SimpleNamespace(completions=None)

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    client = AIClient(provider="groq", api_key="groq-key", api_source="GROQ_API_KEY", available_providers=("groq",))
    models = client.list_models(provider="groq", refresh=True)

    assert [model.id for model in models] == [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
    ]
    assert all(model.capabilities == ("generate",) for model in models)


def test_ai_client_uses_xai_chat_completion_path(monkeypatch) -> None:
    _clear_ai_caches()
    captured: list[dict[str, object]] = []

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(content="xai says hi"))
                ]
            )

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.chat = types.SimpleNamespace(completions=FakeChatCompletions())

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    client = AIClient(provider="xai", api_key="xai-key", api_source="XAI_API_KEY", fast_model="grok-3-mini", available_providers=("xai",))

    assert client.complete("hello", system_instruction="Be helpful") == "xai says hi"
    assert captured == [
        {
            "model": "grok-3-mini",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "hello"},
            ],
        }
    ]


def test_ai_client_uses_groq_chat_completion_path(monkeypatch) -> None:
    _clear_ai_caches()
    captured: list[dict[str, object]] = []

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(content="groq says hi"))
                ]
            )

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            assert api_key == "groq-key"
            assert base_url == "https://api.groq.com/openai/v1"
            self.chat = types.SimpleNamespace(completions=FakeChatCompletions())

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    client = AIClient(
        provider="groq",
        api_key="groq-key",
        api_source="GROQ_API_KEY",
        fast_model="llama-3.1-8b-instant",
        available_providers=("groq",),
    )

    assert client.complete("hello", system_instruction="Be helpful") == "groq says hi"
    assert captured == [
        {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "hello"},
            ],
        }
    ]


def test_ai_client_status_warns_and_falls_back_when_saved_model_is_missing(tmp_path: Path, monkeypatch) -> None:
    _clear_ai_caches()
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    ConfigManager.save_ai_settings(
        AISettings(
            selected_provider="gemini",
            providers={"gemini": ProviderModelConfig(model="ghost-model", deep_model="ghost-deep", embedding_model="ghost-embed")},
        )
    )

    class FakeModels:
        def list(self):
            return [
                types.SimpleNamespace(
                    name="models/gemini-2.5-flash",
                    display_name="Gemini 2.5 Flash",
                    supported_generation_methods=["generateContent"],
                ),
                types.SimpleNamespace(
                    name="models/gemini-2.5-pro",
                    display_name="Gemini 2.5 Pro",
                    supported_generation_methods=["generateContent"],
                ),
                types.SimpleNamespace(
                    name="models/gemini-embedding-001",
                    display_name="Gemini Embedding",
                    supported_generation_methods=["embedContent"],
                ),
            ]

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "google", fake_google)

    status = AIClient.from_env().provider_status(refresh=True)

    assert status.selected_provider == "gemini"
    assert status.fast_model == "gemini-2.5-flash"
    assert status.deep_model == "gemini-2.5-pro"
    assert status.embedding_model == "gemini-embedding-001"
    assert any("ghost-model" in warning for warning in status.warnings)
    assert any("ghost-deep" in warning for warning in status.warnings)
    assert any("ghost-embed" in warning for warning in status.warnings)


def test_ai_client_provider_status_warns_and_continues_when_one_provider_fails(tmp_path: Path, monkeypatch) -> None:
    _clear_ai_caches()
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("XAI_API_KEY", "xai-key")

    class FakeModels:
        def list(self):
            return [
                types.SimpleNamespace(
                    name="models/gemini-2.5-flash",
                    display_name="Gemini 2.5 Flash",
                    supported_generation_methods=["generateContent"],
                ),
                types.SimpleNamespace(
                    name="models/gemini-embedding-001",
                    display_name="Gemini Embedding",
                    supported_generation_methods=["embedContent"],
                ),
            ]

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setattr(
        ai_module,
        "fetch_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("HTTP 403 Forbidden: The caller does not have permission to execute the specified operation. Your newly created team doesn't have any credits or licenses yet.")
        ),
    )

    status = AIClient.from_env().provider_status(refresh=True)

    assert status.selected_provider == "gemini"
    gemini = next(item for item in status.providers if item.provider == "gemini")
    xai = next(item for item in status.providers if item.provider == "xai")
    assert gemini.error is None
    assert gemini.generation_models == 1
    assert gemini.embedding_models == 1
    assert xai.error is not None
    assert "403" in xai.error
    assert any("xAI" in warning for warning in status.warnings)


def test_xai_embed_returns_none_without_embedding_support() -> None:
    client = AIClient(provider="xai", api_key="xai-key", api_source="XAI_API_KEY", embedding_model=None, available_providers=("xai",))

    assert client.embed(["hello"]) is None
