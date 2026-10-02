"""Keep credentials out of anything sent to an AI provider.

Two layers, both applied before indexed text can reach a prompt or an embedding call:

* ``is_sensitive_path`` -- whole files that should never be indexed.
* ``redact_secrets`` -- credential-shaped values inside otherwise ordinary files.

This is pattern matching, not a guarantee. It lowers the chance that a stray key in a
config file leaves the machine; it does not make an unreviewed repository safe to send.
"""

from __future__ import annotations

import re
from pathlib import PurePath

REDACTED = "[REDACTED]"

SENSITIVE_FILE_NAMES = frozenset(
    {
        ".npmrc",
        ".yarnrc",
        ".netrc",
        ".pgpass",
        ".pypirc",
        ".htpasswd",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "terraform.tfstate",
        "terraform.tfstate.backup",
    }
)
SENSITIVE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".tfvars", ".kdbx"})
SENSITIVE_DIR_NAMES = frozenset({".ssh", ".aws", ".gnupg", ".kube", ".docker"})
ENV_TEMPLATE_SUFFIXES = (".example", ".sample", ".template", ".dist")
SECRET_NAME_HINTS = ("credential", "service-account", "service_account", "serviceaccount", "client_secret", "secrets")
STRUCTURED_SUFFIXES = frozenset({".json", ".yml", ".yaml", ".toml", ".ini", ".cfg"})


def is_sensitive_path(path: str | PurePath) -> bool:
    """True for files that commonly hold credentials and must not be indexed."""
    pure = PurePath(str(path).replace("\\", "/"))
    name = pure.name.lower()
    if any(part.lower() in SENSITIVE_DIR_NAMES for part in pure.parts[:-1]):
        return True
    if name == ".env" or name.startswith(".env."):
        return not name.endswith(ENV_TEMPLATE_SUFFIXES)
    if name in SENSITIVE_FILE_NAMES or pure.suffix.lower() in SENSITIVE_SUFFIXES:
        return True
    if pure.suffix.lower() in STRUCTURED_SUFFIXES and any(hint in name for hint in SECRET_NAME_HINTS):
        return True
    return False


_NAME = r"(?:api[_-]?key|secret|token|client[_-]?secret|jwt[_-]?secret|aws[_-]?secret[_-]?access[_-]?key|password|passwd|pwd)"

# Whole match is replaced.
_WHOLE_MATCH_RULES = (
    re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----.*?(?:-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----|\Z)", re.DOTALL),
    re.compile(r"\b(?:A3T[A-Z0-9]|AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9._-]{8,}\.[A-Za-z0-9._-]{8,}\b"),
    re.compile(r"\b(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|amqp)://[^\s'\"`]*:[^\s'\"`@]+@[^\s'\"`]+"),
)

# Group 1 (the name and separator) is kept; group 2 (the value) is replaced.
_VALUE_RULES = (
    re.compile(rf"(?i)(\b[\w.-]*{_NAME}[\w.-]*\b\s*[:=]\s*['\"])([^'\"\n]{{6,}})(?=['\"])"),
    re.compile(rf"(?im)(^\s*(?:export\s+)?[\w.-]*{_NAME}[\w.-]*\s*[:=]\s*)(?!['\"])([^\s#'\"]{{8,}})"),
    re.compile(r"(?i)(\bauthorization\b\s*[:=]\s*['\"]?bearer\s+)([A-Za-z0-9._~+/=-]{12,})"),
)


def redact_secrets(text: str) -> str:
    for rule in _WHOLE_MATCH_RULES:
        text = rule.sub(REDACTED, text)
    for rule in _VALUE_RULES:
        text = rule.sub(lambda match: f"{match.group(1)}{REDACTED}", text)
    return text
