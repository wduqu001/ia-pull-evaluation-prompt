"""Helper utilities for the prompt optimization project."""

import os
import yaml
import json
import time
import random
import threading
from typing import Dict, Any, Optional, Callable, TypeVar
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T")
_invoke_lock = threading.Lock()
_last_invoke_timestamp = 0.0


def load_yaml(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML data or None if loading fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def save_yaml(data: Dict[str, Any], file_path: str) -> bool:
    """
    Save data to a YAML file.

    Args:
        data: Data to persist.
        file_path: Destination path for the YAML file.

    Returns:
        True when save succeeds, False otherwise.
    """
    try:
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)

        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False


def check_env_vars(required_vars: list) -> bool:
    """
    Check that required environment variables exist.

    Args:
        required_vars: List of required variable names.

    Returns:
        True if all variables are present, False otherwise.
    """
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nConfigure them in the .env file before continuing.")
        return False

    return True


def format_score(score: float, threshold: float = 0.9) -> str:
    """Format a score with a pass/fail indicator."""
    symbol = "✓" if score >= threshold else "✗"
    return f"{score:.2f} {symbol}"


def print_section_header(title: str, char: str = "=", width: int = 50):
    """Print a styled section header."""
    print("\n" + char * width)
    print(title)
    print(char * width + "\n")


def validate_prompt_structure(prompt_data: Dict[str, Any]) -> tuple[bool, list]:
    """Validate the basic structure of a prompt payload."""
    errors = []

    required_fields = ['description', 'system_prompt', 'version']
    for field in required_fields:
        if field not in prompt_data:
            errors.append(f"Missing required field: {field}")

    system_prompt = prompt_data.get('system_prompt', '').strip()
    if not system_prompt:
        errors.append("system_prompt is empty")

    if 'TODO' in system_prompt:
        errors.append("system_prompt still contains TODOs")

    techniques = prompt_data.get('techniques_applied', [])
    if len(techniques) < 2:
        errors.append(f"At least 2 techniques are required, found: {len(techniques)}")

    return (len(errors) == 0, errors)


def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from an LLM response that may include surrounding text."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1

        if start != -1 and end > start:
            try:
                json_str = response_text[start:end]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    return None


def get_llm(model: Optional[str] = None, temperature: float = 0.0):
    """Return a configured LLM instance based on the provider."""
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    model_name = model or os.getenv('LLM_MODEL', 'gpt-4o-mini')

    if provider == 'openai':
        from langchain_openai import ChatOpenAI

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not configured in the .env file.\n"
                "Get one at: https://platform.openai.com/api-keys"
            )

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    elif provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not configured in the .env file.\n"
                "Get one at: https://aistudio.google.com/app/apikey"
            )

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )

    else:
        raise ValueError(
            f"Provider '{provider}' is not supported.\n"
            f"Set LLM_PROVIDER to 'openai' or 'google' in the .env file."
        )


def get_eval_llm(temperature: float = 0.0):
    """Return an evaluation-specific LLM instance using EVAL_MODEL."""
    eval_model = os.getenv('EVAL_MODEL', 'gpt-4o')
    return get_llm(model=eval_model, temperature=temperature)


def _is_rate_limit_error(error: Exception) -> bool:
    """Heuristic detection for provider/API rate-limit errors."""
    text = str(error).lower()
    markers = [
        "429",
        "too many requests",
        "rate limit",
        "rate_limit",
        "resource_exhausted",
        "quota exceeded",
    ]
    return any(marker in text for marker in markers)


def invoke_with_throttle_retry(callable_fn: Callable[[], T], context: str = "LLM call") -> T:
    """Invoke an LLM call with global pacing and retry/backoff.

    Environment variables (optional):
    - LLM_MIN_INTERVAL_SECONDS (default: 1.0)
    - LLM_MAX_RETRIES (default: 6)
    - LLM_BACKOFF_BASE_SECONDS (default: 2.0)
    - LLM_MAX_BACKOFF_SECONDS (default: 45.0)
    """
    min_interval = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "1.0"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "6"))
    backoff_base = float(os.getenv("LLM_BACKOFF_BASE_SECONDS", "2.0"))
    max_backoff = float(os.getenv("LLM_MAX_BACKOFF_SECONDS", "45.0"))

    global _last_invoke_timestamp

    for attempt in range(max_retries + 1):
        with _invoke_lock:
            now = time.time()
            elapsed = now - _last_invoke_timestamp
            wait_for = max(0.0, min_interval - elapsed)
            if wait_for > 0:
                time.sleep(wait_for)

            # Reserve slot timestamp before call to keep global pacing deterministic.
            _last_invoke_timestamp = time.time()

        try:
            return callable_fn()
        except Exception as exc:
            is_rate_limit = _is_rate_limit_error(exc)
            has_more_attempts = attempt < max_retries

            if not has_more_attempts or not is_rate_limit:
                raise

            backoff = min(max_backoff, backoff_base * (2 ** attempt))
            jitter = random.uniform(0, min(1.0, backoff * 0.25))
            sleep_time = backoff + jitter
            print(
                f"⚠️  {context}: limite de taxa detectado, tentativa {attempt + 1}/{max_retries + 1}. "
                f"Aguardando {sleep_time:.1f}s..."
            )
            time.sleep(sleep_time)

    # Defensive fallback (the loop either returns or raises).
    raise RuntimeError(f"{context}: falha inesperada no controle de retry")
