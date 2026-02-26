"""Push optimized prompts to the LangSmith Prompt Hub.

This script:
1. Loads optimized prompts from prompts/bug_to_user_story_v2.yml
2. Validates prompts
3. Pushes prompts as PUBLIC to LangSmith Hub
4. Adds metadata (tags, description, techniques used)

SIMPLIFIED: Cleaner and more direct implementation.
"""

import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

try:
    from .utils import load_yaml, check_env_vars, print_section_header
except ImportError:
    from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROMPTS_FILE = "prompts/bug_to_user_story_v2.yml"
REQUIRED_ENV_VARS = ["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]

KEY_MESSAGES = "messages"
KEY_ROLE = "role"
KEY_CONTENT = "content"
KEY_SYSTEM_PROMPT = "system_prompt"
KEY_USER_PROMPT = "user_prompt"
KEY_EXAMPLES = "examples"
KEY_EXAMPLES_LEGACY = "few_shot_examples"
KEY_BUG_REPORT = "bug_report"
KEY_USER_STORY = "user_story"

ROLE_ALIASES = {
    "user": "human",
    "human": "human",
    "assistant": "ai",
    "ai": "ai",
    "system": "system",
}

BUG_REPORT_MESSAGE_TEMPLATE = "Bug report:\n---\n{bug_report}\n---"
DEFAULT_DESCRIPTION = "Optimized prompt"
DEFAULT_TAGS = ["bug-to-user-story", "prompt-optimization"]


def _resolve_examples(prompt_data: dict[str, Any]) -> Any:
    """Return examples from the new schema or legacy schema for backward compatibility."""
    examples = prompt_data.get(KEY_EXAMPLES)
    if examples is None:
        examples = prompt_data.get(KEY_EXAMPLES_LEGACY, [])
    return examples


def build_chat_prompt_template(prompt_data: dict[str, Any]) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate from v2 schema (or raw messages when provided)."""
    messages: list[tuple[str, str]] = []

    raw_messages = prompt_data.get(KEY_MESSAGES)
    if isinstance(raw_messages, list) and raw_messages:
        for i, msg in enumerate(raw_messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Invalid message at index {i} in '{KEY_MESSAGES}'.")

            role = (msg.get(KEY_ROLE) or "").strip().lower()
            content = (msg.get(KEY_CONTENT) or "").strip()
            if not role or not content:
                raise ValueError(f"Message {i} must contain non-empty role and content.")

            normalized_role = ROLE_ALIASES.get(role)
            if not normalized_role:
                raise ValueError(f"Invalid role in message {i}: {role}")

            messages.append((normalized_role, content))

        return ChatPromptTemplate.from_messages(messages)

    system_prompt = (prompt_data.get(KEY_SYSTEM_PROMPT) or "").strip()
    user_prompt = (prompt_data.get(KEY_USER_PROMPT) or "").strip()

    if not system_prompt or not user_prompt:
        raise ValueError(
            f"Prompt must contain non-empty '{KEY_SYSTEM_PROMPT}' and '{KEY_USER_PROMPT}'."
        )

    messages.append(("system", system_prompt))

    examples = _resolve_examples(prompt_data)

    if isinstance(examples, list):
        for idx, example in enumerate(examples):
            if not isinstance(example, dict):
                logger.warning("Ignoring invalid few-shot example at index %d", idx)
                continue

            bug_report = (example.get(KEY_BUG_REPORT) or "").strip()
            user_story = (example.get(KEY_USER_STORY) or "").strip()

            if not bug_report or not user_story:
                logger.warning("Ignoring incomplete few-shot example at index %d", idx)
                continue

            messages.append(("human", BUG_REPORT_MESSAGE_TEMPLATE.format(bug_report=bug_report)))
            messages.append(("ai", user_story))

    messages.append(("human", user_prompt))
    return ChatPromptTemplate.from_messages(messages)


def build_push_metadata(prompt_data: dict[str, Any]) -> tuple[str, list[str]]:
    """Extract Hub description and tags from prompt YAML metadata."""
    meta = prompt_data.get("metadata", {}) if isinstance(prompt_data.get("metadata"), dict) else {}

    description = (prompt_data.get("description") or "").strip() or DEFAULT_DESCRIPTION
    techniques = prompt_data.get("techniques_applied", [])
    if not isinstance(techniques, list):
        techniques = []

    if techniques:
        description = f"{description} | Techniques: {', '.join(str(t) for t in techniques)}"

    tags: list[str] = []
    for source in (meta.get("tags", []), prompt_data.get("tags", [])):
        if isinstance(source, list):
            tags.extend(str(tag).strip() for tag in source if str(tag).strip())

    if not tags:
        tags = DEFAULT_TAGS.copy()

    unique_tags = list(dict.fromkeys(tags))
    return description, unique_tags

def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Push an optimized prompt to the LangSmith Hub (PUBLIC).

    Args:
        prompt_name: Prompt name
        prompt_data: Prompt payload

    Returns:
        True on success, False otherwise
    """
    try:
        username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
        repo_full_name = f"{username}/{prompt_name}"

        prompt_template = build_chat_prompt_template(prompt_data)
        description, tags = build_push_metadata(prompt_data)

        push_url = hub.push(
            repo_full_name=repo_full_name,
            object=prompt_template,
            new_repo_is_public=True,
            new_repo_description=description,
            tags=tags,
        )

        logger.info("Prompt '%s' pushed successfully! URL: %s", prompt_name, push_url)
        return True
    except Exception as e:
        logger.error(f"Error pushing prompt '{prompt_name}': {e}")
        return False


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Validate basic prompt structure (simplified version).

    Args:
        prompt_data: Prompt payload

    Returns:
        (is_valid, errors) - Status tuple with error list
    """
    errors = []
    if not isinstance(prompt_data, dict):
        errors.append("Prompt payload is not a valid dictionary.")
        return False, errors

    has_messages = KEY_MESSAGES in prompt_data
    has_system_user = KEY_SYSTEM_PROMPT in prompt_data and KEY_USER_PROMPT in prompt_data

    if not has_messages and not has_system_user:
        errors.append("Prompt must contain 'messages' or ('system_prompt' and 'user_prompt').")
        return False, errors

    if has_messages:
        if not isinstance(prompt_data[KEY_MESSAGES], list):
            errors.append(f"Field '{KEY_MESSAGES}' is not a valid list.")
            return False, errors

        for i, msg in enumerate(prompt_data[KEY_MESSAGES]):
            if not isinstance(msg, dict):
                errors.append(f"Message {i} is not a valid dictionary.")
                return False, errors
            if KEY_ROLE not in msg:
                errors.append(f"Message {i} is missing '{KEY_ROLE}'.")
                return False, errors
            if KEY_CONTENT not in msg:
                errors.append(f"Message {i} is missing '{KEY_CONTENT}'.")
                return False, errors

    if has_system_user:
        if not str(prompt_data.get(KEY_SYSTEM_PROMPT, "")).strip():
            errors.append(f"Field '{KEY_SYSTEM_PROMPT}' is empty.")
            return False, errors
        if not str(prompt_data.get(KEY_USER_PROMPT, "")).strip():
            errors.append(f"Field '{KEY_USER_PROMPT}' is empty.")
            return False, errors

    examples = _resolve_examples(prompt_data)

    if examples and not isinstance(examples, list):
        errors.append("Field 'examples' must be a list.")
        return False, errors

    if isinstance(examples, list):
        for i, example in enumerate(examples):
            if not isinstance(example, dict):
                errors.append(f"Example {i} is not a valid dictionary.")
                return False, errors
            if not str(example.get(KEY_BUG_REPORT, "")).strip():
                errors.append(f"Example {i} is missing '{KEY_BUG_REPORT}'.")
                return False, errors
            if not str(example.get(KEY_USER_STORY, "")).strip():
                errors.append(f"Example {i} is missing '{KEY_USER_STORY}'.")
                return False, errors

    return True, []


def main():
    """Main entry point."""
    print_section_header("Starting optimized prompt push process")

    # Validate required environment variables
    if not check_env_vars(REQUIRED_ENV_VARS):
        logger.error("Required environment variables are not set. Aborting.")
        return 1

    # Load optimized prompts from YAML file
    if not os.path.exists(PROMPTS_FILE):
        logger.error(f"Prompts file '{PROMPTS_FILE}' not found. Aborting.")
        return 1

    prompts_data = load_yaml(PROMPTS_FILE)
    if not isinstance(prompts_data, dict):
        logger.error(
            "Invalid file structure in '%s'. Expected a prompt dictionary. Aborting.",
            PROMPTS_FILE,
        )
        return 1

    success_count = 0
    total_prompts = len(prompts_data)

    # Process each prompt
    for prompt_name, prompt_data in prompts_data.items():
        print_section_header(f"Processing prompt: {prompt_name}")

        # Validate prompt
        is_valid, errors = validate_prompt(prompt_data)
        if not is_valid:
            logger.warning(f"Prompt '{prompt_name}' is invalid. Errors found:")
            for error in errors:
                logger.warning(f"   - {error}")
            continue  # Skip to next prompt

        # Push prompt to LangSmith Hub
        if push_prompt_to_langsmith(prompt_name, prompt_data):
            success_count += 1

    logger.info("Push finished: %d/%d prompts sent successfully.", success_count, total_prompts)
    return 0 if success_count == total_prompts else 1


if __name__ == "__main__":
    sys.exit(main())
