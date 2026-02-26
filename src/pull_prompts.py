"""Script that pulls prompts from the LangSmith Prompt Hub."""

import logging
import yaml
import sys
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langchain import hub

from utils import check_env_vars, print_section_header, save_yaml

load_dotenv()

PROMPT_ID = "leonanluppi/bug_to_user_story_v1"
RAW_OUTPUT_PATH = "prompts/raw_prompts.yml"
FINAL_OUTPUT_PATH = "prompts/bug_to_user_story_v1.yml"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def pull_prompts_from_langsmith():
    """Return the ChatPromptTemplate dictionary retrieved from LangSmith."""

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        logger.error("LANGSMITH_API_KEY is not set. Please define it in your .env file.")
        return 1

    print_section_header("Pulling Prompts from LangSmith Prompt Hub")
    logger.info("Starting LangSmith pull for %s", PROMPT_ID)

    try:
        prompt_obj = hub.pull(PROMPT_ID)
    except Exception as exc:
        logger.error(f"Error pulling prompt {PROMPT_ID}: {exc}")
        return None

    if not prompt_obj:
        logger.error("Prompt %s was not found.", PROMPT_ID)
        return None

    raw_payload = prompt_obj.model_dump()

    readable_messages: list[dict[str, Any]] = []
    for idx, message in enumerate(getattr(prompt_obj, "messages", []), start=1):
        prompt_section = getattr(message, "prompt", None)
        readable_messages.append(
            {
                "index": idx,
                "role": getattr(message, "role", None),
                "type": getattr(message, "type", None),
                "template": getattr(prompt_section, "template", None),
                "input_variables": getattr(prompt_section, "input_variables", None),
            }
        )

    raw_payload["messages_snapshot"] = readable_messages
    logger.info("ChatPromptTemplate readable messages:\n%s", yaml.safe_dump(readable_messages, allow_unicode=True, sort_keys=False))
    logger.info(f"Successfully pulled prompt: {PROMPT_ID}")
    logger.info("Raw prompt data keys: %s", list(raw_payload.keys()))
    return raw_payload


def extract_templates(prompt_dict: dict[str, Any]) -> tuple[str, str]:
    """Extracts system and user templates contained in the messages."""
    system_prompt = ""
    user_prompt = ""

    messages = prompt_dict.get("messages", []) or []

    def _extract_template(message: dict[str, Any]) -> str:
        prompt_section = message.get("prompt", {})
        return prompt_section.get("template") or message.get("template") or ""

    has_readable_messages = any(_extract_template(message) for message in messages)
    if not has_readable_messages:
        snapshot_messages = prompt_dict.get("messages_snapshot", []) or []
        if snapshot_messages:
            logger.info("Primary messages are empty/unreadable. Using messages_snapshot for template extraction.")
            messages = snapshot_messages

    logger.debug("Messages available for extraction: %d", len(messages))

    for idx, message in enumerate(messages, start=1):
        template = _extract_template(message)
        role = (message.get("role") or message.get("type") or "").lower()
        logger.debug(
            "Message %d | role=%s | template snippet=%s",
            idx,
            role,
            (template or "<empty>")[:80]
        )

        if not template:
            continue

        if "system" in role and not system_prompt:
            system_prompt = template
            logger.info("System prompt extracted from message %d", idx)
        elif any(keyword in role for keyword in ("human", "user")) and not user_prompt:
            user_prompt = template
            logger.info("User prompt extracted from message %d", idx)

        if not system_prompt and not user_prompt and idx == 1:
            system_prompt = template
            logger.info("System prompt defaulted to first template in message %d", idx)
        elif not user_prompt and template != system_prompt:
            user_prompt = template
            logger.info("User prompt defaulted to non-system template in message %d", idx)

    if not system_prompt:
        logger.warning("System prompt not identified via roles. Applying heuristic fallback.")
        for idx, message in enumerate(messages, start=1):
            template = _extract_template(message)
            if "you" in template.lower() or "assistant" in template.lower():
                system_prompt = template
                logger.info("System prompt identified via fallback in message %d", idx)
                break

    logger.debug("Final templates | system prompt len=%d | user prompt len=%d", len(system_prompt), len(user_prompt))
    return system_prompt or "", user_prompt or "{bug_report}"


def build_payload(prompt_dict: dict[str, Any]) -> dict[str, Any]:
    """Turn the hub response into the final YAML payload."""
    system_prompt, user_prompt = extract_templates(prompt_dict)
    metadata = prompt_dict.get("metadata", {})

    logger.info("Building final payload with metadata keys: %s", list(metadata.keys()))

    payload = {
        "bug_to_user_story_v1": {
            "description": metadata.get("description", "Prompt to convert bug reports into user stories."),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "metadata": metadata,
            "version": metadata.get("lc_hub_commit_hash", "v1"),
            "created_at": datetime.now(timezone.utc).isoformat()[:10],
            "tags": metadata.get("tags", ["bug-analysis", "user-story", "product-management"]),
        }
    }

    logger.debug("Payload ready for serialization: %s", payload["bug_to_user_story_v1"].keys())
    return payload


def main() -> int:
    """Entry point for the script."""
    prompt_dict = pull_prompts_from_langsmith()
    if not prompt_dict:
        logger.error("Failed to pull prompt data. Exiting.")
        return 1

    save_yaml(prompt_dict, RAW_OUTPUT_PATH)
    payload = build_payload(prompt_dict)
    save_yaml(payload, FINAL_OUTPUT_PATH)

    print("Prompts pulled and saved successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
