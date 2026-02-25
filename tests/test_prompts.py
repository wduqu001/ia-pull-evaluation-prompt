"""Testes automatizados para validação do prompt otimizado."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure
from pull_prompts import extract_templates


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(name="prompt_data")
def fixture_prompt_data():
    """Retorna os dados do prompt otimizado."""
    prompts = load_prompts("prompts/bug_to_user_story_v2.yml")
    assert prompts, "Arquivos de prompts não foram carregados"
    prompt = prompts.get("bug_to_user_story_v2")
    assert prompt, "O prompt 'bug_to_user_story_v2' não existe"
    return prompt


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt_data):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        system_prompt = prompt_data.get("system_prompt", "").strip()
        assert system_prompt, "system_prompt não pode estar vazio"

    def test_prompt_has_role_definition(self, prompt_data):
        """Verifica se o prompt define uma persona clara."""
        system_prompt = prompt_data.get("system_prompt", "")
        role_keywords = ["Product Manager", "PM Sênior", "Product Leader"]
        assert any(keyword in system_prompt for keyword in role_keywords), (
            "system_prompt deve definir uma persona (ex: Product Manager)"
        )

    def test_prompt_mentions_format(self, prompt_data):
        """Verifica se o prompt exige um formato de User Story ou Markdown."""
        text = "\n".join([prompt_data.get("system_prompt", ""), prompt_data.get("user_prompt", "")])
        format_keywords = ["User Story", "##", "Formato", "Critérios de aceitação"]
        assert any(keyword in text for keyword in format_keywords), (
            "O prompt deve mencionar o formato da resposta (User Story/Markdown)"
        )

    def test_prompt_has_few_shot_examples(self, prompt_data):
        """Verifica se o prompt contém exemplos Few-shot."""
        examples = prompt_data.get("few_shot_examples", [])
        assert isinstance(examples, list) and len(examples) >= 2, (
            "Devem existir ao menos dois exemplos few-shot"
        )
        for example in examples:
            assert example.get("bug_report"), "Cada exemplo deve ter 'bug_report'"
            assert example.get("user_story"), "Cada exemplo deve ter 'user_story'"

    def test_prompt_no_todos(self, prompt_data):
        """Garante que não existem marcadores [TODO] no prompt otimizado."""
        combined = yaml.dump(prompt_data)
        assert "[TODO]" not in combined, "Remova todos os [TODO] do prompt"

    def test_minimum_techniques(self, prompt_data):
        """Verifica se há pelo menos duas técnicas documentadas nos metadados."""
        techniques = prompt_data.get("techniques_applied", [])
        assert isinstance(techniques, list) and len(techniques) >= 2, (
            "Metadados devem listar ao menos duas técnicas"
        )

    def test_prompt_structure_validation(self, prompt_data):
        """Garante que a validação auxiliar da estrutura retorna sucesso."""
        is_valid, errors = validate_prompt_structure(prompt_data)
        assert is_valid, f"Validação falhou: {errors}"


class TestPullPromptExtraction:
    def test_extract_templates_uses_messages_snapshot_when_messages_are_unreadable(self):
        """Usa messages_snapshot quando messages vier vazio/sem template (ex: [{}, {}])."""
        prompt_dict = {
            "messages": [{}, {}],
            "messages_snapshot": [
                {"template": "System from snapshot"},
                {"template": "{bug_report}"},
            ],
        }

        system_prompt, user_prompt = extract_templates(prompt_dict)

        assert system_prompt == "System from snapshot"
        assert user_prompt == "{bug_report}"

    def test_extract_templates_prefers_primary_messages_when_readable(self):
        """Mantém prioridade para messages quando eles já contêm templates válidos."""
        prompt_dict = {
            "messages": [
                {"prompt": {"template": "System from messages"}},
                {"prompt": {"template": "User from messages"}},
            ],
            "messages_snapshot": [
                {"template": "System from snapshot"},
                {"template": "User from snapshot"},
            ],
        }

        system_prompt, user_prompt = extract_templates(prompt_dict)

        assert system_prompt == "System from messages"
        assert user_prompt == "User from messages"

    def test_extract_templates_keeps_default_user_when_only_one_template_exists(self):
        """Se houver só um template, o user_prompt deve permanecer no default esperado."""
        prompt_dict = {
            "messages": [{}],
            "messages_snapshot": [
                {"template": "Only system template"},
            ],
        }

        system_prompt, user_prompt = extract_templates(prompt_dict)

        assert system_prompt == "Only system template"
        assert user_prompt == "{bug_report}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
