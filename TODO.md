# Detailed Roadmap - Prompt Engineering Challenge

## 1. Setup & Environment 🛠️
- [x] Create and activate a virtual environment: `python3 -m venv venv && source venv/bin/activate`.
- [x] Install dependencies: `pip install -r requirements.txt`.
- [x] Configure `.env`:
    - [x] `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_ENDPOINT="https://api.smith.langchain.com"`.
    - [x] `LLM_PROVIDER` (openai or google).
    - [x] `OPENAI_API_KEY` or `GOOGLE_API_KEY`.
    - [x] `LLM_MODEL="gpt-4o-mini"` (for responses).
    - [x] `EVAL_MODEL="gpt-4o"` (for evaluation).

## 2. Phase 1: Pull the Base Prompt 📥
- [x] Implement `src/pull_prompts.py`:
    - [x] Use `langchain.hub.pull("leonanluppi/bug_to_user_story_v1")`.
    - [x] Extract the prompt templates from the returned object.
    - [x] Persist to `prompts/bug_to_user_story_v1.yml`.
- [x] Confirm the YAML file was created with the hub content.

## 3. Phase 2: Prompt Optimization (The Heart of the Challenge) 🧠
- [x] Analyze the `v1` version (understand why it is low quality).
- [x] Create `prompts/bug_to_user_story_v2.yml` applying at least two techniques:
    - [x] **Role Prompting**: Define a detailed persona (e.g., Senior PM).
    - [x] **Few-shot Learning**: Include 2-3 real Bug Report → User Story examples.
    - [x] **Chain of Thought (CoT)**: Prompt the model to think step-by-step.
    - [x] **Skeleton of Thought**: Structure the output into clear segments.
- [x] **Prompt V2 Requirements**:
    - [x] Clear, specific instructions.
    - [x] Explicit behavioral rules.
    - [x] Edge case coverage.
    - [x] Appropriate usage of System vs User prompt.

## 4. Phase 3: Push to the LangSmith Hub 📤
- [x] Implement `src/push_prompts.py`:
    - [x] Load `bug_to_user_story_v2.yml`.
    - [x] Add metadata: tags, description, techniques used.
    - [x] Run `hub.push("{your_username}/bug_to_user_story_v2", prompt_object, public=True)`.
- [x] Confirm the prompt is publicly accessible on the LangSmith dashboard.

## 5. Phase 4: Iterative Evaluation 🔄
- [ ] Run `python src/evaluate.py`.
- [ ] Goal: Achieve ≥ 0.9 in EVERY metric:
    - [ ] `Tone Score`
    - [ ] `Acceptance Criteria Score`
    - [ ] `User Story Format Score`
    - [ ] `Completeness Score`
- [ ] Iteration loop (3-5 times):
    - [ ] Review logs in LangSmith tracing.
    - [ ] Identify gaps (e.g., missing acceptance criteria).
    - [ ] Update YAML → Push → Evaluate.

## 6. Phase 5: Validation Tests (Pytest) ✅
- [x] Implement in `tests/test_prompts.py`:
    - [x] `test_prompt_has_system_prompt`: Exists and is not empty.
    - [x] `test_prompt_has_role_definition`: Persona is defined.
    - [x] `test_prompt_mentions_format`: Enforces Markdown or User Story format.
    - [x] `test_prompt_has_few_shot_examples`: Contains example pairs.
    - [x] `test_prompt_no_todos`: Removes `[TODO]` markers.
    - [x] `test_minimum_techniques`: Metadata lists at least 2 techniques.
- [x] Run `pytest tests/test_prompts.py`.

## 7. Phase 6: Documentation & Delivery 📄
- [ ] Update `README.md`:
    - [ ] **Applied Techniques** section with rationale.
    - [ ] **Final Results** section with LangSmith links & ≥0.9 screenshots.
    - [ ] Comparison table: `v1` vs `v2`.
    - [ ] **How to Run** instructions.
- [ ] Evidence on LangSmith:
    - [ ] Dataset with ≥ 20 examples (or provided `.jsonl`).
    - [ ] Tracing for at least 3 runs.
- [ ] Ensure the GitHub repository is public.
