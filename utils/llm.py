"""
LLM 호출 유틸리티 — OpenAI / Anthropic 공통 인터페이스
LexiChinese 플랫폼용
"""
from openai import OpenAI
import anthropic


def call_openai(system_prompt: str, user_prompt: str, api_key: str,
                model: str = "gpt-5-mini-2025-08-07") -> str:
    client = OpenAI(api_key=api_key)
    # 구형 모델: max_tokens + temperature 지원
    # 신형 모델(gpt-4o 이후): max_completion_tokens, temperature=1(기본값)만 허용
    _legacy_models = ("gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o")
    is_legacy = any(model.startswith(m) for m in _legacy_models)
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens" if is_legacy else "max_completion_tokens": 4096,
    }
    if is_legacy:
        params["temperature"] = 0.3
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content.strip()


def call_claude(system_prompt: str, user_prompt: str, api_key: str,
                model: str = "claude-haiku-4-5") -> str:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text.strip()


def call_llm(system_prompt: str, user_prompt: str,
             provider: str, api_key: str, model: str) -> str:
    """통합 LLM 호출 — provider: 'OpenAI' 또는 'Anthropic'"""
    if provider == "OpenAI":
        return call_openai(system_prompt, user_prompt, api_key, model)
    else:
        return call_claude(system_prompt, user_prompt, api_key, model)
