"""
LLM 호출 유틸리티 — OpenAI / Anthropic 공통 인터페이스
LexiChinese 플랫폼용
"""
from openai import OpenAI
import anthropic


def call_openai(system_prompt: str, user_prompt: str, api_key: str,
                model: str = "gpt-4o") -> str:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


def call_claude(system_prompt: str, user_prompt: str, api_key: str,
                model: str = "claude-4-sonnet-20250514") -> str:
    client = anthropic.Anthropic(api_key=api_key)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text.strip()
    except anthropic.BadRequestError as e:
        # 크레딧 부족 등 400 에러 처리
        err_msg = str(e)
        if "credit balance is too low" in err_msg.lower() or "credit" in err_msg.lower():
            return (
                "⚠️ **크레딧 만료. 추가 크레딧을 구매해 주세요.**\n\n"
                "Anthropic API 크레딧 잔액이 부족하여 요청을 처리할 수 없습니다. "
                "[Anthropic Console](https://console.anthropic.com/settings/billing)에서 "
                "플랜을 업그레이드하거나 크레딧을 충전해 주세요.\n\n"
                f"`anthropic.BadRequestError`: {err_msg}"
            )
        return (
            "⚠️ **크레딧 만료. 추가 크레딧을 구매해 주세요.**\n\n"
            f"`anthropic.BadRequestError`: {err_msg}"
        )


def call_llm(system_prompt: str, user_prompt: str,
             provider: str, api_key: str, model: str) -> str:
    """통합 LLM 호출 — provider: 'OpenAI' 또는 'Anthropic'"""
    if provider == "OpenAI":
        return call_openai(system_prompt, user_prompt, api_key, model)
    else:
        return call_claude(system_prompt, user_prompt, api_key, model)
