from typing import Callable
from openai import AzureOpenAI

from judge.configs import get_trapi_gpt4_config, get_trapi_gpt35_config


def test_endpoint(
    credential: Callable,
    api_version: str,
    endpoint: str,
    deployment_name: str,
    **kwargs
) -> str:
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    )

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "user",
                "content": "Give a one word answer, what is the capital of France?",
            },
        ],
    )
    response_content = response.choices[0].message.content
    return response_content


# print(test_endpoint(**get_trapi_gpt35_config()))
print(test_endpoint(**get_trapi_gpt4_config()))
