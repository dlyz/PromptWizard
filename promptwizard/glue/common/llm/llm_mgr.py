from typing import Callable, Dict, Protocol
from ..base_classes import LLMConfig
from ..constants.str_literals import (
    InstallLibs,
    OAILiterals,
    OAILiterals,
    LLMLiterals,
    LLMOutputTypes,
)
from ..exceptions import GlueLLMException
from ..utils.logging import get_glue_logger
import os

logger = get_glue_logger(__name__)


def getClientAndModel(suffix=""):
    def get_env_prefer_suffix(key: str, default=None):
        return os.environ.get(key + suffix) or os.environ.get(key, default)

    def get_env_prefer_suffix_required(key: str):
        return os.environ.get(key + suffix) or os.environ[key]

    model_type = get_env_prefer_suffix("MODEL_TYPE", "openai").lower()
    model_id = get_env_prefer_suffix_required("MODEL_NAME")

    if model_type == "openai":
        from openai import OpenAI

        return OpenAI(
            api_key=get_env_prefer_suffix("OPENAI_API_KEY"),
            base_url=get_env_prefer_suffix("OPENAI_BASE_URL"),
        ), model_id
    elif model_type == "azureopenai":
        from azure.identity import get_bearer_token_provider, AzureCliCredential
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(
            AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return AzureOpenAI(
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_ad_token_provider=token_provider,
        ), model_id
    else:
        raise ValueError(f"unknown model type '{model_type}'")


# Main logic
def default_chat_model(messages, *, is_prod_model: bool = False):
    client, model_name = getClientAndModel(suffix=("_PROD" if is_prod_model else ""))
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )

    prediction = response.choices[0].message.content
    return prediction


class ChatModelProto(Protocol):
    def __call__(self, messages: list[dict], *, is_prod_model: bool = False) -> str: ...

# class LLMMgr:
#     @staticmethod
#     def chat_completion(messages: list[dict], *, is_prod_model: bool = False):
#         try:
#             return call_api(messages, is_prod_model=is_prod_model)
#         except Exception as e:
#             print(e)
#             return "Sorry, I am not able to understand your query. Please try again."
#             # raise GlueLLMException(f"Exception when calling {llm_handle.__class__.__name__} "
#             #                        f"LLM in chat mode, with message {messages} ", e)
