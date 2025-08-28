# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import tokenizers
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.cache_management import cached
from lighteval.utils.imports import is_litellm_available

logger = logging.getLogger(__name__)

if is_litellm_available():
    import litellm
    from litellm import encode
    from litellm.caching.caching import Cache, LiteLLMCacheType
    from litellm.utils import ModelResponse as LitellmModelResponse
    from litellm.utils import create_pretrained_tokenizer

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").handlers.clear()

    litellm.cache = Cache(type=LiteLLMCacheType.DISK)
else:
    from unittest.mock import Mock

    litellm = Mock()
    encode = Mock()
    LitellmModelResponse = Mock()


class LiteLLMModelConfig(ModelConfig):
    """Configuration class for LiteLLM unified API client.

    This configuration is used to connect to various LLM providers through the LiteLLM
    unified API. LiteLLM provides a consistent interface to multiple providers including
    OpenAI, Anthropic, Google, and many others.

    litellm doc: https://docs.litellm.ai/docs/

    Attributes:
        model_name (str):
            Model identifier. Can include provider prefix (e.g., "gpt-4", "claude-3-sonnet")
            or use provider/model format (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet").
        provider (str | None):
            Optional provider name override. If None, inferred from model_name.
            Examples: "openai", "anthropic", "google", "cohere", etc.
        base_url (str | None):
            Custom base URL for the API. If None, uses provider's default URL.
            Useful for using custom endpoints or local deployments.
        api_key (str | None):
            API key for authentication. If None, reads from environment variables.
            Environment variable names are provider-specific (e.g., OPENAI_API_KEY).
        concurrent_requests (int):
            Maximum number of concurrent API requests to execute in parallel.
            Higher values can improve throughput for batch processing but may hit rate limits
            or exhaust API quotas faster. Default is 10.
        generation_parameters (GenerationParameters, optional, defaults to empty GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc.
        system_prompt (str | None, optional, defaults to None): Optional system prompt to be used with chat models.
            This prompt sets the behavior and context for the model during evaluation.
        cache_dir (str, optional, defaults to "~/.cache/huggingface/lighteval"): Directory to cache the model.

    Example:
        ```python
        config = LiteLLMModelConfig(
            model_name="gpt-4",
            provider="openai",
            base_url="https://api.openai.com/v1",
            concurrent_requests=5,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    model_name: str
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    concurrent_requests: int = 10
    custom_huggingface_tokenizer: Optional[str] = None
    api_base: Optional[str] = None
    use_cache: Optional[bool] = True
    generation_parameters: Optional[GenerationParameters] = None

    concurrent_calls: Optional[int] = 20  # 100 leads to hitting Anthropic rate limits
    api_max_retry: Optional[int] = 8
    api_retry_sleep: Optional[int] = 1
    api_retry_multiplier: Optional[int] = 2
    timeout: Optional[float] = None

    success_callback: Optional[List[Union[str, Callable]]] = None
    failure_callback: Optional[List[Union[str, Callable]]] = None

    def __post_init__(self):
        if not self.generation_parameters:
            self.generation_parameters = GenerationParameters()

    @classmethod
    def from_path(cls, path: str) -> "LiteLLMModelConfig":
        import yaml

        with open(path, "r") as f:
            config = yaml.safe_load(f)["model"]
        generation_parameters = GenerationParameters.from_dict(config)
        return cls(
            model=config["model_name"],
            provider=config["provider"],
            api_base=config["api_base"],
            use_cache=config["use_cache"],
            custom_huggingface_tokenizer=config["custom_huggingface_tokenizer"],
            concurrent_calls=config["concurrent_calls"],
            api_max_retry=config["api_max_retry"],
            api_retry_sleep=config["api_retry_sleep"],
            api_retry_multiplier=config["api_retry_multiplier"],
            timeout=config["timeout"],
            success_callback=config["success_callback"],
            failure_callback=config["failure_callback"],
            generation_parameters=generation_parameters,
        )


class LiteLLMClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config: LiteLLMModelConfig) -> None:
        """IMPORTANT: Your API keys should be set in the environment variables.
        If a base_url is not set, it will default to the public API.
        """
        self.generation_parameters = config.generation_parameters
        self.sampling_params = self.generation_parameters.to_litellm_dict()
        self.use_cache = config.use_cache

        # TODO: remove and just use base_url with environment variable
        self.api_base = config.api_base

        if config.custom_huggingface_tokenizer:
            logger.info("Using custom hugging face tokenizer from repository %s", config.custom_huggingface_tokenizer)
            self.custom_tokenizer = create_pretrained_tokenizer(config.custom_huggingface_tokenizer)
        else:
            self.custom_tokenizer = None

        self.config = config
        self.model = config.model_name
        self.provider = config.provider or config.model_name.split("/")[0]
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.generation_parameters = config.generation_parameters
        self.concurrent_requests = config.concurrent_requests
        self.model_info = ModelInfo(
            model_name=config.model,
            model_sha="",
            model_dtype=None,
            model_size="",
        )

        self.API_MAX_RETRY = config.api_max_retry
        self.API_RETRY_SLEEP = config.api_retry_sleep
        self.API_RETRY_MULTIPLIER = config.api_retry_multiplier
        self.CONCURENT_CALLS = config.concurrent_calls
        self.timeout = config.timeout

        self.model = config.model
        self._tokenizer = encode
        self.pairwise_tokenization = False
        litellm.drop_params = True
        litellm.verbose = True

        if config.success_callback:
            litellm.success_callback = config.success_callback
            config.success_callback = None  # Avoid error during generation of final dict in evaluation tracker

        if config.failure_callback:
            litellm.failure_callback = config.failure_callback
            config.failure_callback = None  # Avoid error during generation of final dict in evaluation tracker

    def _prepare_stop_sequence(self, stop_sequence):
        """Prepare and validate stop sequence."""
        if self.provider == "anthropic":
            # Filter out whitespace-only stop sequences
            if stop_sequence:
                stop_sequence = [s for s in stop_sequence if s and s.strip()]
        return stop_sequence

    def _prepare_max_new_tokens(self, max_new_tokens):
        """Calculate completion tokens based on max_new_tokens."""
        if not max_new_tokens or max_new_tokens <= 0:
            return None

        if any([s in self.model for s in ["o1", "o3", "deepseek-reasoner", "R1"]]):

            if "deepseek-reasoner" in self.model:
                upper_bound = 8192
            else:
                upper_bound = 32000

            # We need to allow more tokens to include reasoning tokens
            max_new_tokens = min(max_new_tokens * 10, upper_bound)

            logger.warning(
                "Reasoning model detected, increasing max_new_tokens to %d to allow for reasoning tokens",
                max_new_tokens,
            )
        return max_new_tokens

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence):  # noqa: C901
        """Make API call with retries."""
        if num_samples > 1 and self.generation_parameters.temperature == 0:
            raise ValueError("num_samples > 1 but temperature is set to 0, this will not sample different outputs.")

        response = LitellmModelResponse()

        stop_sequence = self._prepare_stop_sequence(stop_sequence)
        max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)

        if return_logits and not self.provider == "openai":
            logger.warning("Returning logits is not supported for this provider, ignoring.")

        # Prepare kwargs for completion call
        kwargs = {
            "model": self.model,
            "messages": prompt,
            "response_format": {"type": "text"},
            "max_tokens": max_new_tokens if max_new_tokens else None,
            "logprobs": return_logits if self.provider == "openai" else None,
            "stop": stop_sequence,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "n": num_samples,
            "caching": self.use_cache,
            "timeout": self.timeout,
            **self.sampling_params,
        }

        if "o1" in self.model:
            logger.warning("O1 models do not support temperature, top_p, stop sequence. Disabling.")
        else:
            kwargs.update(self.generation_parameters.to_litellm_dict())

        if kwargs.get("max_completion_tokens", None) is None:
            kwargs["max_completion_tokens"] = max_new_tokens

        if self.api_base:
            kwargs["api_base"] = self.api_base

        for attempt in range(self.API_MAX_RETRY):
            try:
                response = litellm.completion(**kwargs)
                content = response.choices[0].message.content

                # If response is empty, retry without caching (maybe the error is recoverable and solved with a retry)
                if not content:
                    logger.info("Response is empty, retrying without caching")
                    kwargs["caching"] = False
                    response = litellm.completion(**kwargs)
                    content = response.choices[0].message.content

                if content and "<think>" in content:
                    logger.debug(f"Removing <think> tags from response: {content}")
                    response.choices[0].message.content = re.sub(
                        r"<think>.*?</think>", "", content, flags=re.DOTALL
                    ).strip()

                return response
            except litellm.BadRequestError as e:
                if "message" in e.__dict__:
                    error_string = (
                        "The response was filtered due to the prompt triggering Microsoft's content management policy"
                    )
                    if error_string in e.__dict__["message"]:
                        logger.warning(f"{error_string}. Returning empty response.")
                        return LitellmModelResponse()
            except Exception as e:
                wait_time = min(
                    64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
                )  # Exponential backoff with max 64s
                logger.warning(
                    f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)

        logger.error(f"API call failed after {self.API_MAX_RETRY} attempts, returning empty response.")
        return LitellmModelResponse()

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int] | None,
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
    ):
        results = []

        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        stop_sequencess = [stop_sequence for _ in prompts]
        assert (
            len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(stop_sequencess)
        ), f"Length of prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequences, system_prompts should be the same but are {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, {len(num_sampless)}, {len(stop_sequencess)}"

        with ThreadPoolExecutor(self.concurrent_requests) as executor:
            for entry in tqdm(
                executor.map(
                    self.__call_api,
                    prompts,
                    return_logitss,
                    max_new_tokenss,
                    num_sampless,
                    stop_sequencess,
                ),
                total=len(prompts),
            ):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        return results

    @cached("predictions")
    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): List of documents containing the context for generation.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            contexts = [self.prompt_manager.prepare_prompt_api(doc) for doc in dataset]
            max_new_tokens = split[0].generation_size  # could be none
            return_logits = split[0].use_logits
            num_samples = split[0].num_samples
            stop_sequence = split[0].stop_sequences

            if num_samples > 1 and self.generation_parameters.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )

            responses = self.__call_api_parallel(contexts, return_logits, max_new_tokens, num_samples, stop_sequence)

            for response, context in zip(responses, contexts):
                result: list[str] = [choice.message.content for choice in response.choices]
                reasonings: list[str | None] = [
                    getattr(choice.message, "reasoning_content", None) for choice in response.choices
                ]

                cur_response = ModelResponse(
                    # In empty responses, the model should return an empty string instead of None
                    text=result if result[0] else [""],
                    reasonings=reasonings,
                    input=context,
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

    def tok_encode(self, text: str | list[str]):
        if isinstance(text, list):
            toks = [encode(model=self.model, text=t["content"], custom_tokenizer=self.custom_tokenizer) for t in text]
            toks = [tok for tok in toks if tok]
        else:
            toks = encode(model=self.model, text=text, custom_tokenizer=self.custom_tokenizer)

        # Handle Hugging Face tokenizers
        if isinstance(toks, list) and len(toks) > 0 and isinstance(toks[0], tokenizers.Encoding):
            toks = [t.ids for t in toks]
        if isinstance(toks, tokenizers.Encoding):
            toks = toks.ids

        return toks

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        return 4096

    @cached("predictions")
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    @cached("predictions")
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError
