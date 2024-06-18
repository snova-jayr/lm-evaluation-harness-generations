import copy
import os
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple

from tqdm import tqdm

import lm_eval.models.utils
from lm_eval import utils
from lm_eval.api.model import LM, TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions
from lm_eval.utils import eval_logger
import requests
import json
from groq import Groq

def invoke_predict(inps, tuningParams, client):
    chat_completion = client.chat.completions.create(
        messages=inps,
        model="llama3-8b-8192",
        temperature=0.0,
        max_tokens=1000,
    )

    response_text = chat_completion.choices[0].message.content
    return [response_text]

def invoke_model(inps, tuningParams, client):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    responses = invoke_predict(inps=inps, tuningParams=tuningParams, client=client)
    return responses

@register_model("groq")
class GroqChatCompletionsLM(LM):
    def __init__(
        self,
        model: str = "llama3-8b",  # GPT model or Local model using HuggingFace model paths
        base_url: str = None,
        truncate: bool = False,
        **kwargs,
    ) -> None:
        """

        :param model: str
            Implements an OpenAI-style chat completion API for
            accessing both OpenAI OR locally-hosted models using
            HuggingFace Tokenizer
            OpenAI API model (e.g. gpt-3.5-turbo)
            using the **gen_kwargs passed on init
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        
        # TO-DO add all of your keys
        key = os.environ['GROQ_API_KEY']
        self.client = Groq(api_key=key)    
        
        self.model = model
        self.base_url = base_url
        self.truncate = truncate

    @property
    def max_length(self) -> int:
        return 4095

    @property
    def max_gen_toks(self) -> int:
        return 512

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)))
        for request in requests:
            contexts, gen_kwargs = request.args

            until = None
            if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                if "do_sample" in kwargs.keys():
                    kwargs.pop("do_sample")
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected repr(kwargs['until']) to be of type Union[str, list] but got {until}"
                        )
                    kwargs["stop"] = until
                kwargs["max_tokens"] = kwargs.pop("max_gen_toks", self.max_gen_toks)
            else:
                raise ValueError(
                    f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                )

            while True:
                try:
                    response = invoke_model(contexts, kwargs, self.client)
                except:
                    continue
                break
            s = response[0]

            if until is not None:
                for term in until:
                    if len(term) > 0:
                        s = s.split(term)[0]
            res.append(s)
            self.cache_hook.add_partial("generate_until", request, s)
            pbar.update(1)

        pbar.close()

        return res

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")