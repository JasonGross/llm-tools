# %%
import inspect
import random
from typing import Optional

__all__ = ["wrap_with_equivalent_models", "EQUIVALENT_MODELS"]

EQUIVALENT_MODELS = [
    [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-16k-0613",
    ],
    ["gpt-4", "gpt-4-0613"],
    ["gpt-4-32k", "gpt-4-32k-0613"],
]


def model_group_of_model(model, equivalent_models=EQUIVALENT_MODELS):
    for groups in equivalent_models:
        if model in groups:
            return random.sample(groups, len(groups))
    return [model]


def wrap_with_equivalent_models(
    create_completion,
    equivalent_models=EQUIVALENT_MODELS,
    force_async: bool = False,
):
    def sync_wrapped_create_completion(*args, model, **kwargs):
        exception = None
        for model in model_group_of_model(model, equivalent_models=equivalent_models):
            try:
                return create_completion(*args, model=model, **kwargs)
            except Exception as e:
                exception = e
                continue
        else:
            raise exception

    async def async_wrapped_create_completion(*args, model, **kwargs):
        exception = None
        for model in model_group_of_model(model, equivalent_models=equivalent_models):
            try:
                return await create_completion(*args, model=model, **kwargs)
            except Exception as e:
                exception = e
                continue
        else:
            raise exception

    if inspect.iscoroutinefunction(create_completion) or force_async is True:
        return async_wrapped_create_completion
    else:
        return sync_wrapped_create_completion
