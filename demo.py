# %%
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
import logging
import os
import threading
from typing import Optional
from openai import OpenAI, OpenAIError, AsyncOpenAI
import openai
import json
from tqdm import tqdm
from memocache import Memoize
from equivalent_model_wrapper import wrap_with_equivalent_models
from parallel_processor import process_api_requests


# %%
client = OpenAI()
aclient = AsyncOpenAI()


def is_api_exception(e):
    return isinstance(e, OpenAIError)


def is_rate_limit_exception(e):
    return isinstance(e, openai.RateLimitError)


# %%

# Equivalent model wrapper and memoization

client.chat.completions.create = Memoize(
    wrap_with_equivalent_models(client.chat.completions.create),
    name="chat.completions.create",
)
aclient.chat.completions.create = Memoize(
    wrap_with_equivalent_models(aclient.chat.completions.create, force_async=True),
    name="chat.completions.create",
)


# %%

# Workflow for this demo is to create functions of serial requests that depend on each other, and then run them in parallel


# %%

occasions = [
    "Birthday",
    "Anniversary",
    "Wedding",
    "Graduation",
    "New Baby",
    "New Home",
    "New Job",
    "Retirement",
    "Sympathy",
    "Thank You",
    "Thinking of You",
    "Get Well Soon",
    "Good Luck",
    "Congratulations",
    "I Love You",
    "I Miss You",
    "I Am Sorry",
    "Just Because",
    "Other",
]

temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]

output_file = "demo_outputs.jsonl"

# %%
file_path_cache = {}
file_locks = defaultdict(lambda: threading.Lock())


def append_jsonl(file_path, data: list):
    with file_locks[file_path]:
        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                pass
    if file_path not in file_path_cache:
        with file_locks[file_path]:
            with open(file_path, "r") as file:
                file_path_cache[file_path] = (set(file.readlines()), threading.Lock())
    new_data = []
    json_data = [json.dumps(record) + "\n" for record in data]
    cache, lock = file_path_cache[file_path]
    with lock:
        for line in json_data:
            if line not in cache and line not in new_data:
                new_data.append(line)
            else:
                logging.warning(f"Ignoring duplicate: {line[:40]}...")
        cache.update(new_data)
    with file_locks[file_path]:
        with open(file_path, "a") as file:
            file.writelines(new_data)


# %%


def make_requests(occasion, temperature, output_file=output_file):
    async def requests():
        logging.debug(f"Starting request({occasion}, {temperature})")
        request_arguments = dict(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a greeting card text writer.",
                },
                {
                    "role": "user",
                    "content": "Please help me write a case for the following occasion: "
                    + occasion,
                },
            ],
            temperature=temperature,
        )
        logging.debug(f"Calling chat.completions.create(**{request_arguments})")
        response = (
            (await aclient.chat.completions.create(**request_arguments))
            .choices[0]
            .message.content
        )

        result = dict(
            occasion=occasion,
            response=response,
            request_arguments=request_arguments,
        )
        logging.debug(
            f"Finished generating_requests({occasion}, {temperature}) -> {response}"
        )
        await asyncio.to_thread(append_jsonl, output_file, [result])
        logging.info(f"Wrote response to {occasion}, {temperature}) to disk!")
        return result

    requests.__str__ = (
        lambda: f"greeting card test for {occasion} with temperature {temperature}"
    )
    return requests


list_of_requests = [
    make_requests(occasion, temperature)
    for occasion in occasions
    for temperature in temperatures
]
print(len(list_of_requests))

list_of_requests = list_of_requests[:]


# %%
async def process_requests():
    # For speed, don't keep loading the file from disk
    with aclient.chat.completions.create.sync_cache(inplace=True):
        return await process_api_requests(
            requests=tqdm(list_of_requests, desc="Queuing", position=0).__iter__(),
            max_requests_per_minute=10000,
            max_attempts=10,
            is_rate_limit_exception=is_rate_limit_exception,
            is_api_exception=is_api_exception,
            tqdm_requests=tqdm(list_of_requests, desc="Running", position=1).__iter__(),
        )


# %%
async def main():
    # initialize logging
    logging.basicConfig(level=logging.ERROR)
    logging.debug(f"Logging initialized")

    await process_requests()

    logging.basicConfig(level=logging.ERROR)


# %%
if __name__ == "__main__":
    asyncio.run(main())
