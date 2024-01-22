# imports
from typing import Any, Awaitable, Callable, Iterable, Iterator, Optional
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata


async def process_api_requests(
    requests: Iterator[Callable[[], Awaitable[Any]]],
    max_requests_per_minute: float,
    max_attempts: int,
    is_rate_limit_exception: Callable[[Exception], bool],
    is_api_exception: Callable[[Exception], bool],
    tqdm_requests: Optional[Iterator] = None,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    last_update_time = time.time()

    iterator_not_finished = True
    logging.debug(f"Initialization complete.")

    logging.debug(f"Entering main loop")
    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(
                    f"Retrying request {next_request.task_id}: {next_request}"
                )
            elif iterator_not_finished:
                try:
                    # get new request
                    request_func = next(requests)
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        request_func=request_func,
                        attempts_left=max_attempts,
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(
                        f"Reading request {next_request.task_id}: {next_request}"
                    )
                except StopIteration:
                    # if iterator runs out, set flag to stop reading it
                    logging.debug("Iterator exhausted")
                    iterator_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity
            + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )

        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            if available_request_capacity >= 1:
                # update counters
                available_request_capacity -= 1
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        retry_queue=queue_of_requests_to_retry,
                        status_tracker=status_tracker,
                        is_rate_limit_exception=is_rate_limit_exception,
                        is_api_exception=is_api_exception,
                        tqdm_requests=tqdm_requests,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (
            time.time() - status_tracker.time_of_last_rate_limit_error
        )
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (
                seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
            )
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(
                f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
            )

        # after finishing, log final status
    logging.info(f"""Parallel processing complete. I hope you cached your results!""")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to stderr."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs. Contains a method to make an API call."""

    task_id: int
    request_func: Callable[[], Awaitable[Any]]
    attempts_left: int
    result: list = field(default_factory=list)

    async def call_api(
        self,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        is_rate_limit_exception: Callable[[Exception], bool],
        is_api_exception: Callable[[Exception], bool],
        tqdm_requests: Optional[Iterable] = None,
    ):
        """Calls the API"""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            response = await self.request_func()

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            if is_rate_limit_exception(e):
                logging.warning(
                    f"Request {self.task_id} failed with rate limit exception {e}"
                )
                status_tracker.time_of_last_rate_limit_error = time.time()
                status_tracker.num_rate_limit_errors += 1
                error = e
            elif is_api_exception(e):
                logging.warning(f"Request {self.task_id} failed with API Exception {e}")
                status_tracker.num_api_errors += 1
                error = e
            else:
                logging.warning(f"Request {self.task_id} failed with Exception {e}")
                status_tracker.num_other_errors += 1
                error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_func} failed after all attempts. Errors: {self.result}"
                )
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
                next(tqdm_requests, None)
        else:
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.info(f"Request {self.task_id} completed successfully")
            logging.debug(f"Request {self.task_id} returned {response}")
            next(tqdm_requests, None)


# functions


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1
