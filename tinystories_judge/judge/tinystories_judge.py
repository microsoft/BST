import hashlib
import json
import logging
import os
import random
import time
from typing import Dict, List
from openai import OpenAI
import regex as re


class GPTJudge:
    def __init__(
        self,
        seed: int = 42,
        cache_dir: str = None,
        cache_without_prompt: bool = False,
        prompt_ver: str = "v1",
        deployment_name: str = "gpt-4o-mini-2024-07-18",
    ):
        random.seed(seed)

        self.deployment_name = deployment_name
        self.prompt_ver = prompt_ver
        self.cache_without_prompt = cache_without_prompt
        self.cache_dir = (
            cache_dir if cache_dir else f"./judge_cache/{self.deployment_name}"
        )

        os.makedirs(self.cache_dir, exist_ok=True)

        self.api_key_name = "OPENAI_API_KEY"
        self.oai_client = OpenAI()

    def judge(self, story_model_a: Dict, story_model_b: Dict, repeat: int = 3):
        # hash story_model_a and story_model_b to get a unique cache key based on all values
        cache_key = self.hash_two_dicts(story_model_a, story_model_b)

        prompt_ver_str = "" if self.cache_without_prompt else f"_{self.prompt_ver}"
        current_cache_dir = f"{self.cache_dir}/{cache_key[:10]}" + prompt_ver_str
        os.makedirs(current_cache_dir, exist_ok=True)

        print(f"current_cache_dir: {current_cache_dir}, prompt_ver: {self.prompt_ver}")

        result_dict = {story_model_a["model"]: 0, story_model_b["model"]: 0, "tie": 0}
        for i in range(repeat):
            cache_file = f"{current_cache_dir}/{i}.json"

            if os.path.exists(cache_file):
                with open(cache_file, "r") as file:
                    result = json.loads(file.read())
            else:
                if random.choice([True, False]):
                    logging.info("Swapping story models order")
                    story_model_a, story_model_b = story_model_b, story_model_a

                chat_log = GPTJudge.generate_judge_prompt(
                    story1=story_model_a["story"],
                    story2=story_model_b["story"],
                    prompt_ver=self.prompt_ver,
                )
                result = {
                    "prompt": chat_log,
                    "story_1": story_model_a["model"],
                    "story_2": story_model_b["model"],
                    "iteration": i,
                    "model_judge": self.deployment_name,
                }

                oai_args = {
                    "model": self.deployment_name,
                    "messages": chat_log,
                    "temperature": 0,
                    # "max_tokens": max_tokens,
                    # "top_p": 0.95,
                    # "frequency_penalty": 0,
                    # "presence_penalty": 0,
                    # "stop": None,
                }
                response = self.retry_openai_func(
                    self.oai_client.chat.completions.create,
                    expected_finish_reason="stop",
                    **oai_args,
                )
                result["result"] = response.choices[0].message.content.strip()

                with open(cache_file, "w") as f:
                    json.dump(result, f, indent=4)

            # Update result_dict
            result_str = (
                result["result"].split("Winner: ")[1].split("\n", 1)[0].split("*")[0]
            )

            if "None" in result_str:
                result_dict["tie"] += 1
            else:
                if "Story 1" in result_str:
                    result_dict[result["story_1"]] += 1
                elif "Story 2" in result_str:
                    result_dict[result["story_2"]] += 1
                else:
                    raise

        return result_dict

    @staticmethod
    def hash_two_dicts(dict1, dict2):
        dict1_items = [f"{key}={dict1[key]}" for key in dict1]
        dict2_items = [f"{key}={dict2[key]}" for key in dict2]

        all_items = sorted(dict1_items + dict2_items)

        combined_string = ",".join(all_items)

        hashed_value = hashlib.sha256(combined_string.encode("utf-8")).hexdigest()

        return hashed_value

    @staticmethod
    def generate_judge_prompt(story1: str, story2: str, prompt_ver: str) -> List:

        if prompt_ver == "v1":
            instructions_content = f"""You are evaluating two story generation models. First, analyze the outputs and note their properties. Then, make an argument for why one is better than another, or say that both are roughly equal. Finally, using all the analysis above, decide which is the best story writing the string "Winner: name of story" or if both story are comparable write "Winner: None" ."""

        elif prompt_ver == "v2":
            instructions_content = f"""You are evaluating two story generation models that complete a story given the same beginning and end. You are particularly interested in deciding which story has the least abrupt transitions. First, analyze the outputs and note their properties. Then, make an argument for why one story is better than another, or say that both are roughly equal. Finally, using all the analysis above, decide which is the best story writing the string "Winner: name of story" or, if both stories are comparable, write "Winner: None" ."""

        else:
            raise NotImplementedError("Unknown prompt version")

        stories_content = f"\nStory 1: {story1}\n\nStory 2: {story2}"

        messages = [
            {
                "role": "user",
                "content": instructions_content + stories_content,
            },
        ]

        return messages

    @staticmethod
    def retry_openai_func(
        func, expected_finish_reason="stop", max_attempt=10000, **kwargs
    ):
        attempts = 0
        sleep_time = 10
        while True:
            try:
                response = func(**kwargs)
                if response.choices[0].finish_reason not in ("stop", "length"):
                    logging.warning(
                        f"retry_openai_func: successful openai response but finish_reason is not 'stop' or 'length'. actual= {response.choices[0].finish_reason}. retrying.."
                    )
                    time.sleep(10)
                else:
                    if (
                        response.choices[0].finish_reason == "length"
                        and expected_finish_reason == "stop"
                    ):
                        logging.warning(
                            "retry_openai_func: reached max tokens - consider increasing max_tokens"
                        )
                    break
            except Exception as e:
                logging.error(
                    f"retry_openai_func: call {func.__module__}.{func.__name__}: attempt {attempts} failed {e}"
                )
                r = re.search(r"Please retry after\s(\d+)", str(e))
                sleep_time = int(r.group(1)) if r else 10
                attempts += 1
                if attempts == max_attempt:
                    logging.critical(
                        f"retry_openai_func: reached max attempt ({max_attempt}). failing"
                    )
                    return ""
                else:
                    time.sleep(sleep_time)
        return response
