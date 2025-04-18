import argparse
import json
from judge.tinystories_judge import GPTJudge


def judge_files(
    fp1,
    fp2,
    n_stories=-1,
    num_judges=3,
    prompt_ver="v1",
    cache_without_prompt=False,
    model1_name="model1",
    model2_name="model2",
    deployment_name="gpt-4o-mini-2024-07-18",
):

    judge = GPTJudge(
        cache_without_prompt=cache_without_prompt,
        prompt_ver=prompt_ver,
        deployment_name=deployment_name,
    )

    if num_judges == 3:
        possible_keys = [
            "3_0_0",
            "2_1_0",
            "2_0_1",
            "1_2_0",
            "0_3_0",
            "0_2_1",
            "1_0_2",
            "0_1_2",
            "0_0_3",
            "1_1_1",
        ]
        result_dict = {x: 0 for x in possible_keys}
    else:
        possible_keys = []
        result_dict = {}

    with open(fp1, "r") as file1, open(fp2, "r") as file2:
        for i, (line1, line2) in enumerate(zip(file1, file2)):

            print(f"\nStory {i}")
            try:
                dict1 = json.loads(line1.strip())
                dict2 = json.loads(line2.strip())

                assert (
                    dict1["original"] == dict2["original"]
                ), "Original story mismatch between files"

                prompt = dict1["prompt"]
                goal = dict1["goal"]
                infill1 = dict1["generated"]
                infill2 = dict2["generated"]

                story1 = {
                    "story": prompt + infill1 + goal,
                    "model": model1_name,
                }

                story2 = {
                    "story": prompt + infill2 + goal,
                    "model": model2_name,
                }

            except Exception as e:
                print(f"Error in line {i}: {e}")
                continue

            d = judge.judge(story1, story2, repeat=num_judges)

            print(
                f"------- PREFIX (len: {len(prompt)})--------\n",
                prompt,
                f'\n\n------- INFILL {story1["model"]} (len: {len(infill1)}) ------\n',
                infill1,
                f'\n\n------- INFILL {story2["model"]} (len: {len(infill2)}) ------\n',
                infill2,
                f"\n\n------- SUFFIX (len: {len(goal)}) --------\n",
                goal,
                f"\n---------------------\n\nVotes: {d}\n",
            )

            # encoding voting into a string and updating histogram
            result_str = "_".join(
                map(str, [d[story1["model"]], d["tie"], d[story2["model"]]])
            )
            if result_str not in result_dict:
                result_dict[result_str] = 0
                possible_keys.append(result_str)
            result_dict[result_str] += 1

            print("Votes histogram:")
            sum_tot = 0
            for x in possible_keys:
                sum_tot += result_dict[x]
                print(x, result_dict[x])
            print("Total:", sum_tot)

            if i + 1 == n_stories:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load the BST vs GPT files to evaluate"
    )
    parser.add_argument(
        "--file1_path",
        type=str,
        required=True,
        help="Path to inference txt file of model 1",
    )
    parser.add_argument(
        "--file2_path",
        type=str,
        required=True,
        help="Path to inference txt file of model 2",
    )
    parser.add_argument(
        "-n",
        "--n_stories",
        type=int,
        default=-1,
        required=False,
        help="Number of stories to judge",
    )
    parser.add_argument(
        "--num_judges",
        type=int,
        default=3,
        required=False,
        help="Number of independent calls to GPT",
    )
    parser.add_argument(
        "--prompt_ver",
        type=str,
        default="v1",
        required=False,
        help="Version of the judge's prompt",
    )
    parser.add_argument(
        "--cache_without_prompt",
        action="store_true",
        help="Calculate the cache dir using only the stories and not the prompt",
    )
    parser.add_argument(
        "--model1_name",
        type=str,
        default="model1",
        required=False,
        help="Name of model 1",
    )
    parser.add_argument(
        "--model2_name",
        type=str,
        default="model2",
        required=False,
        help="Name of model 2",
    )
    parser.add_argument(
        "--deployment_name",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        required=False,
        help="Name of OPENAI API GPT model to use for judging",
    )

    args, conf_cli = parser.parse_known_args()

    judge_files(
        args.file1_path,
        args.file2_path,
        args.n_stories,
        args.num_judges,
        args.prompt_ver,
        args.cache_without_prompt,
        args.model1_name,
        args.model2_name,
        args.deployment_name,
    )
