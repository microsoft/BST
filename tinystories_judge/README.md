# GPT4 judge for tinystories

## Prerequisities
The pipeline is setup to use Open AI endpoint instance but any other API endpoint compatible with the python `openai` package can be used. It requires:
```sh
pip install openai
```
Open AI azure endpoint instance or any API endpoint compatible with the python `openai` package.

# How to use Judge
To test the GPT Judge, one can run:
```sh
python tinystories_judge/judge_test.py
```

Points of interest:
- each instance of judge sets the seed during init, and the RNG moves everytime it decides to swap order.
- the judge aggresively caches the queries, so if you want to increase repetitions it will use previously generated results.
- the format it saves the generated response from gpt4 is in a json file