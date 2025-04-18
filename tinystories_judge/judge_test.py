from judge.tinystories_judge import GPTJudge


# grabbed exmple from paper page 16
def test_judge():
    judge = GPTJudge()

    story_bst = {
        "story": 'Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. \
One day, Tom lost his red ball. He was very sad. Tom asked his friend, Sam, "Did you see my red ball?" Sam said, \
"No, but let’s look together." They looked and looked, but they could not find the red ball. Tom was very sad. Then, a \
big bird came and dropped the red ball. The bird had the red ball in its beak. Tom and Sam were so happy! They said, \
"Thank you, bird!" The bird flew away, and Tom played with his red ball all day.',
        "model": "bst",
    }

    story_baseline = {
        "story": 'Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. \
One day, Tom lost his red ball. He was very sad. Tom asked his friend, Sam, for help. "Sam, can you help me find my \
red ball?" Sam said, "Yes, I will help you." They looked and looked, but they could not find the red ball. Just when \
they were about to give up, a big bird flew down from the sky. The bird had the red ball in its beak! The bird dropped \
the ball into the pit. Tom and Sam were so happy. They thanked the bird and played with the red ball',
        "model": "baseline",
    }

    judge.judge(story_bst, story_baseline, repeat=3)


test_judge()
print("Test completed successfully.")
