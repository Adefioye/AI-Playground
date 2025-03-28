import reasoning_gym
data = reasoning_gym.create_dataset('calendar_arithmetic', size=10, seed=42)
for i, x in enumerate(data):
    print(f"{i}: q={x['question']}, a={x['answer']}")
    print('metadata:', x['metadata'])
    # use the dataset's `score_answer` method for algorithmic verification
    assert data.score_answer(answer=x['answer'], entry=x) == 1.0