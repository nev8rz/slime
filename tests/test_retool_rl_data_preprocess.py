from examples.retool_post_train_all.rl_data_preprocess import BOXED_PROMPT_TEMPLATE, convert_aime_eval


def test_boxed_prompt_template_uses_dapo_boxed_answer_instruction():
    rendered = BOXED_PROMPT_TEMPLATE.format(problem="Find 1+1.")

    assert "Answer: \\boxed{$Answer}" in rendered
    assert "Find 1+1." in rendered


def test_convert_aime_eval_deduplicates_raw_problem(monkeypatch):
    rows = [
        {
            "prompt": [{"role": "user", "content": "wrapped"}],
            "reward_model": {"ground_truth": "2"},
            "extra_info": {"index": 1, "raw_problem": "Find 1+1."},
        },
        {
            "prompt": [{"role": "user", "content": "wrapped duplicate"}],
            "reward_model": {"ground_truth": "2"},
            "extra_info": {"index": 1, "raw_problem": "Find 1+1."},
        },
    ]

    monkeypatch.setattr("examples.retool_post_train_all.rl_data_preprocess.load_source", lambda *_: rows)

    converted = convert_aime_eval("unused", "train")

    assert converted == [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": BOXED_PROMPT_TEMPLATE.format(problem="Find 1+1."),
                }
            ],
            "label": "2",
        }
    ]
