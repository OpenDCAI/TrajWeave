from trajweave.backends.verl.extensions.registry import _extension_names


def test_verl_extensions_auto_select_drmas_from_recipe():
    assert _extension_names({"trajweave": {"recipe": "doctor_mas_math"}}) == ("drmas_agent_wise_grpo",)


def test_verl_extensions_auto_select_maporl_from_recipe_and_credit():
    assert _extension_names({"trajweave": {"recipe": "maporl_debate_math"}}) == ("trajweave_maporl_full_ppo",)
    assert _extension_names({"trajweave": {"credit_allocator": "maporl_score_bonus"}}) == (
        "trajweave_maporl_single_model",
    )
    assert _extension_names({"trajweave": {"credit_allocator": "maporl_ppo_score_rule"}}) == (
        "trajweave_maporl_full_ppo",
    )


def test_verl_extensions_auto_select_gigpo_from_recipe_and_credit():
    assert _extension_names({"trajweave": {"recipe": "gigpo_solver_verifier_math"}}) == (
        "trajweave_gigpo_hierarchical_grpo",
    )
    assert _extension_names({"trajweave": {"credit_allocator": "gigpo_hierarchical_grpo"}}) == (
        "trajweave_gigpo_hierarchical_grpo",
    )


def test_verl_extensions_auto_select_marti_mars2_tree_grpo():
    assert _extension_names({"trajweave": {"recipe": "marti_mars2_single_mcts"}}) == (
        "trajweave_marti_mars2_tree_grpo",
    )
    assert _extension_names({"trajweave": {"credit_allocator": "marti_mars2_fidelity_group_grpo"}}) == (
        "trajweave_marti_mars2_tree_grpo",
    )
