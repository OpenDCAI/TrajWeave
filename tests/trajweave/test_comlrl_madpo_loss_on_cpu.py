from __future__ import annotations

import pytest
import torch

from trajweave.backends.verl.extensions.comlrl.preference import (
    detached_other_agent_deltas,
    madpo_joint_pair_loss,
    madpo_microbatch_loss,
    sequence_logprob_sums,
    snapshot_pair_deltas,
)
from trajweave.backends.verl.extensions.comlrl.preference import (
    madpo_actor_loss as verl_madpo_actor_loss,
)


def test_sequence_logprob_sum_uses_only_response_tokens_without_length_normalization():
    log_probs = torch.tensor([[-0.2, -0.3, 99.0], [-0.5, 99.0, 99.0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    assert torch.allclose(sequence_logprob_sums(log_probs, mask), torch.tensor([-0.5, -0.5]))


def test_madpo_joint_delta_loss_and_gradient_isolation_match_v141():
    current = torch.tensor([0.7], requires_grad=True)
    other = torch.tensor([-0.2], requires_grad=True)

    loss = madpo_joint_pair_loss(current, other, beta=0.1)

    assert loss.item() == pytest.approx(-torch.nn.functional.logsigmoid(torch.tensor(0.05)).item())
    loss.backward()
    assert current.grad is not None and current.grad.abs().item() > 0
    assert other.grad is None


def test_snapshot_pair_deltas_validate_complete_pairs_and_detach_other_agents():
    logps = torch.tensor([2.0, 1.0, 4.0, 1.5], requires_grad=True)
    deltas = snapshot_pair_deltas(
        logps,
        pair_ids=["pair", "pair", "pair", "pair"],
        sides=["chosen", "rejected", "chosen", "rejected"],
        worker_groups=["actor-a", "actor-a", "actor-b", "actor-b"],
    )

    assert deltas[("pair", "actor-a")].item() == pytest.approx(1.0)
    assert deltas[("pair", "actor-b")].item() == pytest.approx(2.5)
    others = detached_other_agent_deltas(deltas)
    assert others[("pair", "actor-a")].item() == pytest.approx(2.5)
    assert others[("pair", "actor-b")].item() == pytest.approx(1.0)
    loss = madpo_joint_pair_loss(
        deltas[("pair", "actor-a")].reshape(1),
        others[("pair", "actor-a")].reshape(1),
    )
    loss.backward()
    assert logps.grad is not None
    assert logps.grad[0].abs().item() > 0 and logps.grad[1].abs().item() > 0
    assert logps.grad[2:].abs().sum().item() == 0.0

    with pytest.raises(ValueError, match="one chosen and one rejected"):
        snapshot_pair_deltas(
            torch.tensor([1.0]),
            pair_ids=["pair"],
            sides=["chosen"],
            worker_groups=["actor-a"],
        )


def test_production_microbatch_non_finite_token_uses_upstream_constant_fallback():
    log_probs = torch.tensor([[float("nan")], [0.0]], requires_grad=True)

    loss, pair_count = madpo_microbatch_loss(
        log_probs,
        torch.ones(2, 1, dtype=torch.long),
        torch.tensor([0, 0]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 1.0]),
        beta=0.1,
        global_pair_count=1,
    )

    assert loss.item() == pytest.approx(0.1)
    assert pair_count == 0
    loss.backward()
    assert log_probs.grad is None


def test_real_verl_callback_limits_constant_fallback_to_non_finite_token_logprobs(monkeypatch):
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils import padding

    class FakeData:
        def __init__(self, fields, non_tensor):
            self.fields = fields
            self.non_tensor = non_tensor

        def select(self, *keys):
            return FakeData({key: self.fields[key] for key in keys}, self.non_tensor)

        def to_padded_tensor(self):
            return self.fields

    fields = {
        "response_mask": torch.ones(2, 1, dtype=torch.long),
        "madpo_pair_index": torch.tensor([0, 0]),
        "madpo_preference_side": torch.tensor([1.0, -1.0]),
        "madpo_other_agent_delta": torch.tensor([0.0, 0.0]),
        "preference_loss_mask": torch.tensor([1.0, 1.0]),
    }
    data = FakeData(fields, {"madpo_global_pair_count": 1, "dp_size": 1})
    monkeypatch.setattr(padding, "no_padding_2_padding", lambda values, _data: values)
    monkeypatch.setattr(
        tu,
        "get_non_tensor_data",
        lambda actual, key, default=None: actual.non_tensor.get(key, default),
    )

    loss, metrics = verl_madpo_actor_loss(
        {"beta": 0.1},
        {"log_probs": torch.tensor([[float("nan")], [0.0]], requires_grad=True)},
        data,
    )

    assert loss.item() == pytest.approx(0.1)
    assert metrics["actor/madpo_pair_count"] == 0.0

    fields["madpo_other_agent_delta"] = torch.tensor([float("nan"), float("nan")])
    with pytest.raises(FloatingPointError, match="other-agent deltas"):
        verl_madpo_actor_loss(
            {"beta": 0.1},
            {"log_probs": torch.zeros(2, 1, requires_grad=True)},
            data,
        )


def test_non_finite_joint_metadata_fails_fast_instead_of_using_token_fallback():
    current = torch.tensor([float("inf")], requires_grad=True)
    other = torch.tensor([float("-inf")])

    with pytest.raises(FloatingPointError, match="must be finite"):
        madpo_joint_pair_loss(current, other)


def test_non_finite_other_agent_delta_fails_fast_in_production_microbatch():
    with pytest.raises(FloatingPointError, match="other-agent deltas"):
        madpo_microbatch_loss(
            torch.tensor([[0.0], [0.0]], requires_grad=True),
            torch.ones(2, 1, dtype=torch.long),
            torch.tensor([0, 0]),
            torch.tensor([1.0, -1.0]),
            torch.tensor([float("nan"), float("nan")]),
            torch.tensor([1.0, 1.0]),
            beta=0.1,
            global_pair_count=1,
        )


@pytest.mark.parametrize('flat_response', [False, True])
@pytest.mark.parametrize('non_tensor_metadata', [False, True])
def test_verl_madpo_callback_preserves_pair_gradients_across_output_layouts(flat_response, non_tensor_metadata):
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    def nested(rows):
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    data = TensorDict({
        'prompts': nested([torch.tensor([10, 11]), torch.tensor([10])]),
        'responses': nested([torch.tensor([12, 13]), torch.tensor([14])]),
        'response_mask': nested([torch.ones(2, dtype=torch.long), torch.ones(1, dtype=torch.long)]),
        'madpo_pair_index': torch.tensor([0, 0]),
        'madpo_preference_side': torch.tensor([1., -1.]),
        'madpo_other_agent_delta': torch.zeros(2),
        'preference_loss_mask': torch.ones(2),
    }, batch_size=[2])
    if non_tensor_metadata:
        from tensordict.utils import LinkedList

        for key in ('madpo_pair_index', 'madpo_preference_side',
                    'madpo_other_agent_delta', 'preference_loss_mask'):
            values = data.pop(key).tolist()
            tu.assign_non_tensor_stack(data, key, values)
            assert isinstance(data[key], LinkedList)
    tu.assign_non_tensor(data, madpo_global_pair_count=1, dp_size=1)
    values = torch.tensor([-.2, -.3, -.5], requires_grad=True)
    if flat_response:
        output = values.unsqueeze(0)
    else:
        full = torch.cat([values.new_tensor([-9.]), values[:2], values.new_tensor([-9.]),
                          values[2:], values.new_tensor([-9.])])
        output = torch.nested.nested_tensor_from_jagged(full, torch.tensor([0, 4, 6]))
    loss, metrics = verl_madpo_actor_loss({'beta': .1}, {'log_probs': output}, data)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.)).item())
    assert metrics['actor/madpo_pair_count'] == 1
    loss.backward()
    assert values.grad.tolist() == pytest.approx([-.05, -.05, .05])
