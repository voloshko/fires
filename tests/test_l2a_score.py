from scripts.score_l2a_finetuning import l2a_acceptance


def test_l2a_acceptance_requires_two_controls_and_both_regions():
    records = [dict(group_id=f"{d}:{i}", dataset=d) for d in ("ems", "floga") for i in range(3)]
    strong, weak = [[90, 5, 5]] * 6, [[60, 20, 20]] * 6
    assert l2a_acceptance(records, strong, weak, weak)["accepted"]
    assert not l2a_acceptance(records, strong, weak, strong)["accepted"]
    assert not l2a_acceptance(records, weak[:3] + strong[3:], strong, weak)["accepted"]
