"""Тесты загрузки сети (SPEC-19, SPEC-20)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch нужен только для сети")

from src.comp.unet import _modernise  # noqa: E402


def test_старые_имена_весов_переименовываются():
    """Сети, сохранённые до параметризации глубины, должны читаться: переобучать
    их ради одних имён — расточительство."""
    old = {"d1.0.weight": 1, "d4.1.bias": 2, "u3.weight": 3, "c1.0.weight": 4,
           "head.weight": 5, "head.bias": 6}
    new = _modernise(old)
    assert new["down.0.0.weight"] == 1
    assert new["down.3.1.bias"] == 2
    assert new["up.0.weight"] == 3
    assert new["conv.2.0.weight"] == 4
    assert new["head.weight"] == 5 and new["head.bias"] == 6
    assert not any(k.startswith(("d1.", "d4.", "u3.", "c1.")) for k in new)


def test_новые_имена_не_трогаются():
    new = {"down.0.0.weight": 1, "up.0.weight": 2, "head.bias": 3}
    assert _modernise(new) is new, "лишнего копирования быть не должно"


def test_число_ключей_сохраняется():
    old = {f"d{i}.0.weight": i for i in range(1, 5)}
    assert len(_modernise(old)) == len(old), "переименование не должно терять веса"


def test_сиамский_бандл_грузится_и_даёт_вероятности(tmp_path):
    """Бандл SPEC-32 (variant=siam) грузится через unet.load, probs даёт (H, W, 4)."""
    import numpy as np, torch
    from src.comp.hypothesis_models import make_model
    from src.comp import unet
    from src.comp.chips import BsChip
    net = make_model("siam", 4, 3)
    torch.save({"state": net.state_dict(), "variant": "siam", "width": 4, "depth": 3, "fusion_norm": True,
                "mean": np.zeros(29, np.float32), "std": np.ones(29, np.float32)}, tmp_path / "siam.pt")
    model = unet.load(tmp_path / "siam.pt")
    pre = np.full((10, 32, 32), 1000, np.uint16); pre[9] = 4
    chip = BsChip("x", pre, pre.copy(), np.zeros((3, 32, 32)), np.zeros((2, 32, 32)), None)
    p = unet.probs(model, chip, tta=False)
    assert p.shape == (32, 32, 4) and np.allclose(p.sum(2), 1, atol=1e-4)
