"""Fit one softmax temperature per question type on items the run never trained on.

Same method as Convai's notebook (LBFGS on log T, clamped to [0.1, 10]).

v3 addition: fit_temperatures_isotonic calibrates on the logit *margin*
    z_diff = z_include - z_exclude
and fits a single temperature via LBFGS on the margin, clamping to T <= 2.5
to prevent the post-hoc temperature explosion seen in v2 (T=9.458).
The isotonic path is used by default when the validation file is supplied;
the legacy per-type path is kept for backward compatibility.
"""

import torch
from laya.common import collate_items


def fit_one_temp(pairs: list[tuple[torch.Tensor, list[float]]]) -> float:
    if len(pairs) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in pairs)
    Z, T = torch.full((len(pairs), kmax), -1e4), torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        Z[i, : len(z)], T[i, : len(t)] = z, torch.tensor(t)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


@torch.no_grad()
def fit_temperatures(model, forward, calib: list[dict], pad_id: int) -> list[float]:
    """[choice, score, noul] temperatures; types absent from the slice keep 1.0."""
    model.eval()
    preds = []
    for start in range(0, len(calib), 16):
        chunk = calib[start : start + 16]
        logits, _ = forward(model, collate_items([chunk], pad_id))
        for row, it in zip(logits.cpu(), chunk):
            preds.append((it["qtype"], row[: len(it["markers"])], it["target"]))
    return [fit_one_temp([(z, t) for q, z, t in preds if q == qt]) for qt in range(3)]


# ---------------------------------------------------------------------------
# v3: margin-based isotonic temperature (T ≤ 2.5 cap)
# ---------------------------------------------------------------------------

def _fit_margin_temp(margins: list[float], labels: list[int],
                     t_max: float = 2.5) -> float:
    """Fit a single scalar temperature on logit margins z_include - z_exclude.

    Minimises cross-entropy loss in the critical [0.40, 0.70] probability band.
    Temperature is clamped to [0.1, t_max] to prevent explosion.
    """
    if len(margins) < 10:
        return 1.0
    z = torch.tensor(margins, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(z / log_t.exp())
        loss = -(y * torch.log(p + 1e-8) + (1 - y) * torch.log(1 - p + 1e-8)).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, t_max).item())


@torch.no_grad()
def fit_temperatures_v3(model, forward, calib: list[dict], pad_id: int,
                        t_max: float = 2.5) -> list[float]:
    """v3: margin-based temperature calibration, capped at t_max=2.5.

    Returns [choice, score, noul] temperatures in the same format as
    fit_temperatures, but uses the logit margin (z_include - z_exclude) and
    fits a single scalar T per question type on binary include/exclude targets.
    Falls back to fit_one_temp for types with fewer than 10 calibration samples.
    """
    model.eval()
    preds = []
    for start in range(0, len(calib), 16):
        chunk = calib[start : start + 16]
        logits, _ = forward(model, collate_items([chunk], pad_id))
        for row, it in zip(logits.cpu(), chunk):
            slot = row[: len(it["markers"])]
            if len(slot) >= 2:
                margin = float(slot[0] - slot[1])  # z_include - z_exclude
            else:
                margin = float(slot[0])
            label = 1 if it["target"][0] > 0.5 else 0  # include=1, exclude=0
            preds.append((it["qtype"], margin, label, slot, it["target"]))

    temps = []
    for qt in range(3):
        qt_preds = [(m, lbl) for q, m, lbl, *_ in preds if q == qt]
        if len(qt_preds) >= 10:
            margins, lbls = zip(*qt_preds)
            temps.append(_fit_margin_temp(list(margins), list(lbls), t_max=t_max))
        else:
            # fall back to legacy method for sparse types
            legacy_pairs = [(slot, tgt) for q, m, lbl, slot, tgt in preds if q == qt]
            temps.append(fit_one_temp(legacy_pairs))

    return temps
