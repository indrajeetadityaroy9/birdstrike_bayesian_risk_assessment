"""EMA decay computation from dataset statistics."""


def compute_ema_decay(n_samples: int, batch_size: int) -> float:
    """Data-derived EMA decay from dataset size and batch size.

    Decay = 1 - 1/T where T = effective epochs in memory window.
    For N=400k, batch=256: T=1000, decay=0.999.
    For N=1k, batch=32: T=31, decay=0.968.

    Args:
        n_samples: Total training samples
        batch_size: Current batch size

    Returns:
        EMA decay factor in (0, 1)

    """
    span = min(1000, max(10, n_samples // max(batch_size, 1)))
    return 1.0 - 1.0 / span
