"""Explicit cross-embodiment initialization, separate from exact-run resume."""
from pathlib import Path
import logging
import torch

logger = logging.getLogger(__name__)
RESET_PREFIXES = (
    "mixtures.action.action_encoder.",
    "mixtures.action.head.linear.",
)


def transfer_robot_checkpoint(model, payload):
    """Reuse the entire compatible trunk; reset embodiment-dependent projections.

    Shapes alone are insufficient: a 16D EEF checkpoint has different semantics.
    Reject all missing, extra or mismatched trunk keys, even with strict=False.
    The separately stored proprio encoder is deliberately left newly initialized.
    """
    if payload.get("checkpoint_format", "full") != "full":
        raise ValueError("SIMPLE initialization requires a full, non-LoRA checkpoint")
    source = payload["mot"]
    target = model.mot.state_dict()
    reset = lambda key: key.startswith(RESET_PREFIXES)
    extra = set(source) - set(target)
    missing = {key for key in target if key not in source and not reset(key)}
    mismatch = {key: (tuple(source[key].shape), tuple(target[key].shape))
                for key in target if key in source and not reset(key)
                and source[key].shape != target[key].shape}
    if extra or missing or mismatch:
        raise ValueError(f"Incompatible trunk: extra={sorted(extra)}, missing={sorted(missing)}, shapes={mismatch}")
    transferable = {key: source[key] for key in target if not reset(key)}
    result = model.mot.load_state_dict(transferable, strict=False)
    assert not result.unexpected_keys
    assert all(reset(key) for key in result.missing_keys)
    report = {
        "loaded_tensors": len(transferable),
        "loaded_parameters": sum(value.numel() for value in transferable.values()),
        "reset_mot_keys": sorted(result.missing_keys),
        "reset_proprio_keys": sorted(model.proprio_encoder.state_dict()),
    }
    logger.warning("SIMPLE cross-embodiment initialization: %s", report)
    return report


def create_simple_imagewam(pretrained_checkpoint, **kwargs):
    checkpoint = Path(pretrained_checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing ImageWAM robot pretrain checkpoint: {checkpoint}")
    from imagewam.runtime import create_imagewam_flux2_klein
    model = create_imagewam_flux2_klein(**kwargs)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.simple_transfer_report = transfer_robot_checkpoint(model, payload)
    return model
