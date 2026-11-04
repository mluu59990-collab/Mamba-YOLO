"""Differentiable S6 reference for CPU construction/tests; CUDA uses the vendored kernel."""
import torch
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    dtype = u.dtype
    u, delta, A, B, C = [v.float() for v in (u, delta, A, B, C)]
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, channels, length = u.shape
    groups = B.shape[1]
    B = B.repeat_interleave(channels // groups, dim=1)
    C = C.repeat_interleave(channels // groups, dim=1)
    state = u.new_zeros(batch, channels, A.shape[1])
    outputs = []
    for t in range(length):
        dt = delta[:, :, t, None]
        state = torch.exp(dt * A) * state + dt * B[:, :, :, t] * u[:, :, t, None]
        outputs.append((state * C[:, :, :, t]).sum(-1))
    y = torch.stack(outputs, dim=-1)
    if D is not None:
        y = y + u * D.float()[None, :, None]
    return y.to(dtype)
