import pytest
import torch
from ultralytics.nn.modules.paper_blocks import AKConv, AKC3Adapter, AKC4Adapter
from ultralytics.nn.modules.common_utils_mbyolo import CrossScan, CrossMerge, SelectiveScanCore
from ultralytics.nn.modules.scan_reference import selective_scan_ref


def test_scan_merge_rectangular_and_gradient():
    x = torch.randn(2, 3, 3, 5, requires_grad=True)
    scans = CrossScan.apply(x)
    assert torch.equal(scans[:, 0], x.flatten(2))
    assert torch.equal(scans[:, 1], x.transpose(2, 3).flatten(2))
    assert torch.equal(scans[:, 2:], scans[:, :2].flip(-1))
    merged = CrossMerge.apply(scans.reshape(2, 4, 3, 3, 5)).reshape_as(x)
    torch.testing.assert_close(merged, 4 * x)
    merged.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 4))


def test_akconv_zero_offsets_and_border():
    m = AKConv(1, points=5)
    x = torch.arange(12.).reshape(1, 1, 3, 4)
    sampled = m.sample(x).reshape(1, 1, 3, 5, 4)
    for point, (dy, dx) in enumerate(m.lattice.long()):
        yy = (torch.arange(3) + dy).clamp(max=2)
        xx = (torch.arange(4) + dx).clamp(max=3)
        torch.testing.assert_close(sampled[:, :, :, point], x[:, :, yy[:, None], xx[None, :]])
    m.sample(x).sum().backward()
    assert m.offset.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('cls', [AKC3Adapter, AKC4Adapter])
def test_akc_rectangular_backward(cls):
    torch.set_num_threads(2)
    m = cls(8)
    x = torch.randn(2, 8, 3, 5, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad.isfinite().all()
    assert m.ssm.A_logs.grad.abs().sum() > 0


def test_akc_half_checkpoint_inference_cpu():
    # Validator's half EMA must work with FP32 selective scan and LayerNorm.
    m = AKC3Adapter(8).half().eval()
    with torch.no_grad():
        y = m(torch.randn(2, 8, 3, 5).half())
    assert y.dtype == torch.float16 and torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_scan_matches_reference_forward_backward():
    # The same selective recurrence must be used on CPU and CUDA.
    torch.manual_seed(10)
    args = [torch.randn(2, 16, 17), torch.randn(2, 16, 17),
            -torch.rand(16, 4), torch.randn(2, 4, 4, 17),
            torch.randn(2, 4, 4, 17), torch.randn(16), torch.randn(16)]
    ref_args = [v.cuda().requires_grad_() for v in args]
    cuda_args = [v.cuda().requires_grad_() for v in args]
    expected = selective_scan_ref(*ref_args)
    actual = SelectiveScanCore.apply(*cuda_args, True, 1, 1, True)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    expected.sum().backward()
    actual.sum().backward()
    for a, b in zip(cuda_args, ref_args):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_amp_akc_backward():
    m = AKC3Adapter(16).cuda()
    x = torch.randn(2, 16, 8, 8, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.float16):
        y = m(x)
        loss = y.square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert m.ssm.A_logs.grad.abs().sum() > 0
    m.half().eval()
    with torch.no_grad():
        assert torch.isfinite(m(x.detach().half())).all()
