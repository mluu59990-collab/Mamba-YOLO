"""Forward, detector loss/backward, and checkpoint reload. No dataset/download needed."""
import argparse
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--variant', choices=['all', 'baseline', 'mamba'], default='all')
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    root = Path(__file__).resolve().parents[1]
    for variant in (['baseline', 'mamba'] if a.variant == 'all' else [a.variant]):
        yolo = YOLO(root / f'configs/yolov12s-{variant}.yaml', task='detect')
        model = yolo.model.to(a.device).train()
        model.args = get_cfg(model.args)
        # P5 must be >=3x3: at 2x2 all zero-offset samples lie on clamped
        # borders, whose offset gradients legitimately vanish.
        x = torch.rand(2, 3, 96, 96, device=a.device)
        batch = dict(img=x, batch_idx=torch.tensor([0, 1], device=a.device),
                     cls=torch.tensor([[0.], [1.]], device=a.device),
                     bboxes=torch.tensor([[.5,.5,.3,.3], [.4,.4,.2,.2]], device=a.device))
        loss, _ = model(batch)
        assert torch.isfinite(loss).all(), f'{variant}: non-finite loss'
        loss.sum().backward()
        grads = [v.grad for v in model.parameters() if v.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if variant != 'baseline':
            ssm = [v.grad for n,v in model.named_parameters() if 'A_logs' in n]
            assert ssm and all(g is not None and g.abs().sum() > 0 for g in ssm)
        if variant == 'akcmamba':
            offsets = [v.grad for n,v in model.named_parameters() if 'offset.weight' in n]
            assert offsets and all(g is not None and g.abs().sum() > 0 for g in offsets)
        model.eval()
        with torch.no_grad():
            expected = model(x)[0]
            assert expected.shape[1] == 84
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / 'model.pt'
                model.args = vars(model.args)
                yolo.save(checkpoint)
                restored = YOLO(checkpoint).model.to(a.device).float().eval()
                actual = restored(x)[0]
                # YOLO.save stores fp16 weights, so checkpoint output is quantized.
                torch.testing.assert_close(actual, expected, rtol=.03, atol=.1)
        print(f'PASS {variant}: loss={loss.sum().item():.4f}, output={tuple(expected.shape)}, backward + reload')


if __name__ == '__main__':
    main()
