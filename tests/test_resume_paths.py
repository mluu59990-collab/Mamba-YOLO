from types import SimpleNamespace
from unittest.mock import patch
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.cfg import get_cfg


def test_resume_honors_explicit_dataset_and_workers(tmp_path):
    old = tmp_path / 'old.yaml'
    new = tmp_path / 'new.yaml'
    old.touch()
    new.touch()
    checkpoint = tmp_path / 'last.pt'
    checkpoint.touch()
    trainer = object.__new__(BaseTrainer)
    trainer.args = get_cfg(overrides=dict(resume=str(checkpoint), data=str(new)))
    saved = vars(get_cfg(overrides=dict(data=str(old), epochs=300)))
    with patch('ultralytics.engine.trainer.attempt_load_weights', return_value=SimpleNamespace(args=saved)):
        trainer.check_resume(dict(data=str(new), workers=0))
    assert trainer.args.data == str(new)
    assert trainer.args.workers == 0
    assert trainer.args.epochs == 300
    assert trainer.resume
