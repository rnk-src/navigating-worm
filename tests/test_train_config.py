import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from worm.env import WormGoalEnv
from worm.training import (
    AdaptiveLevel,
    TrainConfig,
    level_at,
    load_config,
    load_policy,
    train,
)

ROOT = Path(__file__).parent.parent
CONFIGS = ROOT / "configs"


def smoke(steps: int = 1024, every: int = 512, keep: int = 2) -> TrainConfig:
    cfg = load_config(CONFIGS / "terrain_v2_smoke.toml")
    return replace(
        cfg,
        run=replace(cfg.run, steps=steps, n_envs=1),
        checkpoint=replace(cfg.checkpoint, every_steps=every, keep=keep),
    )


def progress(out: Path) -> dict:
    return json.loads((out / "progress.json").read_text())


def test_load_config_parses_every_section_and_rejects_unknown_keys(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "terrain_v2.toml")
    assert (cfg.run.name, cfg.run.steps, cfg.run.n_envs, cfg.run.seed) == (
        "terrain_v2",
        20_000_000,
        16,
        0,
    )
    assert cfg.env.obs_version == 2 and cfg.env.world == "curriculum"
    assert (cfg.env.world_size, cfg.env.cell, cfg.env.randomizer_level) == (8.0, 0.05, 1.0)
    assert cfg.curriculum.mode == "timesteps" and cfg.curriculum.replay_lower == 0.3
    assert cfg.curriculum.boundaries == (0.0, 0.15, 0.35, 0.6)
    assert (cfg.ppo.n_steps, cfg.ppo.batch_size, cfg.ppo.net_arch) == (1024, 4096, (64, 64))
    assert (cfg.checkpoint.every_steps, cfg.checkpoint.keep) == (1_000_000, 3)
    assert (cfg.eval.every_steps, cfg.eval.maps, cfg.eval.map_size) == (2_000_000, 5, 10.0)

    text = (CONFIGS / "terrain_v2.toml").read_text()
    bad_key = tmp_path / "bad_key.toml"
    bad_key.write_text(text.replace("keep = 3", "keep = 3\nkeeep = 4"))
    with pytest.raises(ValueError, match="keeep"):
        load_config(bad_key)
    bad_section = tmp_path / "bad_section.toml"
    bad_section.write_text(text + "\n[cloud]\nvm = 1\n")
    with pytest.raises(ValueError, match="cloud"):
        load_config(bad_section)


def test_every_committed_config_loads() -> None:
    names = {load_config(path).run.name for path in CONFIGS.glob("*.toml")}
    assert names == {"m1_flat", "terrain_v2", "terrain_v2_smoke"}
    flat = load_config(CONFIGS / "m1_flat.toml")
    assert flat.env.world == "flat" and flat.env.obs_version == 1 and flat.run.steps == 3_000_000


def test_level_follows_the_timestep_boundaries() -> None:
    boundaries = (0.0, 0.15, 0.35, 0.6)
    assert [level_at(boundaries, f) for f in (0.0, 0.1, 0.15, 0.34, 0.5, 0.6, 1.0)] == [
        0,
        0,
        1,
        1,
        2,
        3,
        3,
    ]


def test_adaptive_level_advances_on_a_full_window_of_successes() -> None:
    adaptive = AdaptiveLevel(threshold=0.8, window=5)
    assert [adaptive.record(True) for _ in range(4)] == [False] * 4  # the window is not full
    assert adaptive.record(True) is True
    assert adaptive.record(True) is False  # the window starts again after an advance
    slow = AdaptiveLevel(threshold=0.8, window=5)
    assert not any(slow.record(i % 2 == 0) for i in range(20))


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train(smoke(steps=1024, every=512, keep=1), tmp_path_factory.mktemp("smoke"))


def test_smoke_config_trains_and_writes_tb_checkpoints_and_meta(smoke_run: Path) -> None:
    assert list((smoke_run / "tb").rglob("events.out.tfevents.*"))
    meta = json.loads((smoke_run / "meta.json").read_text())
    assert meta["obs_version"] == 2 and meta["obs_dim"] == 31
    assert meta["config"] == "terrain_v2_smoke" and meta["steps"] == 1024
    assert load_config(smoke_run / "config.toml") == smoke(steps=1024, every=512, keep=1)
    state = progress(smoke_run)
    assert state["steps"] == 1024 and state["target"] == 1024 and state["done"] is True
    assert state["last_checkpoint"] == "step_1024"
    policy = load_policy(smoke_run)
    obs, _ = WormGoalEnv(obs_version=2).reset(seed=0)
    assert policy(obs).shape == (7,)


def test_checkpoints_keep_only_the_newest(smoke_run: Path) -> None:
    names = sorted(path.name for path in (smoke_run / "checkpoints").iterdir())
    assert names == ["step_1024.zip", "step_1024_vecnormalize.pkl"]


def test_curriculum_reaches_the_last_level_by_the_end_of_the_run(smoke_run: Path) -> None:
    assert progress(smoke_run)["level"] == 3


def test_saved_archives_record_no_machine_path(smoke_run: Path) -> None:
    for archive in (smoke_run / "policy.zip", smoke_run / "checkpoints" / "step_1024.zip"):
        with zipfile.ZipFile(archive) as source:
            assert "system_info.txt" not in source.namelist()
            data = source.read("data").decode()
        assert str(smoke_run) not in data and smoke_run.name not in data


def test_resume_continues_the_step_count(tmp_path: Path) -> None:
    train(smoke(steps=512, every=256), tmp_path)
    assert progress(tmp_path)["steps"] == 512
    train(smoke(steps=1024, every=256), tmp_path, resume=True)
    state = progress(tmp_path)
    assert state["steps"] == 1024 and state["done"] is True
    assert json.loads((tmp_path / "meta.json").read_text())["steps"] == 1024
    assert (tmp_path / "checkpoints" / "step_1024.zip").exists()


def test_resume_without_a_checkpoint_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        train(smoke(), tmp_path, resume=True)


def test_sigterm_saves_a_checkpoint(tmp_path: Path) -> None:
    out = tmp_path / "run"
    process = subprocess.Popen(
        [sys.executable, "scripts/train.py", "--config", str(CONFIGS / "terrain_v2_smoke.toml")]
        + ["--steps", "2000000", "--n-envs", "1", "--out", str(out)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        deadline = time.monotonic() + 120.0
        while not list(out.glob("checkpoints/step_*.zip")):
            assert process.poll() is None, process.stderr.read().decode()  # type: ignore[union-attr]
            assert time.monotonic() < deadline, "no checkpoint within 120 s"
            time.sleep(0.2)
        before = progress(out)["steps"]
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == 0
    finally:
        process.kill()
    state = progress(out)
    assert state["done"] is False and state["steps"] >= before
    assert (out / "checkpoints" / f"{state['last_checkpoint']}.zip").exists()
    assert state["last_checkpoint"] == f"step_{state['steps']}"
    assert not (out / "policy.zip").exists()


def test_old_train_signature_still_works(tmp_path: Path) -> None:
    run = train(256, 1, 0, tmp_path)
    assert (run / "policy.zip").exists() and (run / "vecnormalize.pkl").exists()
    assert json.loads((run / "meta.json").read_text())["obs_version"] == 1
