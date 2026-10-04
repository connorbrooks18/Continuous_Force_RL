"""Dev tool: dump training-side outputs for the rl_policy_env parity tests.

Everything below calls the training code itself (apple_pick_gym + skrl), so the rig's
torch-only port in rl_policy_env.py is checked against the real thing, not a second copy.
Run from the rig repo root with the sim's environment:

    uv run --project ~/codes/apple_pick_sim/.claude/worktrees/rl-skrl-ppo \\
        python real_robot_exps/tools/make_vic_harvest_parity_fixture.py \\
        --checkpoint checkpoint_cache/vic_harvest/v2b_s0/ckpt_000016000 \\
        --wall-checkpoint checkpoint_cache/vic_harvest/v2c_s0/ckpt_000009600

``--wall-checkpoint`` ([D17m], a checkpoint trained with the no-advance wall) adds its bounds and a
rollout of its actor; the tests find it under ``checkpoint_cache/vic_harvest/<run>/<ckpt>``.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler

from apple_pick_gym.batched_envs.harvest_action import (
    HarvestActionBounds,
    clamp_target_advance,
    grip_axis,
    integrate_delta_pose,
    leash_target_pose,
    pack_vic_pose_action,
    split_harvest_action,
    tool_delta_to_world,
)
from apple_pick_gym.batched_envs.harvest_obs import _ACTOR_TOOL_FIELDS, tool_frame_obs
from apple_pick_gym.rl.action_scaling import HarvestActionScaler
from apple_pick_gym.rl.models import LstmGaussianActor, RecurrentNetConfig

N_OBS, N_ACT, N_WALL, T = 64, 64, 64, 40
OUT = Path(__file__).resolve().parents[1] / "testdata" / "vic_harvest_tool_parity.json"


def _unit_quat(n, g):
    q = torch.randn(n, 4, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def _xyzw(q_wxyz):
    return q_wxyz[:, [1, 2, 3, 0]]


def obs_cases(g, scaler):
    pos = torch.randn(N_OBS, 3, generator=g) * 0.3
    q = _unit_quat(N_OBS, g)
    vel = torch.randn(N_OBS, 6, generator=g) * 0.3
    start_pos = pos + torch.randn(N_OBS, 3, generator=g) * 0.05
    start_q = _unit_quat(N_OBS, g)
    tgt_pos = pos + torch.randn(N_OBS, 3, generator=g) * 0.02
    tgt_q = _unit_quat(N_OBS, g)
    ft_tool = torch.randn(N_OBS, 6, generator=g) * 5.0
    last_env = scaler.to_env(torch.rand(N_OBS, 13, generator=g) * 2 - 1)
    tool = tool_frame_obs(
        tcp_pos=pos, tcp_quat=_xyzw(q), tcp_velocity=vel, ft=torch.zeros(N_OBS, 6),
        start_pos=start_pos, start_quat=_xyzw(start_q), target_pos=tgt_pos, target_quat=_xyzw(tgt_q),
    )
    tool["ft_tool"] = ft_tool  # the env passes the sensor-frame reading through unchanged
    expected = torch.cat([tool[name] for name, _ in _ACTOR_TOOL_FIELDS] + [last_env], dim=-1)
    return [
        dict(tcp_pos=pos[i].tolist(), tcp_quat_wxyz=q[i].tolist(), linvel=vel[i, :3].tolist(),
             angvel=vel[i, 3:].tolist(), ft_tool=ft_tool[i].tolist(),
             start_pose_wxyz=torch.cat([start_pos[i], start_q[i]]).tolist(),
             target_pose_wxyz=torch.cat([tgt_pos[i], tgt_q[i]]).tolist(),
             last_action_env=last_env[i].tolist(), expected_obs=expected[i].tolist())
        for i in range(N_OBS)
    ]


def action_cases(g, scaler, b):
    u = torch.rand(N_ACT, 13, generator=g) * 2.4 - 1.2  # includes out-of-box values
    tcp = torch.cat([torch.randn(N_ACT, 3, generator=g) * 0.3, _unit_quat(N_ACT, g)], dim=-1)
    target = tcp.clone()
    target[:, :3] += torch.randn(N_ACT, 3, generator=g) * 0.1  # some beyond the 0.15 m leash
    target[:, 3:7] = _unit_quat(N_ACT, g)  # some beyond the 0.5 rad leash
    env_action = scaler.to_env(u)
    split = split_harvest_action(env_action, b)
    delta = tool_delta_to_world(split.delta, tcp[:, 3:7])
    new_target = integrate_delta_pose(target, delta)
    new_target = leash_target_pose(new_target, tcp, max_pos_offset_m=b.max_target_pos_offset_m,
                                   max_rot_offset_rad=b.max_target_rot_offset_rad)
    vic = pack_vic_pose_action(new_target, split.linear_k, split.angular_k, split.zeta)
    return [
        dict(u=u[i].tolist(), tcp_pose_wxyz=tcp[i].tolist(), target_pose_wxyz=target[i].tolist(),
             expected_env_action=env_action[i].tolist(), expected_target_pose_wxyz=new_target[i].tolist(),
             expected_vic_action=vic[i].tolist())
        for i in range(N_ACT)
    ]


def wall_cases(g, scaler, b, max_advance_m):
    """[D17m] The action chain with the no-advance wall, in the env's order: tool delta -> integrate ->
    leash (against the last TCP) -> wall (origin = the reset TCP, axis = its +z)."""
    b = dataclasses.replace(b, max_advance_m=max_advance_m)
    u = torch.rand(N_WALL, 13, generator=g) * 2.4 - 1.2
    tcp = torch.cat([torch.randn(N_WALL, 3, generator=g) * 0.3, _unit_quat(N_WALL, g)], dim=-1)
    start = torch.cat([tcp[:, :3] + torch.randn(N_WALL, 3, generator=g) * 0.02, _unit_quat(N_WALL, g)], dim=-1)
    target = tcp.clone()
    target[:, :3] += torch.randn(N_WALL, 3, generator=g) * 0.03  # about half land past the wall
    target[:, 3:7] = _unit_quat(N_WALL, g)
    env_action = scaler.to_env(u)
    split = split_harvest_action(env_action, b)
    leashed = leash_target_pose(
        integrate_delta_pose(target, tool_delta_to_world(split.delta, tcp[:, 3:7])), tcp,
        max_pos_offset_m=b.max_target_pos_offset_m, max_rot_offset_rad=b.max_target_rot_offset_rad,
    )
    axis = grip_axis(_xyzw(start[:, 3:7]))  # the env's start pose is obs-convention xyzw
    walled = clamp_target_advance(leashed, start[:, :3], axis, max_advance_m=b.max_advance_m)
    return [
        dict(u=u[i].tolist(), tcp_pose_wxyz=tcp[i].tolist(), start_pose_wxyz=start[i].tolist(),
             target_pose_wxyz=target[i].tolist(), grip_axis=axis[i].tolist(),
             leashed_target_pose_wxyz=leashed[i].tolist(), expected_target_pose_wxyz=walled[i].tolist())
        for i in range(N_WALL)
    ]


def policy_rollout(ckpt: Path, meta: dict, g):
    """The training actor + skrl's scaler on the eval path (RecurrentPolicyRunner.act): clamped mean."""
    mods = torch.load(ckpt / "agent.pt", map_location="cpu", weights_only=False)
    a = meta["config"]["actor"]
    cfg = RecurrentNetConfig(
        pre_mlp=tuple(a["pre_mlp"]), lstm_hidden=a["lstm_hidden"], lstm_layers=a["lstm_layers"],
        post_mlp=tuple(a["post_mlp"]), sequence_length=a["sequence_length"], mean_bound=a["mean_bound"],
    )
    box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), dtype=np.float32)
    actor = LstmGaussianActor(observation_space=box(40), state_space=box(139),
                              action_space=gym.spaces.Box(-1.0, 1.0, (13,), dtype=np.float32),
                              device="cpu", num_envs=1, cfg=cfg)
    actor.load_state_dict(mods["policy"])
    actor.eval()
    scaler = RunningStandardScaler(size=40, device="cpu")
    scaler.load_state_dict(mods["observation_preprocessor"])
    mean, std = scaler.running_mean, scaler.running_variance.sqrt()
    raw = (mean + std * torch.randn(T, 40, generator=g, dtype=mean.dtype)).float()  # in-distribution inputs
    rnn = [torch.zeros(1, 1, a["lstm_hidden"]), torch.zeros(1, 1, a["lstm_hidden"])]
    acts = []
    with torch.no_grad():
        for t in range(T):
            m, out = actor.compute({"observations": scaler(raw[t : t + 1], train=False), "rnn": rnn}, role="policy")
            rnn = out["rnn"]
            acts.append(m.clamp(-1.0, 1.0)[0])
    label = f"{ckpt.parent.name}/{ckpt.name}"  # <run>/<ckpt>, as under checkpoint_cache/vic_harvest
    return dict(checkpoint=label, raw_obs=raw.tolist(), expected_action=torch.stack(acts).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--wall-checkpoint", help="[D17m] a checkpoint whose action_bounds carry max_advance_m")
    args = p.parse_args()
    ckpt = Path(args.checkpoint)
    meta = json.loads((ckpt / "meta.json").read_text())
    b = HarvestActionBounds(**meta["action_bounds"])
    scaler = HarvestActionScaler(b)
    g = torch.Generator().manual_seed(1234)
    out = dict(bounds=meta["action_bounds"], obs_cases=obs_cases(g, scaler),
               action_cases=action_cases(g, scaler, b), policy_rollout=policy_rollout(ckpt, meta, g))
    out["wall_cases"] = {str(m): wall_cases(g, scaler, b, m) for m in (0.0, 0.005)}  # after the rest: same stream
    if args.wall_checkpoint:
        wall_ckpt = Path(args.wall_checkpoint)
        wall_meta = json.loads((wall_ckpt / "meta.json").read_text())
        if wall_meta["action_bounds"].get("max_advance_m") is None:
            raise SystemExit(f"{wall_ckpt}: action_bounds has no max_advance_m; not a [D17m] wall checkpoint")
        out["wall_bounds"] = wall_meta["action_bounds"]
        out["policy_rollout_wall"] = policy_rollout(wall_ckpt, wall_meta, g)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
