"""
Real Robot Data Collection Script

Runs trained policy on the real robot (like pro_real_robot_eval.py) but focuses
on data collection: synchronized RGB images from Intel RealSense camera alongside
policy observations, actions, and filtered forces. Live matplotlib plots of force
data during trials.

Usage:
    python eval/real_robot_data_collection.py \
        --tag "MATCH:2024-01-15_10:00" --policy_idx 0 \
        --output_dir ./data_collection_output --num_episodes 20

    # Without camera:
    python eval/real_robot_data_collection.py \
        --tag "MATCH:2024-01-15_10:00" --policy_idx 0 \
        --output_dir ./data_collection_output --no_camera

    # Without live plot:
    python eval/real_robot_data_collection.py \
        --tag "MATCH:2024-01-15_10:00" --policy_idx 0 \
        --output_dir ./data_collection_output --no_plot
"""

import argparse
import json
import multiprocessing
import os
import select
import shutil
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.SimBa import SimBaNet
from real_robot_exps.pro_robot_interface import FrankaInterface, StateSnapshot, make_ee_target_pose
from real_robot_exps.robot_interface import SafetyViolation
from real_robot_exps.observation_builder import ObservationBuilder, ObservationNormalizer, OBS_DIM_MAP
from real_robot_exps.hybrid_controller import RealRobotController, get_euler_xyz


# ============================================================================
# Forge eval noise ranges (matching wandb_eval.py NOISE_RANGES)
# ============================================================================

FORGE_NOISE_RANGES = [
    (0.0, 0.001, "0mm-1mm"),
    (0.001, 0.0025, "1mm-2.5mm"),
    (0.0025, 0.005, "2.5mm-5mm"),
    (0.005, 0.0075, "5mm-7.5mm"),
]


# ============================================================================
# Config loading (reused from pro_real_robot_eval.py)
# ============================================================================

def load_real_robot_config(config_path: str, overrides: Optional[List[str]] = None) -> dict:
    """Load real robot config from YAML and apply CLI overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if overrides:
        for override in overrides:
            if '=' not in override:
                raise ValueError(f"Override must be 'key=value', got: {override}")
            key_path, value_str = override.split('=', 1)
            keys = key_path.split('.')

            parent = config
            for k in keys[:-1]:
                if k not in parent:
                    raise ValueError(f"Config key not found: {key_path}")
                parent = parent[k]

            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    if value_str.lower() == 'true':
                        value = True
                    elif value_str.lower() == 'false':
                        value = False
                    else:
                        value = value_str

            parent[keys[-1]] = value
            print(f"  Override: {key_path} = {value}")

    return config


# ============================================================================
# Checkpoint caching (reused from pro_real_robot_eval.py)
# ============================================================================

def sanitize_tag(tag: str) -> str:
    """Convert a WandB tag to a filesystem-safe directory name."""
    return tag.replace(':', '_').replace('/', '_').replace(' ', '_')


def save_to_cache(
    cache_path: str,
    config_temp_dir: str,
    runs: list,
    best_checkpoints: Dict[str, int],
    checkpoint_paths: Dict[str, Tuple[str, str]],
    best_scores: Optional[Dict[str, dict]] = None,
) -> None:
    """Save WandB data to local cache directory."""
    os.makedirs(cache_path, exist_ok=True)

    base_src = os.path.join(config_temp_dir, 'config_base.yaml')
    if not os.path.exists(base_src):
        raise RuntimeError(f"Base config not found in temp dir: {base_src}")
    shutil.copy2(base_src, os.path.join(cache_path, 'config_base.yaml'))

    exp_src = os.path.join(config_temp_dir, 'config_experiment.yaml')
    if os.path.exists(exp_src):
        with open(exp_src, 'r') as f:
            exp_data = yaml.safe_load(f)
        exp_data['base_config'] = os.path.join(cache_path, 'config_base.yaml')
        with open(os.path.join(cache_path, 'config_experiment.yaml'), 'w') as f:
            yaml.safe_dump(exp_data, f, default_flow_style=False)

    for run_id, (policy_path, critic_path) in checkpoint_paths.items():
        run_dir = os.path.join(cache_path, run_id)
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy2(policy_path, os.path.join(run_dir, 'policy.pt'))
        shutil.copy2(critic_path, os.path.join(run_dir, 'critic.pt'))

    runs_data = []
    for r in runs:
        entry = {
            'id': r.id,
            'name': r.name,
            'group': r.group,
            'project': r.project,
            'tags': list(r.tags),
            'best_step': best_checkpoints[r.id],
        }
        if best_scores and r.id in best_scores:
            entry['best_score'] = best_scores[r.id]['score']
            entry['best_successes'] = best_scores[r.id]['successes']
            entry['best_breaks'] = best_scores[r.id]['breaks']
        runs_data.append(entry)
    with open(os.path.join(cache_path, 'runs.json'), 'w') as f:
        json.dump(runs_data, f, indent=2)

    print(f"  Checkpoint cache saved to: {cache_path}")


def load_from_cache(
    cache_path: str,
    run_id_filter: Optional[str] = None,
) -> Tuple[dict, list, Dict[str, int], Dict[str, dict]]:
    """Load cached WandB data from local directory."""
    from types import SimpleNamespace
    from configs.config_manager_v3 import ConfigManagerV3

    runs_json_path = os.path.join(cache_path, 'runs.json')
    with open(runs_json_path, 'r') as f:
        runs_data = json.load(f)

    run_infos = []
    best_checkpoints = {}
    best_scores = {}
    for entry in runs_data:
        for required_field in ('group', 'project', 'tags'):
            if required_field not in entry:
                raise RuntimeError(
                    f"Cache is missing '{required_field}' field (old format). "
                    f"Delete the cache directory and re-download: {cache_path}"
                )
        run_infos.append(SimpleNamespace(
            id=entry['id'], name=entry['name'],
            group=entry['group'], project=entry['project'],
            tags=entry['tags'],
        ))
        best_checkpoints[entry['id']] = entry['best_step']
        if 'best_score' in entry:
            best_scores[entry['id']] = {
                'score': entry['best_score'],
                'successes': entry['best_successes'],
                'breaks': entry['best_breaks'],
            }

    if run_id_filter is not None:
        run_infos = [r for r in run_infos if r.id == run_id_filter]
        if len(run_infos) == 0:
            available = [e['id'] for e in runs_data]
            raise RuntimeError(
                f"Run ID '{run_id_filter}' not found in cache. "
                f"Available run IDs: {available}. "
                f"Delete the cache directory to re-download: {cache_path}"
            )
        best_checkpoints = {r.id: best_checkpoints[r.id] for r in run_infos}
        best_scores = {r.id: best_scores[r.id] for r in run_infos if r.id in best_scores}

    for run_info in run_infos:
        run_dir = os.path.join(cache_path, run_info.id)
        for fname in ['policy.pt', 'critic.pt']:
            fpath = os.path.join(run_dir, fname)
            if not os.path.exists(fpath):
                raise RuntimeError(
                    f"Cached checkpoint missing: {fpath}. "
                    f"Delete the cache directory to re-download: {cache_path}"
                )

    config_manager = ConfigManagerV3()
    exp_path = os.path.join(cache_path, 'config_experiment.yaml')
    base_path = os.path.join(cache_path, 'config_base.yaml')

    if os.path.exists(exp_path):
        configs = config_manager.process_config(exp_path)
    elif os.path.exists(base_path):
        configs = config_manager.process_config(base_path)
    else:
        raise RuntimeError(
            f"No config YAML files found in cache: {cache_path}. "
            f"Delete the cache directory to re-download: {cache_path}"
        )

    print(f"  Loaded {len(run_infos)} run(s) from cache:")
    for r in run_infos:
        print(f"    - {r.id} ({r.name}) [best step: {best_checkpoints[r.id]}]")

    return configs, run_infos, best_checkpoints, best_scores


# ============================================================================
# obs_order reconstruction from training config
# ============================================================================

def reconstruct_obs_order(configs: dict) -> list:
    """Reconstruct the obs_order that was used during training."""
    obs_order = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]

    ft_cfg = configs['wrappers'].force_torque_sensor
    if getattr(ft_cfg, 'add_force_obs', False):
        obs_order.append("force_torque")
    if getattr(ft_cfg, 'add_contact_obs', False):
        obs_order.append("in_contact")

    env_cfg = configs['environment']
    if hasattr(env_cfg, 'obs_rand') and getattr(env_cfg.obs_rand, 'use_fixed_asset_yaw_noise', False):
        obs_order.append("fingertip_yaw_rel_fixed")

    return obs_order


# ============================================================================
# Policy loading (no sim env needed)
# ============================================================================

def load_single_agent_policy(
    policy_path: str,
    configs: dict,
    obs_dim: int,
    device: str = "cpu",
) -> Tuple[SimBaNet, ObservationNormalizer, dict]:
    """Load a trained policy network and normalizer from checkpoint."""
    checkpoint = torch.load(policy_path, map_location=device, weights_only=False)

    if 'net_state_dict' not in checkpoint:
        raise RuntimeError(f"Policy checkpoint missing 'net_state_dict': {policy_path}")
    if 'state_preprocessor' not in checkpoint:
        raise RuntimeError(f"Policy checkpoint missing 'state_preprocessor': {policy_path}")

    net_input_dim = checkpoint['net_state_dict']['input.0.weight'].shape[1]
    if net_input_dim != obs_dim:
        raise RuntimeError(
            f"obs_dim mismatch: obs_order+action_dim gives {obs_dim} but "
            f"network input layer expects {net_input_dim}. "
            f"Check that obs_order reconstruction matches training config."
        )

    actor_n = configs['model'].actor.n
    actor_latent = configs['model'].actor.latent_size
    use_state_dependent_std = getattr(configs['model'].actor, 'use_state_dependent_std', False)

    hybrid_enabled = configs['wrappers'].hybrid_control.enabled
    vic_enabled = getattr(configs['wrappers'].vic_pose, 'enabled', False)
    if hybrid_enabled:
        from configs.cfg_exts.ctrl_mode import get_force_size
        ctrl_mode = getattr(configs['primary'], 'ctrl_mode', 'force_only')
        force_size = get_force_size(ctrl_mode)
        sigma_idx = force_size
        action_dim = 2 * force_size + 6
    elif vic_enabled:
        sigma_idx = 0
        action_dim = 9
    else:
        sigma_idx = 0
        action_dim = 6

    squash_actions = getattr(configs['model'], 'squash_actions', False)
    tan_out = (sigma_idx == 0) and (not squash_actions)

    if use_state_dependent_std:
        std_out_dim = action_dim - sigma_idx
    else:
        std_out_dim = 0

    out_size = action_dim + std_out_dim

    policy_net = SimBaNet(
        n=actor_n,
        in_size=obs_dim,
        out_size=out_size,
        latent_size=actor_latent,
        device=device,
        tan_out=tan_out,
    )
    policy_net.load_state_dict(checkpoint['net_state_dict'])
    policy_net.eval()

    normalizer = ObservationNormalizer(
        checkpoint['state_preprocessor'], device=device, obs_dim=obs_dim
    )

    if use_state_dependent_std:
        log_std = None
    else:
        if 'log_std' not in checkpoint:
            raise RuntimeError(
                f"Policy checkpoint missing 'log_std' (required for non-state-dependent std): {policy_path}"
            )
        log_std = checkpoint['log_std'].to(device)

    model_info = {
        'sigma_idx': sigma_idx,
        'action_dim': action_dim,
        'use_state_dependent_std': use_state_dependent_std,
        'squash_actions': squash_actions,
        'obs_dim': obs_dim,
        'log_std': log_std,
    }

    return policy_net, normalizer, model_info


# ============================================================================
# Action inference
# ============================================================================

@torch.no_grad()
def get_action(
    policy_net: SimBaNet,
    normalizer: ObservationNormalizer,
    obs: torch.Tensor,
    model_info: dict,
    std_scale: float = 0.0,
) -> torch.Tensor:
    """Get action from policy, optionally with stochastic sampling."""
    norm_obs = normalizer.normalize(obs.unsqueeze(0))
    raw_output = policy_net(norm_obs)
    mean_action = raw_output[0, :model_info['action_dim']]
    sigma_idx = model_info['sigma_idx']

    if std_scale <= 0.0:
        if sigma_idx == 0:
            if model_info.get('squash_actions', False):
                return torch.tanh(mean_action)
            else:
                return mean_action
        else:
            selection = torch.sigmoid(mean_action[:sigma_idx])
            components = torch.tanh(mean_action[sigma_idx:])
            return torch.cat([selection, components])

    # Stochastic path
    if model_info['log_std'] is not None:
        log_std = model_info['log_std'].squeeze(0)
    else:
        log_std = raw_output[0, model_info['action_dim']:]

    log_std = torch.clamp(log_std, -20.0, 2.0)
    std = torch.exp(log_std) * std_scale

    if sigma_idx == 0:
        noise = torch.randn_like(mean_action)
        if model_info.get('squash_actions', False):
            return torch.tanh(mean_action + std * noise)
        else:
            return mean_action + std * noise
    else:
        selection = (torch.sigmoid(mean_action[:sigma_idx]) > 0.5).float()
        raw_components = mean_action[sigma_idx:]
        noise = torch.randn_like(raw_components)

        if model_info.get('squash_actions', False):
            components = torch.tanh(raw_components + std * noise)
        else:
            components = torch.tanh(raw_components) + std * noise

        return torch.cat([selection, components])


# ============================================================================
# Detection logic
# ============================================================================

def check_success(
    ee_pos: torch.Tensor,
    ee_to_peg_base_offset: torch.Tensor,
    target_peg_base_pos: torch.Tensor,
    xy_centering_threshold: float,
    hole_height: float,
    threshold: float,
) -> bool:
    """Check if peg is successfully inserted."""
    peg_base_pos = ee_pos + ee_to_peg_base_offset
    xy_dist = torch.norm(peg_base_pos[:2] - target_peg_base_pos[:2])
    z_disp = peg_base_pos[2] - target_peg_base_pos[2]
    is_centered = xy_dist < xy_centering_threshold
    is_inserted = z_disp < hole_height * threshold
    return bool(is_centered and is_inserted)


def check_break(
    force_torque: torch.Tensor,
    break_force_threshold: float,
) -> bool:
    """Check if force exceeds break threshold."""
    force_magnitude = torch.norm(force_torque[:3])
    return bool(force_magnitude >= break_force_threshold)


# ============================================================================
# Keyboard controller (skip key changed to 'x' to avoid conflict with save)
# ============================================================================

class EvalKeyboardController:
    """Non-blocking keyboard listener for eval control.

    Keys (during episode):
        'x' - skip: end current episode immediately (counted as BREAK)
        'p' - pause: finish current episode, then pause before next

    Keys (while paused):
        'c' - calibrate: move robot to goal XY, 5cm above goal Z
        Enter - resume: continue running episodes

    Keys (any time):
        ESC - quit: end current episode immediately and shut down
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._skip = False
        self._pause = False
        self._resume = False
        self._calibrate = False
        self._quit = False
        self._paused = False
        self._stop = threading.Event()
        self._old_settings = None

    def start(self):
        """Save terminal settings, set raw mode, start listener thread."""
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        self._stop.clear()
        thread = threading.Thread(target=self._read_loop, daemon=True)
        thread.start()

    def stop(self):
        """Restore terminal settings."""
        self._stop.set()
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    @property
    def should_skip(self) -> bool:
        with self._lock:
            val = self._skip
            self._skip = False
            return val

    @property
    def should_pause(self) -> bool:
        with self._lock:
            return self._pause

    @property
    def should_calibrate(self) -> bool:
        with self._lock:
            val = self._calibrate
            self._calibrate = False
            return val

    @property
    def should_resume(self) -> bool:
        with self._lock:
            val = self._resume
            self._resume = False
            return val

    @property
    def should_quit(self) -> bool:
        with self._lock:
            return self._quit

    def set_paused(self, paused: bool):
        with self._lock:
            self._paused = paused
            if not paused:
                self._pause = False
                self._resume = False
                self._calibrate = False

    @staticmethod
    def raw_print(msg: str):
        """Print with \\r\\n so output isn't garbled in raw terminal mode."""
        sys.stdout.write(msg + "\r\n")
        sys.stdout.flush()

    def _read_loop(self):
        """Background thread: poll stdin for keypresses."""
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if ready:
                ch = sys.stdin.read(1)
                with self._lock:
                    if ch == '\x1b':
                        self._quit = True
                        self._skip = True
                    elif ch.lower() == 'x' and not self._paused:
                        self._skip = True
                    elif ch.lower() == 'p' and not self._paused:
                        self._pause = True
                    elif ch.lower() == 'c' and self._paused:
                        self._calibrate = True
                    elif ch in ('\r', '\n') and self._paused:
                        self._resume = True


# ============================================================================
# RealsenseCapture — separate process for RGB capture (avoids GIL contention)
# ============================================================================

def _camera_process(
    cmd_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event,
    fps: int,
    width: int,
    height: int,
    serial_number: Optional[str],
    show_preview: bool = False,
):
    """Camera capture process. Runs continuously to keep autofocus active.

    Always captures frames from the sensor, but only stores them in the
    buffer when recording is enabled via 'RECORD' command.

    Commands on cmd_queue:
        'RECORD' — start buffering frames
        'FLUSH'  — stop buffering, send buffered frames via result_queue
    """
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()

    if serial_number is not None:
        config.enable_device(serial_number)

    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    pipeline.start(config)

    buffer = []
    recording = False

    try:
        while not stop_event.is_set():
            # Check for commands (non-blocking)
            try:
                cmd = cmd_queue.get_nowait()
                if cmd == 'RECORD':
                    buffer.clear()
                    recording = True
                elif cmd == 'FLUSH':
                    recording = False
                    result_queue.put(buffer)
                    buffer = []
            except Exception:
                pass

            try:
                frames = pipeline.wait_for_frames(timeout_ms=100)
            except RuntimeError:
                if stop_event.is_set():
                    break
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb_array = np.asanyarray(color_frame.get_data())

            if recording:
                wall_ts = time.time()
                buffer.append((wall_ts, rgb_array.copy()))

            if show_preview:
                bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                cv2.imshow('Camera', bgr)
                cv2.waitKey(1)
    finally:
        pipeline.stop()
        if show_preview:
            cv2.destroyWindow('Camera')


class RealsenseCapture:
    """Captures RGB frames from Intel RealSense camera in a separate process.

    The camera process runs continuously (keeping autofocus active).
    Call start_recording() to begin buffering frames and stop_recording()
    to retrieve them.
    """

    def __init__(self, fps: int = 60, width: int = 640, height: int = 480,
                 serial_number: Optional[str] = None, show_preview: bool = False):
        import pyrealsense2 as rs

        # Validate camera is available by starting/stopping pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        if serial_number is not None:
            ctx = rs.context()
            devices = ctx.query_devices()
            found = False
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == serial_number:
                    found = True
                    break
            if not found:
                available = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
                raise RuntimeError(
                    f"RealSense camera with serial '{serial_number}' not found. "
                    f"Available: {available}"
                )
            config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        try:
            pipeline.start(config)
            pipeline.stop()
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to initialize RealSense camera: {e}. "
                f"Check that a RealSense camera is connected."
            ) from e

        self._fps = fps
        self._width = width
        self._height = height
        self._serial_number = serial_number
        self._show_preview = show_preview
        self._cmd_queue = multiprocessing.Queue()
        self._result_queue = multiprocessing.Queue()
        self._stop_event = multiprocessing.Event()
        self._process = None

    def start(self):
        """Start the camera process (keeps autofocus active)."""
        self._stop_event.clear()
        self._process = multiprocessing.Process(
            target=_camera_process,
            args=(self._cmd_queue, self._result_queue, self._stop_event,
                  self._fps, self._width, self._height, self._serial_number,
                  self._show_preview),
            daemon=True,
        )
        self._process.start()

    def start_recording(self):
        """Begin buffering frames."""
        self._cmd_queue.put('RECORD')

    def stop_recording(self, timeout: float = 30.0) -> list:
        """Stop buffering and return all recorded frames.

        Returns list of (wall_timestamp, rgb_array).
        """
        self._cmd_queue.put('FLUSH')
        try:
            return self._result_queue.get(timeout=timeout)
        except Exception:
            raise RuntimeError(
                f"Camera process did not return frames within {timeout}s. "
                f"Camera process may have crashed."
            )

    def stop(self):
        """Shut down camera process entirely."""
        if self._process is not None:
            self._stop_event.set()
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None


# ============================================================================
# LiveForcePlotter — runs in separate process
# ============================================================================

def _plotter_process(queue: multiprocessing.Queue, is_hybrid: bool):
    """Main function for the plotter subprocess.

    Reads from queue, updates matplotlib plots at ~10Hz.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if is_hybrid:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
        ax2 = None
        ax3 = None

    fig.suptitle('Live Force Data')

    # Position window to the right so it doesn't overlap the terminal/camera
    try:
        manager = fig.canvas.manager
        manager.window.wm_geometry("+900+100")
    except Exception:
        pass

    # Data buffers
    times = []
    fx_meas, fy_meas, fz_meas = [], [], []
    fx_cmd, fy_cmd, fz_cmd = [], [], []
    sel_x, sel_y, sel_z = [], [], []
    act_x, act_y, act_z = [], [], []

    # Measured force lines
    line_fx, = ax1.plot([], [], 'r-', linewidth=1.5, label='Fx measured')
    line_fy, = ax1.plot([], [], 'g-', linewidth=1.5, label='Fy measured')
    line_fz, = ax1.plot([], [], 'b-', linewidth=1.5, label='Fz measured')

    # Commanded force lines (hybrid only)
    if is_hybrid:
        line_fx_cmd, = ax1.plot([], [], 'r--', linewidth=1.0, label='Fx commanded')
        line_fy_cmd, = ax1.plot([], [], 'g--', linewidth=1.0, label='Fy commanded')
        line_fz_cmd, = ax1.plot([], [], 'b--', linewidth=1.0, label='Fz commanded')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Force (N)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3)

    # Selection probability subplot (hybrid only)
    if ax2 is not None:
        line_sx, = ax2.plot([], [], 'r-', linewidth=1.5, label='Sel prob X')
        line_sy, = ax2.plot([], [], 'g-', linewidth=1.5, label='Sel prob Y')
        line_sz, = ax2.plot([], [], 'b-', linewidth=1.5, label='Sel prob Z')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Selection Probability')
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlim(0, 10)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

    # Actual selection subplot (hybrid only)
    if ax3 is not None:
        line_ax, = ax3.plot([], [], 'r-', linewidth=1.5, label='Sel X')
        line_ay, = ax3.plot([], [], 'g-', linewidth=1.5, label='Sel Y')
        line_az, = ax3.plot([], [], 'b-', linewidth=1.5, label='Sel Z')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Actual Selection')
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_xlim(0, 10)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    def update(_frame):
        # Drain queue
        batch_count = 0
        while batch_count < 100:  # process up to 100 items per update
            try:
                msg = queue.get_nowait()
            except Exception:
                break
            batch_count += 1

            if msg is None:
                # Shutdown signal
                plt.close(fig)
                return

            if isinstance(msg, str) and msg.startswith('SAVE:'):
                save_path = msg[5:]
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                continue

            # msg is (timestamp, force_xyz, sel_probability, sel_matrix, target_force)
            t, force_xyz, sel_prob, sel_mat, target_force = msg
            times.append(t)
            fx_meas.append(force_xyz[0])
            fy_meas.append(force_xyz[1])
            fz_meas.append(force_xyz[2])

            if target_force is not None:
                fx_cmd.append(target_force[0])
                fy_cmd.append(target_force[1])
                fz_cmd.append(target_force[2])

            if sel_prob is not None:
                sel_x.append(sel_prob[0])
                sel_y.append(sel_prob[1])
                sel_z.append(sel_prob[2])

            if sel_mat is not None:
                act_x.append(sel_mat[0])
                act_y.append(sel_mat[1])
                act_z.append(sel_mat[2])

        if not times:
            return

        # Update lines
        line_fx.set_data(times, fx_meas)
        line_fy.set_data(times, fy_meas)
        line_fz.set_data(times, fz_meas)

        if is_hybrid and fx_cmd:
            line_fx_cmd.set_data(times[:len(fx_cmd)], fx_cmd)
            line_fy_cmd.set_data(times[:len(fy_cmd)], fy_cmd)
            line_fz_cmd.set_data(times[:len(fz_cmd)], fz_cmd)

        if ax2 is not None and sel_x:
            line_sx.set_data(times[:len(sel_x)], sel_x)
            line_sy.set_data(times[:len(sel_y)], sel_y)
            line_sz.set_data(times[:len(sel_z)], sel_z)

        if ax3 is not None and act_x:
            line_ax.set_data(times[:len(act_x)], act_x)
            line_ay.set_data(times[:len(act_y)], act_y)
            line_az.set_data(times[:len(act_z)], act_z)

        # Auto-scale x axis
        max_t = times[-1]
        if max_t > ax1.get_xlim()[1] * 0.9:
            new_xlim = max_t * 1.2
            ax1.set_xlim(0, new_xlim)
            if ax2 is not None:
                ax2.set_xlim(0, new_xlim)
            if ax3 is not None:
                ax3.set_xlim(0, new_xlim)

        # Auto-scale y axis for force (10% padding, minimum range of [-1, 1])
        all_forces = fx_meas + fy_meas + fz_meas
        if is_hybrid:
            all_forces = all_forces + fx_cmd + fy_cmd + fz_cmd
        if all_forces:
            data_min = min(all_forces)
            data_max = max(all_forces)
            margin = max((data_max - data_min) * 0.1, 0.5)
            ymin = min(data_min - margin, -1)
            ymax = max(data_max + margin, 1)
            ax1.set_ylim(ymin, ymax)

    ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.show()


class LiveForcePlotter:
    """Manages force plotter in a separate process.

    Started/stopped per episode. Call start() before the step loop and
    stop() after the save/skip prompt to close the window.
    """

    def __init__(self, is_hybrid: bool):
        self._is_hybrid = is_hybrid
        self._queue = None
        self._process = None

    def start(self):
        """Spawn plotter process and open the plot window."""
        self._queue = multiprocessing.Queue(maxsize=500)
        self._process = multiprocessing.Process(
            target=_plotter_process,
            args=(self._queue, self._is_hybrid),
            daemon=True,
        )
        self._process.start()

    def send(self, timestamp: float, force_xyz: np.ndarray,
             sel_probability: Optional[np.ndarray], sel_matrix: Optional[np.ndarray],
             target_force: Optional[np.ndarray]):
        """Non-blocking queue push from policy loop."""
        if self._queue is None:
            return
        try:
            self._queue.put_nowait((timestamp, force_xyz, sel_probability, sel_matrix, target_force))
        except Exception:
            pass  # Queue full — drop data point (visualization only)

    def save(self, path: str):
        """Save the current plot to a file (png)."""
        if self._queue is None:
            return
        try:
            self._queue.put(f'SAVE:{path}', timeout=2.0)
        except Exception:
            pass

    def stop(self):
        """Shut down plotter process and close the window."""
        if self._process is None:
            return
        try:
            self._queue.put_nowait(None)  # shutdown signal
        except Exception:
            pass
        self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
        self._process = None
        self._queue = None


# ============================================================================
# save_episode_data — saves camera frames + policy data to disk
# ============================================================================

def _align_frames_to_policy_steps(frames, policy_steps):
    """Align camera frames to the most recent prior policy step.

    For each camera frame, finds the last policy step whose wall timestamp
    is <= the frame's wall timestamp. This is the policy state that was
    actually in effect when the frame was captured.

    Args:
        frames: List of (wall_timestamp, rgb_array) from camera process.
        policy_steps: List of dicts with 'wall_timestamp', 'step', 'obs',
                      'action', 'force', 'target_force', 'sel_probability'.

    Returns:
        List of (wall_ts, rgb_array, matched_policy_dict) where
        matched_policy_dict has the same keys as policy_steps entries,
        or None values if the frame was captured before any policy step.
    """
    aligned = []
    ps_idx = 0  # pointer into policy_steps (both lists are chronological)
    n_ps = len(policy_steps)

    for wall_ts, rgb in frames:
        # Advance pointer to the last policy step with wall_timestamp <= frame wall_ts
        while ps_idx < n_ps - 1 and policy_steps[ps_idx + 1]['wall_timestamp'] <= wall_ts:
            ps_idx += 1

        if n_ps > 0 and policy_steps[ps_idx]['wall_timestamp'] <= wall_ts:
            aligned.append((wall_ts, rgb, policy_steps[ps_idx]))
        else:
            # Frame captured before any policy step
            aligned.append((wall_ts, rgb, None))

    return aligned


def save_episode_data(
    output_dir: str,
    tag: str,
    episode_idx: int,
    frames: list,
    policy_steps: list,
    result: dict,
    image_format: str = "mp4",
    jpeg_quality: int = 95,
    camera_fps: int = 60,
    image_resolution: Tuple[int, int] = (640, 480),
    forge_range_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    video_fourcc: str = "mp4v",
):
    """Save episode data (video or images + numeric data + metadata) to disk.

    Args:
        output_dir: Base output directory.
        tag: Experiment tag (sanitized for filesystem).
        episode_idx: Episode number.
        frames: List of (wall_timestamp, rgb_array) tuples from camera process.
        policy_steps: List of dicts from run_episode with wall_timestamp, step,
                      obs, action, force, target_force, sel_probability.
        result: Episode result dict from run_episode().
        image_format: "mp4" for video, or "jpg"/"png" for individual images.
        jpeg_quality: JPEG quality (1-100, ignored for mp4).
        camera_fps: Camera capture rate (used as video FPS for mp4).
        image_resolution: (width, height).
        forge_range_name: If forge eval, the noise range name.
        start_time: Episode start datetime.
        end_time: Episode end datetime.
        video_fourcc: FourCC codec string for mp4 output (e.g. "mp4v", "avc1").
    """
    if image_format not in ("mp4", "jpg", "png"):
        raise ValueError(f"image_format must be 'mp4', 'jpg', or 'png', got: '{image_format}'")

    # Align camera frames to most recent prior policy step
    aligned = _align_frames_to_policy_steps(frames, policy_steps)

    tag_safe = sanitize_tag(tag)
    if forge_range_name is not None:
        ep_dir = os.path.join(output_dir, tag_safe, forge_range_name, f"episode_{episode_idx:03d}")
    else:
        ep_dir = os.path.join(output_dir, tag_safe, f"episode_{episode_idx:03d}")
    os.makedirs(ep_dir, exist_ok=True)

    num_frames = len(aligned)

    # Determine data dimensions from first aligned frame with policy data
    obs_dim = None
    act_dim = None
    for _, _, pd in aligned:
        if pd is not None:
            obs_dim = pd['obs'].shape[0]
            act_dim = pd['action'].shape[0]
            break

    # Build numeric arrays
    wall_timestamps = np.zeros(num_frames, dtype=np.float64)
    matched_policy_steps = np.zeros(num_frames, dtype=np.int32)
    episode_timestamps = np.zeros(num_frames, dtype=np.float64)
    observations = np.zeros((num_frames, obs_dim if obs_dim else 1), dtype=np.float32)
    actions = np.zeros((num_frames, act_dim if act_dim else 1), dtype=np.float32)
    forces = np.zeros((num_frames, 6), dtype=np.float32)
    target_forces = np.zeros((num_frames, 6), dtype=np.float32)
    sel_probabilities = np.zeros((num_frames, 6), dtype=np.float32)

    ep_start_wall = aligned[0][0] if aligned else 0.0

    # Set up video writer or image directory
    video_writer = None
    if image_format == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*video_fourcc)
        video_path = os.path.join(ep_dir, "episode.mp4")
        width, height = image_resolution
        video_writer = cv2.VideoWriter(video_path, fourcc, camera_fps, (width, height))
        if not video_writer.isOpened():
            raise RuntimeError(
                f"Failed to open VideoWriter with fourcc='{video_fourcc}' at {video_path}. "
                f"Check that OpenCV was built with the required codec support."
            )
    else:
        img_dir = os.path.join(ep_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

    for i, (wall_ts, rgb, pd) in enumerate(aligned):
        wall_timestamps[i] = wall_ts
        episode_timestamps[i] = wall_ts - ep_start_wall

        if pd is not None:
            matched_policy_steps[i] = pd['step']
            observations[i] = pd['obs']
            actions[i] = pd['action']
            forces[i] = pd['force']
            if pd['target_force'] is not None:
                target_forces[i] = pd['target_force']
            if pd['sel_probability'] is not None:
                sel_probabilities[i] = pd['sel_probability']
        else:
            matched_policy_steps[i] = -1

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if image_format == "mp4":
            video_writer.write(bgr)
        elif image_format == "jpg":
            cv2.imwrite(os.path.join(img_dir, f"frame_{i:06d}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        else:
            cv2.imwrite(os.path.join(img_dir, f"frame_{i:06d}.png"), bgr)

    if video_writer is not None:
        video_writer.release()

    # Save numeric data
    np.savez_compressed(
        os.path.join(ep_dir, "episode_data.npz"),
        wall_timestamps=wall_timestamps,
        policy_steps=matched_policy_steps,
        episode_timestamps=episode_timestamps,
        observations=observations,
        actions=actions,
        forces=forces,
        target_forces=target_forces,
        sel_probabilities=sel_probabilities,
    )

    # Save metadata
    metadata = {
        'tag': tag,
        'episode_idx': episode_idx,
        'num_frames': num_frames,
        'num_policy_steps': len(policy_steps),
        'succeeded': result['succeeded'],
        'terminated': result['terminated'],
        'engaged': result['engaged'],
        'episode_length': result['length'],
        'camera_fps': camera_fps,
        'image_resolution': list(image_resolution),
        'image_format': image_format,
        'start_time_iso': start_time.isoformat() if start_time else None,
        'end_time_iso': end_time.isoformat() if end_time else None,
    }
    if forge_range_name is not None:
        metadata['forge_range_name'] = forge_range_name

    with open(os.path.join(ep_dir, "episode_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    return ep_dir


# ============================================================================
# Single episode execution (stripped version for data collection)
# ============================================================================

def run_episode(
    robot: FrankaInterface,
    policy_net: SimBaNet,
    normalizer: ObservationNormalizer,
    model_info: dict,
    obs_builder: ObservationBuilder,
    controller: RealRobotController,
    real_config: dict,
    episode_noise: dict,
    hand_init_pos: torch.Tensor,
    hand_init_orn: list,
    hand_init_orn_noise: list,
    keyboard: EvalKeyboardController,
    device: str = "cpu",
    std_scale: float = 0.0,
    live_plotter: Optional[LiveForcePlotter] = None,
    camera: Optional['RealsenseCapture'] = None,
) -> Optional[dict]:
    """Run a single episode on the real robot for data collection.

    Returns:
        Episode result dict with: succeeded, engaged, terminated, length,
        success_step, termination_step, policy_steps (list of per-step dicts).
    """
    task_cfg = real_config['task']
    fixed_asset_position = torch.tensor(task_cfg['fixed_asset_position'], device=device, dtype=torch.float32)
    obs_frame_z_offset = task_cfg['hole_height'] + task_cfg['fixed_asset_base_height']
    goal_position = fixed_asset_position.clone()
    goal_position[2] += obs_frame_z_offset
    target_peg_base_pos = torch.tensor(task_cfg['target_peg_base_position'], device=device, dtype=torch.float32)
    ee_to_peg_base_offset = torch.tensor(task_cfg['ee_to_peg_base_offset'], device=device, dtype=torch.float32)

    xy_centering = task_cfg['xy_centering_threshold']
    hole_height = task_cfg['hole_height']
    success_threshold = task_cfg['success_threshold']
    engage_threshold = task_cfg['engage_threshold']
    break_force_threshold = task_cfg['break_force_threshold']
    max_steps = task_cfg['episode_timeout_steps']
    terminate_on_success = task_cfg['terminate_on_success']

    is_hybrid = model_info['sigma_idx'] > 0
    if is_hybrid:
        if 'force_select_max_height_mm' not in task_cfg:
            raise RuntimeError(
                "task.force_select_max_height_mm is required in config.yaml for hybrid models."
            )
        if 'force_select_max_xy_dist_mm' not in task_cfg:
            raise RuntimeError(
                "task.force_select_max_xy_dist_mm is required in config.yaml for hybrid models."
            )
        force_select_max_height_m = task_cfg['force_select_max_height_mm'] / 1000.0
        force_select_max_xy_dist_m = task_cfg['force_select_max_xy_dist_mm'] / 1000.0

    # Per-episode noise
    pos_noise = episode_noise['goal_pos_noise']
    noisy_goal = goal_position + pos_noise
    yaw_offset = episode_noise['goal_yaw_noise']
    start_pos_noise = episode_noise['start_pos_noise']

    target_ee_pos = goal_position + hand_init_pos + start_pos_noise
    target_rpy = [hand_init_orn[0], hand_init_orn[1], hand_init_orn[2]]
    target_pose = make_ee_target_pose(target_ee_pos.cpu().numpy(), np.array(target_rpy))

    # Retract + reset with retry logic
    MAX_MOTION_RETRIES = 5
    retract_height = real_config['robot']['retract_height_m']
    for attempt in range(MAX_MOTION_RETRIES):
        try:
            robot.retract_up(retract_height)
            robot.reset_to_start_pose(target_pose)
            break
        except RuntimeError as e:
            print(f"  [MOTION RETRY {attempt+1}/{MAX_MOTION_RETRIES}] {e}")
            try:
                robot.error_recovery()
            except RuntimeError:
                pass
            time.sleep(1.0)
    else:
        print(f"  [MOTION FAILED] All {MAX_MOTION_RETRIES} retries exhausted")
        return None

    # Calibrate FT bias
    ft_bias = robot.calibrate_ft_bias()

    prev_actions = torch.zeros(model_info['action_dim'], device=device)

    # Episode tracking (simplified — no eval metrics)
    succeeded = False
    engaged = False
    terminated = False
    success_step = -1
    termination_step = -1

    contact_force_threshold = obs_builder.contact_force_threshold

    # Initialize controller
    snap = robot.get_state_snapshot()
    controller.reset(snap.ee_pos, noisy_goal)

    # Warmup policy + controller
    for _wi in range(3):
        _warmup_obs = obs_builder.build_observation(snap, noisy_goal, prev_actions, fixed_yaw_offset=yaw_offset)
        _warmup_action = get_action(policy_net, normalizer, _warmup_obs, model_info, std_scale=std_scale)
        _warmup_ctrl = controller.compute_action(
            _warmup_action, snap.ee_pos, snap.ee_quat, snap.ee_linvel, snap.ee_angvel,
            snap.force_torque, snap.joint_pos, snap.joint_vel, snap.jacobian, snap.mass_matrix,
            noisy_goal,
        )
    controller.reset(snap.ee_pos, noisy_goal)
    prev_actions = torch.zeros(model_info['action_dim'], device=device)

    # Start recording frames (camera process is already running for autofocus)
    if camera is not None:
        camera.start_recording()
    if live_plotter is not None:
        live_plotter.start()

    camera_frames = []
    step = 0

    try:
        # Start torque control
        robot.start_torque_mode(log_trajectory=False)
        _ep_start = time.time()

        # Policy step log: each entry is a dict with wall_timestamp, step, obs, action, etc.
        policy_step_log = []

        # Enter raw mode only for the step loop (keyboard skip/quit detection)
        keyboard.start()

        for step in range(max_steps):
            robot.wait_for_policy_step()
            snap = robot.get_state_snapshot()
            robot.check_safety(snap)

            if keyboard.should_skip:
                terminated = True
                termination_step = step
                break

            obs = obs_builder.build_observation(snap, noisy_goal, prev_actions, fixed_yaw_offset=yaw_offset)

            action = get_action(policy_net, normalizer, obs, model_info, std_scale=std_scale)

            ctrl_output = controller.compute_action(
                action, snap.ee_pos, snap.ee_quat, snap.ee_linvel, snap.ee_angvel,
                snap.force_torque, snap.joint_pos, snap.joint_vel, snap.jacobian, snap.mass_matrix,
                noisy_goal,
            )

            # Safety: block force selection when peg tip is too far from hole
            if is_hybrid:
                peg_tip_pos = snap.ee_pos + ee_to_peg_base_offset
                z_above_hole_m = (peg_tip_pos[2] - goal_position[2]).item()
                xy_dist_m = torch.norm(peg_tip_pos[:2] - goal_position[:2]).item()
                sel = ctrl_output['sel_matrix']
                any_force_selected = (sel > 0.5).any().item()

                should_block = (
                    any_force_selected
                    and (z_above_hole_m > force_select_max_height_m
                         or xy_dist_m > force_select_max_xy_dist_m)
                )
                if should_block:
                    zero_sel = torch.zeros(6, device=device)
                    ctrl_output['sel_matrix'] = zero_sel
                    ctrl_output['control_targets'] = ctrl_output['control_targets']._replace(
                        sel_matrix=zero_sel,
                    )

            # Set control targets for 1kHz torque recomputation (do this FIRST,
            # before any data collection work, so the robot gets new targets ASAP)
            robot.set_control_targets(ctrl_output['control_targets'])

            # Update prev_actions
            prev_actions = ctrl_output['ema_actions']

            # --- Data collection instrumentation (after control targets are set) ---
            _wall_now = time.time()
            _now = _wall_now - _ep_start
            sel_prob = ctrl_output.get('sel_probability', None)
            tf = ctrl_output.get('target_force', None)

            policy_step_log.append({
                'wall_timestamp': _wall_now,
                'step': step,
                'obs': obs.detach().cpu().numpy().copy(),
                'action': action.detach().cpu().numpy().copy(),
                'force': snap.force_torque.detach().cpu().numpy().copy(),
                'target_force': tf.detach().cpu().numpy().copy() if tf is not None else None,
                'sel_probability': sel_prob.detach().cpu().numpy().copy() if sel_prob is not None else None,
            })

            if live_plotter is not None:
                sel_mat = ctrl_output.get('sel_matrix', None)
                live_plotter.send(
                    timestamp=_now,
                    force_xyz=snap.force_torque[:3].detach().cpu().numpy(),
                    sel_probability=sel_prob.detach().cpu().numpy() if sel_prob is not None else None,
                    sel_matrix=sel_mat.detach().cpu().numpy() if sel_mat is not None else None,
                    target_force=ctrl_output.get('target_force', torch.zeros(6)).detach().cpu().numpy(),
                )

            # Check termination conditions
            if not engaged:
                is_engaged = check_success(
                    snap.ee_pos, ee_to_peg_base_offset, target_peg_base_pos,
                    xy_centering, hole_height, engage_threshold,
                )
                if is_engaged:
                    engaged = True

            if not succeeded:
                is_success = check_success(
                    snap.ee_pos, ee_to_peg_base_offset, target_peg_base_pos,
                    xy_centering, hole_height, success_threshold,
                )
                if is_success:
                    succeeded = True
                    success_step = step
                    if terminate_on_success:
                        break

            if not terminated:
                is_break = check_break(snap.force_torque, break_force_threshold)
                if is_break:
                    terminated = True
                    termination_step = step
                    break

    finally:
        # Always clean up keyboard and camera even on error
        keyboard.stop()

        if camera is not None:
            try:
                camera_frames = camera.stop_recording()
            except Exception:
                pass

    _ep_elapsed = time.time() - _ep_start
    print(f"  [TIMING] {step+1} steps in {_ep_elapsed:.2f}s ({_ep_elapsed/(step+1)*1000:.1f}ms/step)")

    robot.end_control()

    episode_length = step + 1

    return {
        'succeeded': succeeded,
        'engaged': engaged,
        'terminated': terminated,
        'length': episode_length,
        'success_step': success_step if success_step >= 0 else episode_length,
        'termination_step': termination_step if termination_step >= 0 else episode_length,
        'policy_steps': policy_step_log,
        'camera_frames': camera_frames,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Real Robot Data Collection")
    parser.add_argument("--tag", type=str, required=True, help="WandB experiment tag")
    parser.add_argument("--policy_idx", type=int, required=True, help="Policy index (0-based)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for collected data")
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--eval_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="real_robot_exps/config.yaml", help="Config path")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--override", action="append", default=[], help="Override config values")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint_cache",
                        help="Local directory for caching downloaded checkpoints")
    parser.add_argument("--forge_eval", action="store_true", default=False,
                        help="Forge eval mode: spherical goal noise sampling across 4 ranges")
    parser.add_argument("--start_forge_idx", type=int, default=0,
                        help="Forge eval: skip to this noise range index (0-based)")
    parser.add_argument("--autoplay_mode", action="store_true",
                        help="Skip all Enter prompts, auto-save all episodes")
    parser.add_argument("--no_camera", action="store_true", help="Disable RealSense camera capture")
    parser.add_argument("--no_preview", action="store_true", help="Disable live camera preview window")
    parser.add_argument("--no_plot", action="store_true", help="Disable live force plots")
    parser.add_argument("--run_id", type=str, default=None, help="Filter to specific run ID in cache")
    parser.add_argument("--entity", type=str, default="hur", help="WandB entity")
    parser.add_argument("--project", type=str, default="SG_Exps", help="WandB project")
    args = parser.parse_args()

    # Validate
    if args.start_forge_idx != 0 and not args.forge_eval:
        raise ValueError("--start_forge_idx requires --forge_eval")
    if args.start_forge_idx < 0 or args.start_forge_idx >= len(FORGE_NOISE_RANGES):
        raise ValueError(
            f"--start_forge_idx {args.start_forge_idx} out of range "
            f"[0, {len(FORGE_NOISE_RANGES) - 1}]"
        )

    torch.manual_seed(args.eval_seed)

    print("=" * 80)
    print("REAL ROBOT DATA COLLECTION")
    print("=" * 80)

    # 1. Load real robot config
    print(f"\nLoading config: {args.config}")
    real_config = load_real_robot_config(args.config, args.override)

    from configs.cfg_exts.version_compat import set_no_sim_mode
    set_no_sim_mode(True)

    # 2. Check for cached checkpoints or download from WandB
    cache_path = os.path.abspath(os.path.join(args.checkpoint_dir, sanitize_tag(args.tag)))
    cache_exists = os.path.exists(os.path.join(cache_path, 'runs.json'))

    if not cache_exists:
        print(f"\nNo local cache found. Downloading from WandB...")

        import wandb
        from eval.checkpoint_utils import (
            query_runs_by_tag,
            reconstruct_config_from_wandb,
            download_checkpoint_pair,
            get_best_checkpoints_for_runs,
        )

        runs_wb = query_runs_by_tag(args.tag, args.entity, args.project, run_id=None)

        print("\nReconstructing training config...")
        configs_wb, temp_dir = reconstruct_config_from_wandb(runs_wb[0])

        print("\nFinding best checkpoints...")
        api = wandb.Api(timeout=60)
        best_checkpoints_wb, best_scores_wb = get_best_checkpoints_for_runs(
            api, runs_wb, args.tag, args.entity, args.project
        )

        print("\nDownloading checkpoints...")
        checkpoint_paths = {}
        download_dirs = []
        for run_wb in runs_wb:
            policy_path, critic_path = download_checkpoint_pair(run_wb, best_checkpoints_wb[run_wb.id])
            checkpoint_paths[run_wb.id] = (policy_path, critic_path)
            download_dirs.append(os.path.dirname(os.path.dirname(policy_path)))

        save_to_cache(cache_path, temp_dir, runs_wb, best_checkpoints_wb, checkpoint_paths, best_scores_wb)

        for d in download_dirs:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print(f"\nUsing cached checkpoints from: {cache_path}")

    configs, runs, best_checkpoints, best_scores = load_from_cache(cache_path, args.run_id)

    # Select policy by index
    if args.policy_idx < 0 or args.policy_idx >= len(runs):
        print(f"\nAvailable policies for tag '{args.tag}':")
        for i, r in enumerate(runs):
            print(f"  [{i}] {r.name} (id={r.id}, best step={best_checkpoints[r.id]})")
        raise ValueError(
            f"--policy_idx {args.policy_idx} out of range [0, {len(runs)-1}]"
        )
    selected_run = runs[args.policy_idx]
    print(f"\n  Selected policy [{args.policy_idx}]: '{selected_run.name}' (id={selected_run.id})")

    # 3. Reconstruct obs_order and determine model properties
    obs_order = reconstruct_obs_order(configs)

    hybrid_enabled = configs['wrappers'].hybrid_control.enabled
    vic_enabled = getattr(configs['wrappers'].vic_pose, 'enabled', False)
    if hybrid_enabled:
        from configs.cfg_exts.ctrl_mode import get_force_size
        ctrl_mode = getattr(configs['primary'], 'ctrl_mode', 'force_only')
        force_size = get_force_size(ctrl_mode)
        action_dim = 2 * force_size + 6
    elif vic_enabled:
        action_dim = 9
    else:
        action_dim = 6

    ft_cfg = configs['wrappers'].force_torque_sensor
    use_tanh = getattr(ft_cfg, 'use_tanh_scaling', False)
    tanh_scale = getattr(ft_cfg, 'tanh_scale', 0.03)
    contact_threshold = getattr(ft_cfg, 'contact_force_threshold', 1.5)
    exclude_torques = getattr(ft_cfg, 'exclude_torques', False)
    ee_pose_noise_enabled = getattr(configs['wrappers'].ee_pose_noise, 'enabled', False)

    # Noise config
    noise_cfg = real_config.get('noise', {})
    use_rr_noise = noise_cfg.get('use_rr_noise', False)

    if use_rr_noise:
        goal_pos_noise_scale = torch.tensor(noise_cfg['goal_pos_noise'], device=args.device, dtype=torch.float32)
        use_fixed_asset_yaw_noise = noise_cfg['use_fixed_asset_yaw_noise']
        goal_yaw_noise_scale = noise_cfg['goal_yaw_noise'] if use_fixed_asset_yaw_noise else 0.0
        hand_init_pos = torch.tensor(noise_cfg['hand_init_pos'], device=args.device, dtype=torch.float32)
        hand_init_pos_noise = torch.tensor(noise_cfg['hand_init_pos_noise'], device=args.device, dtype=torch.float32)
        hand_init_orn = list(noise_cfg['hand_init_orn'])
        hand_init_orn_noise = list(noise_cfg['hand_init_orn_noise'])
    else:
        obs_rand = configs['environment'].obs_rand
        goal_pos_noise_scale = torch.tensor(obs_rand.fixed_asset_pos, device=args.device, dtype=torch.float32)
        use_fixed_asset_yaw_noise = hasattr(obs_rand, 'use_fixed_asset_yaw_noise') and obs_rand.use_fixed_asset_yaw_noise
        goal_yaw_noise_scale = obs_rand.fixed_asset_yaw if use_fixed_asset_yaw_noise else 0.0

        cfg_task = getattr(configs['environment'], 'task', None) or configs['environment']
        hand_init_pos = torch.tensor(getattr(cfg_task, 'hand_init_pos', [0.0, 0.0, 0.047]),
                                     device=args.device, dtype=torch.float32)
        hand_init_pos_noise = torch.tensor(getattr(cfg_task, 'hand_init_pos_noise', [0.02, 0.02, 0.01]),
                                           device=args.device, dtype=torch.float32)
        hand_init_orn = list(getattr(cfg_task, 'hand_init_orn', [3.1416, 0.0, 0.0]))
        hand_init_orn_noise = list(getattr(cfg_task, 'hand_init_orn_noise', [0.0, 0.0, 0.785]))

    # 4. Initialize observation builder
    fixed_asset_yaw = real_config['task']['fixed_asset_yaw']
    obs_builder = ObservationBuilder(
        obs_order=obs_order,
        action_dim=action_dim,
        use_tanh_ft_scaling=use_tanh,
        tanh_ft_scale=tanh_scale,
        contact_force_threshold=contact_threshold,
        fixed_asset_yaw=fixed_asset_yaw,
        ee_pose_noise_enabled=ee_pose_noise_enabled,
        exclude_torques=exclude_torques,
        device=args.device,
    )

    # 5. Initialize robot interface
    print("\nInitializing robot interface...")
    robot = FrankaInterface(real_config, device=args.device)
    print("\nClosing gripper...")
    robot.close_gripper()

    # 6. Initialize controller
    print("\nInitializing controller...")
    controller = RealRobotController(configs, real_config, device=args.device)

    std_scale = real_config.get('policy', {}).get('std_scale', 0.0)
    if std_scale > 0.0:
        print(f"[Policy] Stochastic sampling ENABLED (std_scale={std_scale})")
    else:
        print("[Policy] Deterministic (mean only)")

    # 7. Load policy
    run_id = selected_run.id
    best_step = best_checkpoints[run_id]
    policy_path = os.path.join(cache_path, run_id, 'policy.pt')
    policy_net, normalizer, model_info = load_single_agent_policy(
        policy_path, configs, obs_dim=obs_builder.obs_dim, device=args.device,
    )
    obs_builder.validate_against_checkpoint(model_info['obs_dim'])

    # 8. Data collection config
    dc_config = real_config.get('data_collection', {})
    cam_cfg = dc_config.get('camera', {})

    # 9. Initialize camera (unless --no_camera)
    camera = None
    if not args.no_camera:
        print("\nInitializing RealSense camera...")
        camera = RealsenseCapture(
            fps=cam_cfg.get('fps', 60),
            width=cam_cfg.get('width', 640),
            height=cam_cfg.get('height', 480),
            serial_number=cam_cfg.get('serial_number', None),
            show_preview=not args.no_preview,
        )
        camera.start()
        print(f"  Camera started: {cam_cfg.get('width', 640)}x{cam_cfg.get('height', 480)} @ {cam_cfg.get('fps', 60)}Hz")

    # 10. Initialize live plotter (unless --no_plot)
    plotter = None
    if not args.no_plot:
        print("\nLive force plotter: enabled")
        plotter = LiveForcePlotter(is_hybrid=hybrid_enabled)

    # 11. Compute calibration pose
    task_cfg = real_config['task']
    fixed_asset_position = torch.tensor(task_cfg['fixed_asset_position'], device=args.device, dtype=torch.float32)
    obs_frame_z_offset = task_cfg['hole_height'] + task_cfg['fixed_asset_base_height']
    cal_goal = fixed_asset_position.clone()
    cal_goal[2] += obs_frame_z_offset + 0.05
    cal_pose = make_ee_target_pose(cal_goal.cpu().numpy(), np.array(hand_init_orn))
    retract_height = real_config['robot']['retract_height_m']

    # 12. Move to calibration pose
    print("\nMoving to calibration pose (goal XY, 5cm above goal Z)...")
    robot.retract_up(retract_height)
    robot.reset_to_start_pose(cal_pose)
    snap = robot.get_state_snapshot()
    print(f"  Calibration pose: xyz=[{snap.ee_pos[0].item():.4f}, "
          f"{snap.ee_pos[1].item():.4f}, {snap.ee_pos[2].item():.4f}]")
    if not args.autoplay_mode:
        input("  Press Enter to begin data collection...")

    # 13. Pre-generate episode noise
    if args.forge_eval:
        total_episodes = args.num_episodes * len(FORGE_NOISE_RANGES)
        forge_range_indices = []
        goal_pos_noises = []
        for range_idx, (min_val, max_val, _range_name) in enumerate(FORGE_NOISE_RANGES):
            for _ in range(args.num_episodes):
                direction = torch.randn(3, device=args.device)
                direction = direction / direction.norm()
                radius = torch.rand(1, device=args.device).item() * (max_val - min_val) + min_val
                goal_pos_noises.append(direction * radius)
                forge_range_indices.append(range_idx)
    else:
        total_episodes = args.num_episodes
        forge_range_indices = None
        goal_pos_noises = [
            torch.randn(3, device=args.device) * goal_pos_noise_scale
            for _ in range(total_episodes)
        ]

    episode_noises = []
    for ep_idx in range(total_episodes):
        episode_noises.append({
            'goal_pos_noise': goal_pos_noises[ep_idx],
            'goal_yaw_noise': (torch.randn(1, device=args.device) * goal_yaw_noise_scale).item()
                              if goal_yaw_noise_scale > 0 else 0.0,
            'start_pos_noise': (2 * torch.rand(3, device=args.device) - 1) * hand_init_pos_noise,
        })

    # 14. Create keyboard controller (started/stopped per-episode inside run_episode)
    keyboard = EvalKeyboardController()

    print(f"{'=' * 80}")
    if args.forge_eval:
        print(f"DATA COLLECTION: {args.num_episodes} EPISODES x {len(FORGE_NOISE_RANGES)} RANGES = {total_episodes} TOTAL")
    else:
        print(f"DATA COLLECTION: {total_episodes} EPISODES")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Camera: {'enabled' if camera else 'disabled'}")
    print(f"  Live plot: {'enabled' if plotter else 'disabled'}")
    print(f"{'=' * 80}")
    print("  Keyboard controls (during episode):")
    print("    'x' = skip (end episode as BREAK)")
    print("    'p' = pause (finish episode, then pause)")
    print("    ESC = quit")
    print(f"{'=' * 80}")

    saved_episodes = 0

    try:
        for ep_idx in range(total_episodes):
            # Determine forge range info
            forge_range_name = None
            if args.forge_eval:
                range_idx = forge_range_indices[ep_idx]
                if range_idx < args.start_forge_idx:
                    continue
                forge_range_name = FORGE_NOISE_RANGES[range_idx][2]

            # Wait for Enter to begin (terminal is in normal mode here)
            if args.autoplay_mode:
                ep_label = f"\nEpisode {ep_idx+1}/{total_episodes}"
                if forge_range_name:
                    ep_label += f" [{forge_range_name}]"
                print(ep_label)
            else:
                prompt = f"\nEpisode {ep_idx+1}/{total_episodes}"
                if forge_range_name:
                    prompt += f" [{forge_range_name}]"
                prompt += " — Press Enter to begin (or 'q' to quit): "
                user_input = input(prompt).strip().lower()
                if user_input == 'q':
                    print("Quitting...")
                    break

            # Run episode (keyboard raw mode is managed inside run_episode)
            ep_start_time = datetime.now(timezone.utc)
            result = None
            for _attempt in range(5):
                try:
                    result = run_episode(
                        robot, policy_net, normalizer, model_info,
                        obs_builder, controller, real_config,
                        episode_noises[ep_idx],
                        hand_init_pos,
                        hand_init_orn, hand_init_orn_noise,
                        keyboard,
                        args.device,
                        std_scale=std_scale,
                        live_plotter=plotter,
                        camera=camera,
                    )
                    break
                except (RuntimeError, SafetyViolation) as e:
                    print(f"  [EPISODE RETRY {_attempt+1}/5] {e}")
                    # Clean up plotter on error (camera/keyboard cleaned in run_episode finally)
                    if plotter is not None:
                        try:
                            plotter.stop()
                        except Exception:
                            pass
                    try:
                        robot.end_control()
                    except Exception:
                        pass
                    try:
                        robot.error_recovery()
                    except Exception:
                        pass
                    time.sleep(1.0)

            ep_end_time = datetime.now(timezone.utc)

            if result is None:
                print("  [ABORT] Motion failed after retries")
                continue

            # Get captured frames (returned from run_episode via stop_capture)
            frames = result.get('camera_frames', [])

            # Determine outcome string
            if result['succeeded'] and not result['terminated']:
                outcome = "SUCCESS"
            elif result['terminated']:
                outcome = "BREAK"
            else:
                outcome = "TIMEOUT"

            # Terminal is in normal mode here (keyboard.stop() called inside run_episode)
            print(f"  Episode {ep_idx+1}: {outcome}, {result['length']} steps, {len(frames)} frames captured")

            if keyboard.should_quit:
                print("  [QUIT] Shutting down...")
                break

            if args.autoplay_mode:
                # Auto-save everything
                save_idx = ep_idx % args.num_episodes if args.forge_eval else ep_idx
                ep_path = save_episode_data(
                    output_dir=args.output_dir,
                    tag=args.tag,
                    episode_idx=save_idx,
                    frames=frames,
                    policy_steps=result.get('policy_steps', []),
                    result=result,
                    image_format=cam_cfg.get('image_format', 'mp4'),
                    jpeg_quality=cam_cfg.get('jpeg_quality', 95),
                    camera_fps=cam_cfg.get('fps', 60),
                    image_resolution=(cam_cfg.get('width', 640), cam_cfg.get('height', 480)),
                    forge_range_name=forge_range_name,
                    start_time=ep_start_time,
                    end_time=ep_end_time,
                    video_fourcc=cam_cfg.get('video_fourcc', 'mp4v'),
                )
                saved_episodes += 1
                if plotter is not None:
                    plot_path = os.path.join(ep_path, "force_plot.png")
                    plotter.save(plot_path)
                    time.sleep(0.5)
                    plotter.stop()
                print(f"  Saved to {ep_path}")

                # Check if pause was requested during the episode
                if keyboard.should_pause:
                    keyboard.set_paused(True)
                    print("  [PAUSED] 'c' = calibrate, Enter = resume, 'q' = quit")
                    while True:
                        pause_input = input("  > ").strip().lower()
                        if pause_input == 'q':
                            keyboard.set_paused(False)
                            raise KeyboardInterrupt
                        elif pause_input == 'c':
                            print("  [CALIBRATING] Moving to goal XY, 5cm above goal Z...")
                            robot.retract_up(retract_height)
                            robot.reset_to_start_pose(cal_pose)
                            snap = robot.get_state_snapshot()
                            print(f"  [CALIBRATED] xyz=[{snap.ee_pos[0].item():.4f}, "
                                  f"{snap.ee_pos[1].item():.4f}, {snap.ee_pos[2].item():.4f}]")
                            print("  [PAUSED] 'c' = calibrate, Enter = resume, 'q' = quit")
                        elif pause_input == '':
                            keyboard.set_paused(False)
                            print("  [RESUMED]")
                            break
            else:
                # Interactive save/continue prompt (plotter stays open so user can inspect)
                while True:
                    user_choice = input("  Press 's' to save, Enter to skip, 'q' to quit: ").strip().lower()
                    if user_choice == 's':
                        # For forge eval, use within-range index so each range starts at 000
                        save_idx = ep_idx % args.num_episodes if args.forge_eval else ep_idx
                        ep_path = save_episode_data(
                            output_dir=args.output_dir,
                            tag=args.tag,
                            episode_idx=save_idx,
                            frames=frames,
                            policy_steps=result.get('policy_steps', []),
                            result=result,
                            image_format=cam_cfg.get('image_format', 'mp4'),
                            jpeg_quality=cam_cfg.get('jpeg_quality', 95),
                            camera_fps=cam_cfg.get('fps', 60),
                            image_resolution=(cam_cfg.get('width', 640), cam_cfg.get('height', 480)),
                            forge_range_name=forge_range_name,
                            start_time=ep_start_time,
                            end_time=ep_end_time,
                            video_fourcc=cam_cfg.get('video_fourcc', 'mp4v'),
                        )
                        saved_episodes += 1
                        if plotter is not None:
                            plot_path = os.path.join(ep_path, "force_plot.png")
                            plotter.save(plot_path)
                            time.sleep(0.5)
                        print(f"  Saved to {ep_path}")
                        break
                    elif user_choice == 'q':
                        print("Quitting...")
                        raise KeyboardInterrupt
                    elif user_choice == '':
                        print("  Skipped (frames discarded)")
                        break
                    else:
                        print("  Invalid input. Press 's' to save, Enter to skip, 'q' to quit.")

                # Close plotter window after save/skip decision
                if plotter is not None:
                    plotter.stop()

                # Check if pause was requested during the episode
                if keyboard.should_pause:
                    keyboard.set_paused(True)
                    print("  [PAUSED] 'c' = calibrate, Enter = resume, 'q' = quit")
                    while True:
                        pause_input = input("  > ").strip().lower()
                        if pause_input == 'q':
                            keyboard.set_paused(False)
                            raise KeyboardInterrupt
                        elif pause_input == 'c':
                            print("  [CALIBRATING] Moving to goal XY, 5cm above goal Z...")
                            robot.retract_up(retract_height)
                            robot.reset_to_start_pose(cal_pose)
                            snap = robot.get_state_snapshot()
                            print(f"  [CALIBRATED] xyz=[{snap.ee_pos[0].item():.4f}, "
                                  f"{snap.ee_pos[1].item():.4f}, {snap.ee_pos[2].item():.4f}]")
                            print("  [PAUSED] 'c' = calibrate, Enter = resume, 'q' = quit")
                        elif pause_input == '':
                            keyboard.set_paused(False)
                            print("  [RESUMED]")
                            break

    except KeyboardInterrupt:
        pass
    finally:
        keyboard.stop()

    # Cleanup
    print(f"\n{'=' * 80}")
    print(f"DATA COLLECTION COMPLETE")
    print(f"  Episodes saved: {saved_episodes}/{total_episodes}")
    print(f"  Output dir: {args.output_dir}")
    print(f"{'=' * 80}")

    if camera is not None:
        camera.stop()
    if plotter is not None:
        plotter.stop()

    robot.shutdown()


if __name__ == "__main__":
    main()
