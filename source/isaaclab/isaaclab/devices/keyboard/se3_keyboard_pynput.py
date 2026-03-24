# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control using pynput for OS-level key capture.

This device bypasses the Omniverse GUI focus issue where UI widgets capture
W/A/S/D/Q/E before they reach the carb keyboard subscription. pynput captures
keys at the OS level, so teleop works regardless of which window has focus.

Requires: pip install pynput

Note: May have issues on Wayland; use X11 (e.g. XDG_SESSION_TYPE=x11) if needed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..device_base import DeviceBase, DeviceCfg

try:
    from pynput import keyboard
except ImportError as e:
    raise ImportError(
        "pynput is required for Se3KeyboardPynput. Install with: pip install pynput"
    ) from e


class Se3KeyboardPynput(DeviceBase):
    """A keyboard controller for SE(3) using pynput (OS-level capture).

    Captures keyboard input at the OS level, bypassing GUI focus issues
    where Omniverse UI widgets consume W/A/S/D before teleop receives them.

    Key bindings (same as Se3Keyboard):
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    K
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================
    """

    def __init__(self, cfg: Se3KeyboardPynputCfg):
        """Initialize the keyboard layer.

        Args:
            cfg: Configuration object for keyboard settings.
        """
        super().__init__()
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device

        self._create_key_bindings()

        self._close_gripper = False
        self._additional_callbacks: dict[str, Callable[[], None]] = {}

        self._lock = threading.Lock()
        self._active_keys: set[str] = set()
        self._last_seen: dict[str, float] = {}  # key -> last press time (for debouncing key repeat)
        self._pending_callbacks: list[str] = []
        self._release_debounce_s = 0.1  # Only remove key after no press for this long

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def __del__(self):
        """Stop the keyboard listener."""
        try:
            if hasattr(self, "_listener") and self._listener is not None:
                self._listener.stop()
        except Exception:
            pass

    def __str__(self) -> str:
        """Returns: A string containing the information of the keyboard."""
        msg = f"Keyboard Controller for SE(3) (pynput): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): K\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        msg += "\tRotate arm along x-axis: Z/X\n"
        msg += "\tRotate arm along y-axis: T/G\n"
        msg += "\tRotate arm along z-axis: C/V"
        return msg

    def reset(self):
        """Reset the command buffers."""
        with self._lock:
            self._reset_internal()

    def _reset_internal(self):
        """Reset state (must be called with _lock held)."""
        self._close_gripper = False
        self._active_keys.clear()
        self._last_seen.clear()
        self._pending_callbacks.clear()

    def add_callback(self, key: str, func: Callable[[], None]):
        """Add a callback for when a key is pressed.

        Args:
            key: The keyboard button (e.g. "R", "L").
            func: The function to call when key is pressed (no arguments).
        """
        self._additional_callbacks[key.upper()] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from keyboard event state.

        Returns:
            torch.Tensor: A 7-element tensor containing:
                - delta pose: First 6 elements as [x, y, z, rx, ry, rz].
                - gripper command: Last element (+1.0 open, -1.0 close).
        """
        with self._lock:
            now = time.time()
            # Remove keys that haven't been seen recently (debounce key-repeat release)
            stale = [k for k in self._active_keys if now - self._last_seen.get(k, 0) > self._release_debounce_s]
            for k in stale:
                self._active_keys.discard(k)
                self._last_seen.pop(k, None)
            # Compute delta from currently held keys
            delta_pos = np.zeros(3)
            delta_rot = np.zeros(3)
            for k in self._active_keys:
                if k in ["W", "S", "A", "D", "Q", "E"]:
                    delta_pos += self._INPUT_KEY_MAPPING[k]
                elif k in ["Z", "X", "T", "G", "C", "V"]:
                    delta_rot += self._INPUT_KEY_MAPPING[k]
            pending = list(self._pending_callbacks)
            self._pending_callbacks.clear()

        for key in pending:
            if key in self._additional_callbacks:
                self._additional_callbacks[key]()

        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
        command = np.concatenate([delta_pos, rot_vec])
        if self.gripper_term:
            gripper_value = -1.0 if self._close_gripper else 1.0
            command = np.append(command, gripper_value)

        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def _on_press(self, key):
        """Handle key press (called from pynput thread)."""
        k = self._key_to_char(key)
        if k is None:
            return
        with self._lock:
            self._last_seen[k] = time.time()
            if k not in self._active_keys:
                self._active_keys.add(k)
                if k == "L":
                    self._reset_internal()
                    return
                if k == "K":
                    self._close_gripper = not self._close_gripper
                    return
                if k in self._additional_callbacks:
                    self._pending_callbacks.append(k)
                    return

    def _on_release(self, key):
        """Handle key release (called from pynput thread).

        We do NOT remove from _active_keys here. Key repeat sends rapid press-release
        cycles; removing on release would clear the key before advance() reads it.
        Instead, advance() removes keys that haven't been seen for _release_debounce_s.
        """
        # No-op: removal is done in advance() via debounce

    def _key_to_char(self, key) -> str | None:
        """Convert pynput key to uppercase char for mapping."""
        try:
            if hasattr(key, "char") and key.char is not None:
                return key.char.upper()
            if hasattr(key, "name"):
                return key.name.upper()
        except Exception:
            pass
        return None

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }


@dataclass
class Se3KeyboardPynputCfg(DeviceCfg):
    """Configuration for SE3 keyboard devices (pynput)."""

    gripper_term: bool = True
    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    retargeters: None = None
    class_type: type[DeviceBase] = Se3KeyboardPynput
