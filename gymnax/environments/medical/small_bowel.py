"""Small Bowel Environment for JAX."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


# Helper functions (JAX-compatible)
def _is_valid_pos(pos_vox: jnp.ndarray, shape: tuple[int, int, int]) -> jnp.ndarray:
    """Check if a position is within the volume bounds."""
    return jnp.all(pos_vox >= 0) & jnp.all(pos_vox < jnp.array(shape))


def get_patch_jax(
    volume: jnp.ndarray, center_coords: jnp.ndarray, patch_size: tuple[int, int, int]
) -> jnp.ndarray:
    """
    Extracts a patch from a 3D volume centered at center_coords.
    Handles padding for out-of-bounds regions.
    """
    volume_shape = jnp.array(volume.shape)
    patch_size_arr = jnp.array(patch_size)
    half_patch = patch_size_arr // 2

    start_coords = center_coords - half_patch
    end_coords = start_coords + patch_size_arr

    # Calculate padding
    pad_before = jnp.maximum(0, -start_coords)
    pad_after = jnp.maximum(0, end_coords - volume_shape)

    # Pad the volume
    padded_volume = jnp.pad(
        volume,
        (
            (pad_before[0], pad_after[0]),
            (pad_before[1], pad_after[1]),
            (pad_before[2], pad_after[2]),
        ),
        mode="constant",
        constant_values=0,
    )

    # Adjust slice coordinates for the padded volume
    padded_start_coords = start_coords + pad_before

    # Extract the patch
    patch = jax.lax.dynamic_slice(padded_volume, padded_start_coords, patch_size)
    return patch


def line_nd_jax(
    start: jnp.ndarray, end: jnp.ndarray, shape: tuple[int, int, int]
) -> jnp.ndarray:
    """
    JAX-compatible function to get voxels along a 3D line segment using Bresenham's algorithm.
    Returns a boolean mask of the line in the given shape.
    """
    # Ensure integer coordinates
    start = jnp.round(start).astype(jnp.int32)
    end = jnp.round(end).astype(jnp.int32)

    # Calculate differences
    dx, dy, dz = end - start

    # Determine step directions
    sx = jnp.sign(dx)
    sy = jnp.sign(dy)
    sz = jnp.sign(dz)

    # Absolute differences
    ax = jnp.abs(dx)
    ay = jnp.abs(dy)
    az = jnp.abs(dz)

    # Initialize current position
    x, y, z = start[0], start[1], start[2]

    # Initialize mask
    mask = jnp.zeros(shape, dtype=jnp.bool_)

    # Helper to update mask and position
    def _update_mask_and_pos(carry, _):
        current_x, current_y, current_z, err_1, err_2, err_3, current_mask = carry

        # Mark current voxel
        current_mask = jax.lax.dynamic_update_slice(
            current_mask, jnp.array([True]), (current_x, current_y, current_z)
        )

        # Bresenham logic
        e2 = 2 * err_1
        e3 = 2 * err_2
        e4 = 2 * err_3

        # Update x
        x_cond = (e2 > -ax) & (e3 > -ax) & (e4 > -ax)
        current_x = jnp.where(x_cond, current_x + sx, current_x)
        err_1 = jnp.where(x_cond, err_1 - ax, err_1)
        err_2 = jnp.where(x_cond, err_2 - ax, err_2)
        err_3 = jnp.where(x_cond, err_3 - ax, err_3)

        # Update y
        y_cond = (e2 < ay) & (e3 < ay) & (e4 < ay)
        current_y = jnp.where(y_cond, current_y + sy, current_y)
        err_1 = jnp.where(y_cond, err_1 + ay, err_1)
        err_2 = jnp.where(y_cond, err_2 + ay, err_2)
        err_3 = jnp.where(y_cond, err_3 + ay, err_3)

        # Update z
        z_cond = (e2 < az) & (e3 < az) & (e4 < az)
        current_z = jnp.where(z_cond, current_z + sz, current_z)
        err_1 = jnp.where(z_cond, err_1 + az, err_1)
        err_2 = jnp.where(z_cond, err_2 + az, err_2)
        err_3 = jnp.where(z_cond, err_3 + az, err_3)

        return (
            current_x,
            current_y,
            current_z,
            err_1,
            err_2,
            err_3,
            current_mask,
        ), None

    # Max length of the line segment
    max_len = jnp.max(jnp.array([ax, ay, az])) + 1

    # Initial errors
    err_1 = ax + ay + az
    err_2 = err_1 - 2 * ax
    err_3 = err_1 - 2 * ay
    err_4 = (
        err_1 - 2 * az
    )  # This was err_3 in the original, but it should be err_4 for 3D

    # Use lax.scan for the loop
    (final_x, final_y, final_z, final_err1, final_err2, final_err3, final_mask), _ = (
        jax.lax.scan(
            _update_mask_and_pos,
            (x, y, z, err_1, err_2, err_3, mask),
            None,
            length=max_len,
        )
    )

    # Mark the end point explicitly
    mask = jax.lax.dynamic_update_slice(
        final_mask, jnp.array([True]), (end[0], end[1], end[2])
    )

    return mask


def binary_dilation_3d_jax(mask: jnp.ndarray, radius: int) -> jnp.ndarray:
    """
    Performs 3D binary dilation on a boolean mask using a cubic structuring element.
    """
    if radius == 0:
        return mask

    # Create a cubic structuring element
    struct_elem_size = 2 * radius + 1
    struct_elem = jnp.ones(
        (struct_elem_size, struct_elem_size, struct_elem_size), dtype=jnp.bool_
    )

    # Pad the mask to handle boundaries during convolution
    padded_mask = jnp.pad(mask, radius, mode="constant", constant_values=False)

    # Use max pooling as dilation for binary masks
    dilated_mask = jax.lax.max_pool(
        padded_mask,
        window_shape=(struct_elem_size, struct_elem_size, struct_elem_size),
        strides=(1, 1, 1),
        padding="VALID",
    )
    return dilated_mask


def draw_path_sphere_2_jax(
    cumulative_path_mask: jnp.ndarray,
    line_voxels_mask: jnp.ndarray,
    dilation_radius: int,
    gt_path_vol: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Updates the cumulative path mask and ground truth path volume.
    This is a JAX-compatible version of the original TorchRL function.
    """
    # Dilate the line voxels to create a "sphere" around the path
    dilated_line_mask = binary_dilation_3d_jax(line_voxels_mask, dilation_radius)

    # Update cumulative_path_mask: union of current mask and new dilated path
    new_cumulative_path_mask = cumulative_path_mask | dilated_line_mask

    # Update gt_path_vol: this part is tricky as the original modifies in place.
    # The original logic was `self.gt_path_vol[self.cumulative_path_mask > 0] = 0`.
    # This means any part of the GT path that is now covered by the *new* cumulative path
    # should be zeroed out.
    new_gt_path_vol = gt_path_vol * (~new_cumulative_path_mask)

    return new_cumulative_path_mask, new_gt_path_vol


@struct.dataclass
class SmallBowelState(environment.EnvState):
    """State of the Small Bowel environment."""

    current_pos_vox: jnp.ndarray  # (3,) int
    cumulative_path_mask: jnp.ndarray  # (D, H, W) bool
    max_gdt_achieved: float
    cum_reward: float
    reward_map: jnp.ndarray  # (D, H, W) float, for peaks
    wall_gradient: float
    key: jax.Array


@struct.dataclass
class SmallBowelParams(environment.EnvParams):
    """Parameters for the Small Bowel environment."""

    image: jnp.ndarray  # (D, H, W) float
    seg: jnp.ndarray  # (D, H, W) bool
    wall_map: jnp.ndarray  # (D, H, W) float
    gdt_start: jnp.ndarray  # (D, H, W) float
    gdt_end: jnp.ndarray  # (D, H, W) float
    gt_path_vol: jnp.ndarray  # (D, H, W) bool
    local_peaks: jnp.ndarray  # (N, 3) int
    start_coord: jnp.ndarray  # (3,) int
    end_coord: jnp.ndarray  # (3,) int
    goal: jnp.ndarray  # (3,) int (this will be set in reset, but needs a default)

    r_zero_mov: float = 10.0
    r_val1: float = 1.0
    r_val2: float = 1.0
    r_final: float = 100.0
    r_peaks: float = 10.0
    gdt_max_increase_theta: float = 5.0
    max_step_vox: float = 5.0  # Max movement in voxels
    patch_size_vox: tuple[int, int, int] = (32, 32, 32)
    cumulative_path_radius_vox: int = 3
    seg_volume: float = 0.0  # Will be calculated from seg
    image_shape: tuple[int, int, int] = (0, 0, 0)  # Will be set from image shape


class SmallBowel(environment.Environment[SmallBowelState, SmallBowelParams]):
    """
    Gymnax-compatible environment for RL-based small bowel path tracking.
    Implemented in JAX.
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> SmallBowelParams:
        """Default environment parameters."""
        # These are placeholders. Actual data will be loaded externally.
        return SmallBowelParams(
            image=jnp.zeros((1, 1, 1), dtype=jnp.float32),
            seg=jnp.zeros((1, 1, 1), dtype=jnp.bool_),
            wall_map=jnp.zeros((1, 1, 1), dtype=jnp.float32),
            gdt_start=jnp.zeros((1, 1, 1), dtype=jnp.float32),
            gdt_end=jnp.zeros((1, 1, 1), dtype=jnp.float32),
            gt_path_vol=jnp.zeros((1, 1, 1), dtype=jnp.bool_),
            local_peaks=jnp.zeros((1, 3), dtype=jnp.int32),
            start_coord=jnp.zeros((3,), dtype=jnp.int32),
            end_coord=jnp.zeros((3,), dtype=jnp.int32),
            goal=jnp.zeros((3,), dtype=jnp.int32),
            image_shape=(1, 1, 1),
            seg_volume=0.0,
        )

    @partial(jax.jit, static_argnames=("self",))
    def reset_env(
        self, key: jax.Array, params: SmallBowelParams
    ) -> tuple[jnp.ndarray, SmallBowelState]:
        """Resets the environment."""
        key, key_choice, key_shuffle = jax.random.split(key, 3)

        # Determine start position and select appropriate GDT
        # rand = jax.random.randint(key_choice, (), 0, 10) # 40-40-20 distribution
        # Using a simpler random choice for now, can be expanded later

        # For simplicity, let's always start at start_coord and go to end_coord for now.
        # The original logic with random start/goal and local_peaks is more complex
        # and involves `random.choice` which is not directly JAX-compatible without
        # careful `jax.random` usage and `lax.cond` or `lax.switch`.
        # I will implement a simplified version first, then refine.

        # Simplified start/goal selection: always start at start_coord, go to end_coord
        current_pos_vox = params.start_coord
        goal = params.end_coord
        gdt_map = params.gdt_start

        # If local_peaks are available, randomly choose one as start/goal
        # This requires more complex JAX logic (e.g., lax.cond, lax.switch)
        # to replicate the 40-40-20 distribution and random choice.
        # For now, I'll use a simpler random choice from local_peaks if available,
        # otherwise default to start_coord.

        # Replicating the random start logic from TorchRL env:
        # 0-3: start_coord -> end_coord (gdt_start)
        # 4-5: random local peak -> (end_coord or start_coord) (gdt_start or gdt_end)
        # 6-9: end_coord -> start_coord (gdt_end)

        rand_choice = jax.random.randint(key_choice, (), 0, 10)

        # Case 1: Start at beginning, go to end
        current_pos_vox_case1 = params.start_coord
        goal_case1 = params.end_coord
        gdt_map_case1 = params.gdt_start

        # Case 2: Start at random local peak
        peak_idx = jax.random.randint(key_shuffle, (), 0, params.local_peaks.shape[0])
        current_pos_vox_case2 = params.local_peaks[peak_idx]

        # Randomly go in either direction for local peak start
        goal_case2_opt1 = params.end_coord
        gdt_map_case2_opt1 = params.gdt_start
        goal_case2_opt2 = params.start_coord
        gdt_map_case2_opt2 = params.gdt_end

        # Use key_shuffle for this binary choice
        peak_direction_choice = jax.random.bernoulli(key_shuffle)
        goal_case2 = jax.lax.select(
            peak_direction_choice, goal_case2_opt1, goal_case2_opt2
        )
        gdt_map_case2 = jax.lax.select(
            peak_direction_choice, gdt_map_case2_opt1, gdt_map_case2_opt2
        )

        # Case 3: Start at end, go to start
        current_pos_vox_case3 = params.end_coord
        goal_case3 = params.start_coord
        gdt_map_case3 = params.gdt_end

        # Select based on rand_choice
        current_pos_vox = jax.lax.switch(
            rand_choice,
            [
                lambda: current_pos_vox_case1,  # 0
                lambda: current_pos_vox_case1,  # 1
                lambda: current_pos_vox_case1,  # 2
                lambda: current_pos_vox_case1,  # 3
                lambda: current_pos_vox_case2,  # 4
                lambda: current_pos_vox_case2,  # 5
                lambda: current_pos_vox_case3,  # 6
                lambda: current_pos_vox_case3,  # 7
                lambda: current_pos_vox_case3,  # 8
                lambda: current_pos_vox_case3,  # 9
            ],
        )
        goal = jax.lax.switch(
            rand_choice,
            [
                lambda: goal_case1,
                lambda: goal_case1,
                lambda: goal_case1,
                lambda: goal_case1,
                lambda: goal_case2,
                lambda: goal_case2,
                lambda: goal_case3,
                lambda: goal_case3,
                lambda: goal_case3,
                lambda: goal_case3,
            ],
        )
        gdt_map = jax.lax.switch(
            rand_choice,
            [
                lambda: gdt_map_case1,
                lambda: gdt_map_case1,
                lambda: gdt_map_case1,
                lambda: gdt_map_case1,
                lambda: gdt_map_case2,
                lambda: gdt_map_case2,
                lambda: gdt_map_case3,
                lambda: gdt_map_case3,
                lambda: gdt_map_case3,
                lambda: gdt_map_case3,
            ],
        )

        # Validate start position (simplified check for now)
        # The original had a while loop, which is not JAX-compatible.
        # We need to use lax.cond or ensure valid inputs.
        # For now, assume valid start_coord from params.
        is_valid_start = (
            _is_valid_pos(current_pos_vox, params.image_shape)
            & params.seg[tuple(current_pos_vox)]
        )

        # If start is invalid, default to start_coord and gdt_start
        current_pos_vox = jax.lax.select(
            is_valid_start, current_pos_vox, params.start_coord
        )
        goal = jax.lax.select(is_valid_start, goal, params.end_coord)
        gdt_map = jax.lax.select(is_valid_start, gdt_map, params.gdt_start)

        # Initialize path tracking
        cumulative_path_mask = jnp.zeros(params.image_shape, dtype=jnp.bool_)

        # Mark initial position on cumulative path mask
        initial_line_mask = line_nd_jax(
            current_pos_vox, current_pos_vox, params.image_shape
        )
        cumulative_path_mask, new_gt_path_vol = draw_path_sphere_2_jax(
            cumulative_path_mask,
            initial_line_mask,
            params.cumulative_path_radius_vox,
            params.gt_path_vol,
        )

        # Initialize various tracking variables
        cum_reward = 0.0
        max_gdt_achieved = gdt_map[tuple(current_pos_vox)]

        # Initialize reward_map for peaks
        reward_map = jnp.zeros(params.image_shape, dtype=jnp.float32)
        # Scatter 1s at local peaks
        reward_map = reward_map.at[tuple(params.local_peaks.T)].set(1.0)

        wall_gradient = 0.0

        state = SmallBowelState(
            time=0,  # EnvState's time
            current_pos_vox=current_pos_vox,
            cumulative_path_mask=cumulative_path_mask,
            max_gdt_achieved=max_gdt_achieved,
            cum_reward=cum_reward,
            reward_map=reward_map,
            wall_gradient=wall_gradient,
            key=key,
        )

        obs = self.get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnames=("self",))
    def step_env(
        self,
        key: jax.Array,
        state: SmallBowelState,
        action: jnp.ndarray,
        params: SmallBowelParams,
    ) -> tuple[jnp.ndarray, SmallBowelState, jnp.ndarray, jnp.ndarray, dict]:
        """Performs a step transition in the environment."""
        key, key_reward = jax.random.split(key)

        # Extract Action: action is normalized [-1, 1]
        action_mapped = action * params.max_step_vox
        action_vox_delta = jnp.round(action_mapped).astype(jnp.int32)

        # Calculate next position
        next_pos_vox = state.current_pos_vox + action_vox_delta

        # Calculate reward and update masks
        reward, new_cumulative_path_mask, new_reward_map, wall_stuff = (
            self._calculate_reward(
                state.current_pos_vox,
                next_pos_vox,
                action_vox_delta,
                state.cumulative_path_mask,
                state.reward_map,
                params.gdt_start,  # Use gdt_start as the base GDT map for reward calculation
                params.seg,
                params.wall_map,
                params.gt_path_vol,
                params.cumulative_path_radius_vox,
                params.r_zero_mov,
                params.r_val1,
                params.r_val2,
                params.r_peaks,
                params.gdt_max_increase_theta,
                state.max_gdt_achieved,
                params.image_shape,
                params.seg_volume,
            )
        )

        # Update state variables
        new_cum_reward = state.cum_reward + reward
        new_wall_gradient = state.wall_gradient + wall_stuff

        # Update max_gdt_achieved based on the GDT map used for reward calculation
        new_max_gdt_achieved = jnp.maximum(
            state.max_gdt_achieved, params.gdt_start[tuple(next_pos_vox)]
        )

        # Check Termination Conditions
        done = self.is_terminal(
            state, params, next_pos_vox, action_vox_delta, wall_stuff
        )

        # Final Reward Adjustment if done
        final_coverage = jax.lax.select(
            done,
            self._get_final_coverage(
                new_cumulative_path_mask, params.seg, params.seg_volume
            ),
            0.0,
        )

        # Determine termination reason for final reward adjustment
        # This is simplified as JAX doesn't easily allow string reasons in jit.
        # We'll use boolean flags.
        reached_goal = (
            jnp.linalg.norm(next_pos_vox - params.goal)
            < params.cumulative_path_radius_vox
        )

        reward = jax.lax.select(
            done,
            reward
            + jax.lax.select(
                reached_goal,
                final_coverage * params.r_final,
                (final_coverage - 1) * params.r_final,
            ),
            reward,
        )

        # Update state
        state = SmallBowelState(
            time=state.time + 1,
            current_pos_vox=next_pos_vox,
            cumulative_path_mask=new_cumulative_path_mask,
            max_gdt_achieved=new_max_gdt_achieved,
            cum_reward=new_cum_reward,
            reward_map=new_reward_map,
            wall_gradient=new_wall_gradient,
            key=key,
        )

        obs = self.get_obs(state, params)

        # Info dictionary (simplified for JAX)
        info = {
            "final_coverage": final_coverage,
            "final_step_count": jax.lax.select(done, state.time, 0),
            "final_length": jax.lax.select(
                done,
                jnp.linalg.norm(state.current_pos_vox - state.current_pos_vox),
                0.0,
            ),  # Simplified length
            "final_wall_gradient": jax.lax.select(done, state.wall_gradient, 0.0),
            "total_reward": jax.lax.select(done, state.cum_reward, 0.0),
            "max_gdt_achieved": state.max_gdt_achieved,
        }

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnames=("self",))
    def _calculate_reward(
        self,
        current_pos_vox: jnp.ndarray,
        next_pos_vox: jnp.ndarray,
        action_vox_delta: jnp.ndarray,
        cumulative_path_mask: jnp.ndarray,
        reward_map: jnp.ndarray,
        gdt_map: jnp.ndarray,
        seg: jnp.ndarray,
        wall_map: jnp.ndarray,
        gt_path_vol: jnp.ndarray,
        cumulative_path_radius_vox: int,
        r_zero_mov: float,
        r_val1: float,
        r_val2: float,
        r_peaks: float,
        gdt_max_increase_theta: float,
        max_gdt_achieved: float,
        image_shape: tuple[int, int, int],
        seg_volume: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Calculate the reward for the current step (JAX-compatible)."""
        rt = jnp.array(0.0, dtype=jnp.float32)
        wall_stuff = jnp.array(0.0, dtype=jnp.float32)

        is_next_pos_valid = _is_valid_pos(next_pos_vox, image_shape)
        is_in_seg = jax.lax.select(is_next_pos_valid, seg[tuple(next_pos_vox)], False)

        # --- 1. Zero movement or goes out of the segmentation penalty ---
        # If not any(action_vox) or not self._is_valid_pos(next_pos_vox) or not self.seg_np[next_pos_vox]:
        cond_invalid_move = (
            (jnp.all(action_vox_delta == 0)) | (~is_next_pos_valid) | (~is_in_seg)
        )
        rt = jax.lax.select(cond_invalid_move, rt - r_zero_mov, rt)

        # Set of voxels S on the line segment
        line_mask = line_nd_jax(current_pos_vox, next_pos_vox, image_shape)

        # Only proceed with rewards if move is valid
        rt, new_cumulative_path_mask, new_reward_map, wall_stuff = jax.lax.cond(
            cond_invalid_move,
            lambda: (
                rt,
                cumulative_path_mask,
                reward_map,
                wall_stuff,
            ),  # If invalid, no further reward/mask updates
            lambda: self._calculate_valid_move_rewards(
                rt,
                line_mask,
                cumulative_path_mask,
                reward_map,
                gdt_map,
                seg,
                wall_map,
                gt_path_vol,
                cumulative_path_radius_vox,
                r_val1,
                r_val2,
                r_peaks,
                gdt_max_increase_theta,
                max_gdt_achieved,
                next_pos_vox,
                seg_volume,
            ),
        )
        return rt, new_cumulative_path_mask, new_reward_map, wall_stuff

    @partial(jax.jit, static_argnames=("self",))
    def _calculate_valid_move_rewards(
        self,
        rt: jnp.ndarray,
        line_mask: jnp.ndarray,
        cumulative_path_mask: jnp.ndarray,
        reward_map: jnp.ndarray,
        gdt_map: jnp.ndarray,
        seg: jnp.ndarray,
        wall_map: jnp.ndarray,
        gt_path_vol: jnp.ndarray,
        cumulative_path_radius_vox: int,
        r_val1: float,
        r_val2: float,
        r_peaks: float,
        gdt_max_increase_theta: float,
        max_gdt_achieved: float,
        next_pos_vox: jnp.ndarray,
        seg_volume: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Helper for reward calculation when the move is valid."""

        # --- 2. GDT-based reward ---
        next_gdt_val = gdt_map[tuple(next_pos_vox)]
        delta = next_gdt_val - max_gdt_achieved

        # Penalty if too large, reward if within margins
        rt = rt + jax.lax.select(
            delta > gdt_max_increase_theta,
            -r_val2,
            r_val2 * (delta / gdt_max_increase_theta),
        )

        # --- 2.5 Peaks-based reward ---
        peaks_reward_sum = jnp.sum(reward_map * line_mask)
        rt = rt + peaks_reward_sum * r_peaks

        # Discard the reward for visited nodes (set to 0 in reward_map)
        new_reward_map = reward_map.at[line_mask].set(0.0)

        # Update cumulative path mask and gt_path_vol
        new_cumulative_path_mask, new_gt_path_vol_updated = draw_path_sphere_2_jax(
            cumulative_path_mask, line_mask, cumulative_path_radius_vox, gt_path_vol
        )

        # Reward for coverage (based on Dice within the segmentation on the path)
        # The original used `self.gt_path_vol` which was modified in place.
        # Here, `new_gt_path_vol_updated` is the updated GT path volume.
        intersection = jnp.sum(new_gt_path_vol_updated * seg)
        union = seg_volume + jnp.sum(new_gt_path_vol_updated)
        coverage = jax.lax.select(union != 0, 2 * intersection / union, 0.0)
        rt = rt + coverage * r_val2

        # --- 3. Wall-based penalty ---
        wall_map_max = jnp.max(wall_map * line_mask)  # Max wall value along the line
        wall_stuff = wall_map_max
        rt = rt - r_val2 * wall_map_max * 30

        # --- 4. Revisiting penalty ---
        revisit_penalty = jnp.sum(
            new_cumulative_path_mask * line_mask
        )  # Check against the *new* mask
        rt = rt - r_val1 * jax.lax.select(
            revisit_penalty > 0, 1.0, 0.0
        )  # Apply penalty if any overlap

        # Penalty if next position is outside segmentation (already handled by cond_invalid_move, but double check)
        # This was `if not self.seg_np[next_pos_vox]: rt -= self.config.r_val1`
        # This is now covered by `cond_invalid_move` at the top level.
        # If we reach here, it means `is_in_seg` was True.

        return rt, new_cumulative_path_mask, new_reward_map, wall_stuff

    @partial(jax.jit, static_argnames=("self",))
    def get_obs(
        self, state: SmallBowelState, params: SmallBowelParams
    ) -> dict[str, jnp.ndarray]:
        """Get state patches centered at current position."""
        img_patch = get_patch_jax(
            params.image, state.current_pos_vox, params.patch_size_vox
        )
        wall_patch = get_patch_jax(
            params.wall_map, state.current_pos_vox, params.patch_size_vox
        )
        cum_path_patch = get_patch_jax(
            state.cumulative_path_mask.astype(
                jnp.float32
            ),  # Convert bool to float for stacking
            state.current_pos_vox,
            params.patch_size_vox,
        )

        # Stack patches for actor and critic
        actor_state = jnp.stack([img_patch, wall_patch, cum_path_patch], axis=0)
        critic_state = jnp.stack([img_patch, wall_patch, cum_path_patch], axis=0)

        # Add batch dimension (Gymnax expects observations without batch dim, but TorchRL had it)
        # For Gymnax, we return the raw observation. The agent will handle batching.
        return {"actor": actor_state, "critic": critic_state}

    @partial(jax.jit, static_argnames=("self",))
    def is_terminal(
        self,
        state: SmallBowelState,
        params: SmallBowelParams,
        next_pos_vox: jnp.ndarray,
        action_vox_delta: jnp.ndarray,
        wall_stuff: jnp.ndarray,
    ) -> jnp.ndarray:
        """Check whether state transition is terminal."""
        # Max steps reached
        max_steps_reached = state.time >= params.max_steps_in_episode

        # Out of bounds or invalid move (already checked in _calculate_reward, but re-check for termination)
        is_next_pos_valid = _is_valid_pos(next_pos_vox, params.image_shape)
        is_in_seg = jax.lax.select(
            is_next_pos_valid, params.seg[tuple(next_pos_vox)], False
        )

        invalid_move = (
            (jnp.all(action_vox_delta == 0))
            | (~is_next_pos_valid)
            | (~is_in_seg)
            | (wall_stuff > 0.03)
        )

        # Reached goal
        reached_goal = (
            jnp.linalg.norm(next_pos_vox - params.goal)
            < params.cumulative_path_radius_vox
        )

        return max_steps_reached | invalid_move | reached_goal

    @partial(jax.jit, static_argnames=("self",))
    def _get_final_coverage(
        self, cumulative_path_mask: jnp.ndarray, seg: jnp.ndarray, seg_volume: float
    ) -> jnp.ndarray:
        """Calculate coverage. Assumes tensors are valid."""
        intersection = jnp.sum(cumulative_path_mask * seg)
        union = seg_volume + jnp.sum(cumulative_path_mask)
        return jax.lax.select(union != 0, 2 * intersection / union, 0.0)

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3  # 3D movement delta

    def action_space(self, params: SmallBowelParams) -> spaces.Box:
        """Action space of the environment."""
        # Action is a 3D vector representing delta movement, normalized to [-1, 1]
        return spaces.Box(
            low=jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
            high=jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
            shape=(3,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: SmallBowelParams) -> spaces.dict:
        """Observation space of the environment."""
        # Observations are actor and critic patches, each 3 channels
        patch_shape = (3, *params.patch_size_vox)
        return spaces.dict({
            "actor": spaces.Box(
                low=-jnp.inf, high=jnp.inf, shape=patch_shape, dtype=jnp.float32
            ),
            "critic": spaces.Box(
                low=-jnp.inf, high=jnp.inf, shape=patch_shape, dtype=jnp.float32
            ),
        })

    def state_space(self, params: SmallBowelParams) -> spaces.dict:
        """State space of the environment."""
        return spaces.dict({
            "time": spaces.Discrete(params.max_steps_in_episode + 1),
            "current_pos_vox": spaces.Box(
                low=jnp.array([0, 0, 0], dtype=jnp.int32),
                high=jnp.array(params.image_shape, dtype=jnp.int32),
                shape=(3,),
                dtype=jnp.int32,
            ),
            "cumulative_path_mask": spaces.Box(
                low=0, high=1, shape=params.image_shape, dtype=jnp.bool_
            ),
            "max_gdt_achieved": spaces.Box(
                low=-jnp.inf, high=jnp.inf, shape=(), dtype=jnp.float32
            ),
            "cum_reward": spaces.Box(
                low=-jnp.inf, high=jnp.inf, shape=(), dtype=jnp.float32
            ),
            "reward_map": spaces.Box(
                low=0, high=1, shape=params.image_shape, dtype=jnp.float32
            ),
            "wall_gradient": spaces.Box(
                low=-jnp.inf, high=jnp.inf, shape=(), dtype=jnp.float32
            ),
            "key": spaces.Box(
                low=-jnp.inf, high=jnp.inf, shape=(2,), dtype=jnp.uint32
            ),  # JAX random key is a pair of uint32
        })
