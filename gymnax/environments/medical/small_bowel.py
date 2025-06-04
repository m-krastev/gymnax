"""Small Bowel Environment for JAX."""

from dataclasses import field
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


# Helper functions (JAX-compatible)
def _is_valid_pos(pos_vox: jnp.ndarray, shape: tuple[int, int, int]) -> jnp.ndarray:
    """Check if a position is within the volume bounds."""
    return jnp.all(pos_vox >= 0) & jnp.all(pos_vox < jnp.array(shape))


@partial(jax.jit, static_argnames=("patch_size",))
def get_patch_jax(
    volume: jnp.ndarray, center_coords: jnp.ndarray, patch_size: jnp.ndarray
) -> jnp.ndarray:
    """
    Extracts a patch from a 3D volume centered at center_coords.
    Handles padding for out-of-bounds regions.
    """
    patch = jnp.zeros_like(volume, shape=patch_size)

    # Calculate the slice indices for the patch
    start_coords = jnp.maximum(0, center_coords - jnp.asarray(patch_size) // 2)
    patch = jax.lax.dynamic_update_slice(
        patch, jax.lax.dynamic_slice(volume, start_coords, patch_size), start_coords
    )
    return patch


@partial(jax.jit, static_argnames=("max_npoints",))
def line_nd_jax(
    start: jnp.ndarray, stop: jnp.ndarray, max_npoints: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX-compatible function to get voxels along a 3D line segment using Bresenham's algorithm.

    NOTE: This function is JIT-able using a fixed `max_npoints` to ensure the output shape is static.

    # The minimum number of points along the segment is `int(ceil(max(abs(stop - start))))`

    Args:
        start: JAX array representing the start coordinates (e.g., shape (D,)).
        stop: JAX array representing the stop coordinates (e.g., shape (D,)).
        max_npoints: A static integer indicating the maximum possible number of points the line could have. This defines the fixed size of the output array.

    Returns:
        A tuple containing:
        - padded_coords: A JAX array of shape (D, max_npoints) with the line coordinates.
    """
    return jnp.round(
        jnp.linspace(start, stop, num=max_npoints, endpoint=False).T
    ).astype(int)

@partial(
    jax.jit,
)
def binary_dilation_3d_jax(mask: jnp.ndarray, radius: int) -> jnp.ndarray:
    """
    Performs 3D binary dilation on a boolean mask using a cubic structuring element.
    """
    # Dilation is defined as a convolution with the structuring element.
    # We can use a fori_loop to iterate over the structuring element

    kernel = jnp.array(
        [
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],  # 3x3x3 kernel for dilation
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        ],
        dtype=mask.dtype,
    )[:, :, :, None, None]

    mask = mask[None, ..., None]  # Add batch and channel dimensions
    # Describes batch, spatial, and feature dimensions of a convolution.
    # 1: input_kernel_channels, 2: input_spatial_dimensions, 3: output_spatial_dimensions
    dn = jax.lax.conv_dimension_numbers(
        mask.shape, kernel.shape, ("NHWDC", "HWDIO", "NHWDC")
    )
    return jax.lax.fori_loop(
        0,
        radius + 1,
        lambda i, acc: jax.lax.conv_general_dilated(
            acc,
            kernel,
            window_strides=(1, 1, 1),
            padding="SAME",
            dimension_numbers=dn,
        ),
        mask,
        # unroll=True,
    )[0, ..., 0]  # Remove batch and channel dimensions


def draw_path_sphere(
    cumulative_path_mask: jnp.ndarray,
    line: tuple[jnp.ndarray],
    dilation_radius: int,
    fill_value: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Draws a "sphere" around the path defined by `line` in the cumulative path mask.
    This function dilates the line voxels to create a spherical region around the path.

    Args:
        cumulative_path_mask: The cumulative path mask to update.
        line: A tuple containing the coordinates of the line voxels (e.g., from `line_nd_jax`).
        dilation_radius: The radius for dilation to create the spherical region.

    Returns:
        A tuple containing:
        - new_cumulative_path_mask: The updated cumulative path mask after adding the spherical region.
        - dilated_line_mask: The mask representing the dilated spherical region around the path.
    """
    # Dilate the line voxels to create a "sphere" around the path
    buffer = (
        jnp.zeros_like(cumulative_path_mask, dtype=jnp.bool_).at[*line].set(fill_value)
    )
    dilated_line_mask = binary_dilation_3d_jax(buffer, dilation_radius)

    # Update cumulative_path_mask: union of current mask and new dilated path
    new_cumulative_path_mask = cumulative_path_mask | dilated_line_mask

    return new_cumulative_path_mask, dilated_line_mask

@jax.jit
def draw_sphere_point(
    array_3d: jnp.ndarray, center_point: jnp.ndarray, radius: int, fill_value=1
):
    """
    Fills a 3D NumPy array with a specified value inside a sphere.

    Args:
        array_3d (np.ndarray): The 3D NumPy array (e.g., of zeros) to modify.
                                Its shape defines the coordinate space.
        center_point (tuple or list or np.ndarray): The (x, y, z) coordinates
                                                     of the sphere's center.
        radius (float or int): The radius of the sphere.
        fill_value (int or float, optional): The value to fill inside the sphere.
                                             Defaults to 1.

    Returns:
        np.ndarray: The modified 3D array with the sphere filled.
    """
    # Generate 1D coordinate arrays for each axis using np.ogrid
    # These will broadcast to the full 3D shape for the distance calculation
    # We take the actual shape of the input array for coordinates
    x, y, z = jnp.ogrid[
        0 : array_3d.shape[0], 0 : array_3d.shape[1], 0 : array_3d.shape[2]
    ]

    # Calculate the squared distance from the sphere's center for every point
    distance_squared = (
        (x - center_point[0]) ** 2
        + (y - center_point[1]) ** 2
        + (z - center_point[2]) ** 2
    )
    # Fill the array at the appropriate places using the boolean mask
    return jnp.where(distance_squared <= radius**2, fill_value, array_3d)


@jax.jit
def draw_sphere_line(array_3d: jnp.ndarray, pts: tuple, radius: int, fill_value=1):
    return jax.lax.scan(
        lambda a, p: (draw_sphere_point(a, p, radius, fill_value), p),
        array_3d,
        jnp.asarray(pts).T,
    )[0]


def draw_path_sphere(array_3d: jnp.ndarray, pts: tuple, radius: int, fill_value=True):
    """
    Draws a sphere around each point in the path defined by `pts` in the 3D array.

    Args:
        array_3d: The 3D array to update.
        pts: A tuple containing the coordinates of the path points (e.g., from `line_nd_jax`).
        radius: The radius for the spheres to be drawn around each point.
        fill_value: The value to fill inside the spheres.
    """
    # Draw spheres around each point in the path
    new_array_3d = draw_sphere_line(jnp.zeros_like(array_3d), pts, radius, fill_value)
    # Update the original array with the new spheres
    updated_array_3d = jnp.maximum(array_3d, new_array_3d)
    return updated_array_3d, new_array_3d

@struct.dataclass
class SmallBowelState(environment.EnvState):
    """State of the Small Bowel environment."""

    current_pos_vox: jnp.ndarray  # (3,) int
    cumulative_path_mask: jnp.ndarray  # (D, H, W) bool
    max_gdt_achieved: float
    cum_reward: float
    reward_map: jnp.ndarray  # (D, H, W) float, for peaks
    wall_gradient: float
    length: float


@struct.dataclass(kw_only=True)
class SmallBowelParams(environment.EnvParams):
    """Parameters for the Small Bowel environment."""

    image: jnp.ndarray  # (D, H, W) float
    seg: jnp.ndarray  # (D, H, W) bool
    wall_map: jnp.ndarray  # (D, H, W) float
    gt_path_vol: jnp.ndarray  # (D, H, W) bool
    gdt_start: jnp.ndarray  # (D, H, W) float
    gdt_end: jnp.ndarray  # (D, H, W) float
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
    patch_size_vox: jnp.ndarray = struct.field(
        pytree_node=False,
        default=(32, 32, 32),  # Default patch size in voxels
    )
    cumulative_path_radius_vox: int = 3
    seg_volume: float = 0.0  # Will be calculated from seg
    image_shape: jnp.ndarray = field(
        default_factory=lambda: jnp.array([1, 1, 1], dtype=jnp.int32)
    )


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
            image_shape=jnp.ones((3,), dtype=jnp.int32),  # Placeholder
            seg_volume=0.0,
        )

    def reset_env(
        self, key: jax.Array, params: SmallBowelParams
    ) -> tuple[jnp.ndarray, SmallBowelState]:
        """Resets the environment."""
        key, key_choice, key_shuffle = jax.random.split(key, 3)
        # Random start logic:
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
        # print(params.image_shape)
        cumulative_path_mask = jnp.zeros_like(params.image, dtype=jnp.bool_)

        # Mark initial position on cumulative path mask
        line = line_nd_jax(current_pos_vox, current_pos_vox, 256)
        cumulative_path_mask, _ = draw_path_sphere(
            cumulative_path_mask, line, params.cumulative_path_radius_vox, True
        )

        # Initialize various tracking variables
        cum_reward = 0.0
        max_gdt_achieved = gdt_map[tuple(current_pos_vox)]

        # Initialize reward_map for peaks
        reward_map = jnp.zeros_like(params.image, dtype=jnp.float32)
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
            length=0.0,
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

        # Extract Action: action is normalized [0, 1]
        action_mapped = (2 * action - 1) * params.max_step_vox
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
            length=state.length + jnp.linalg.norm(next_pos_vox - state.current_pos_vox),
        )

        obs = self.get_obs(state, params)

        # Info dictionary (simplified for JAX)
        info = {
            "final_coverage": final_coverage,
            "final_step_count": jax.lax.select(done, state.time, 0),
            "final_length": jax.lax.select(
                done,
                state.length,
                0.0,
            ),
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
        # is_in_seg = jnp.where(is_next_pos_valid, seg[tuple(next_pos_vox)], False)

        # --- 1. Zero movement or goes out of the segmentation penalty ---
        cond_invalid_move = (
            (jnp.all(action_vox_delta == 0)) | (~is_next_pos_valid)  # | (~is_in_seg)
        )
        rt = jax.lax.select(cond_invalid_move, rt - r_zero_mov, rt)

        # Set of voxels S on the line segment
        line = line_nd_jax(current_pos_vox, next_pos_vox, 256)

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
                line,
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
        line: tuple[jnp.ndarray],
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
        rt += jax.lax.select(
            delta > 0,
            jax.lax.select(
                delta > gdt_max_increase_theta,
                -r_val2,
                r_val2 * (delta / gdt_max_increase_theta),
            ),
            0.0,
        )

        # --- 2.5 Peaks-based reward ---
        peaks_reward_sum = jnp.sum(
            reward_map.at[line[0], line[1], line[2]].get(indices_are_sorted=True).mean()
        )
        rt += peaks_reward_sum * r_peaks

        # Discard the reward for visited nodes (set to 0 in reward_map)
        new_reward_map = reward_map.at[line[0], line[1], line[2]].set(0.0)

        # Update cumulative path mask
        new_cumulative_path_mask, coverage = draw_path_sphere(
            cumulative_path_mask, line, cumulative_path_radius_vox, True
        )

        # Reward for coverage (based on Dice within the segmentation on the path)
        intersection = jnp.sum(coverage * seg)
        union = seg_volume + jnp.sum(coverage)
        coverage = jax.lax.select(union != 0, 2 * intersection / union, 0.0)
        rt = rt + coverage * r_val2

        # --- 3. Wall-based penalty ---
        wall_map_max = wall_map.at[line[0], line[1], line[2]].get().max()
        wall_stuff = wall_map_max
        rt = rt - r_val2 * wall_map_max * 30

        # --- 4. Revisiting penalty ---
        revisit_penalty = (
            new_cumulative_path_mask.at[line[0], line[1], line[2]].get().any()
        )  # Check against the *new* mask
        rt = rt - r_val1 * jax.lax.select(
            revisit_penalty > 0, 1.0, 0.0
        )  # Apply penalty if any overlap

        # Penalty if next position is outside segmentation
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
            state.cumulative_path_mask,
            state.current_pos_vox,
            params.patch_size_vox,
        )

        # Stack patches for actor and critic
        actor_state = jnp.stack([img_patch, wall_patch, cum_path_patch], axis=0)
        critic_state = jnp.stack([img_patch, wall_patch, cum_path_patch], axis=0)

        # For Gymnax, we return the raw observation. The agent will handle batching.
        return jnp.stack([actor_state, critic_state], axis=0)

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
        # is_in_seg = jnp.where(
        #     is_next_pos_valid, params.seg[tuple(next_pos_vox)] > 0, False
        # )

        invalid_move = (
            (jnp.all(action_vox_delta == 0))
            | (~is_next_pos_valid)
            | (wall_stuff > 0.03)
            # | (~is_in_seg)
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
        return spaces.Box(
            low=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
            high=jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
            shape=(3,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: SmallBowelParams) -> spaces.Dict:
        """Observation space of the environment."""
        # Observations are actor and critic patches, each 3 channels
        patch_shape = (3, *params.patch_size_vox)
        return spaces.Dict({
            "actor": spaces.Box(
                low=-jnp.inf, high=jnp.inf, shape=patch_shape, dtype=jnp.float32
            ),
            "critic": spaces.Box(
                low=-jnp.inf, high=jnp.inf, shape=patch_shape, dtype=jnp.float32
            ),
        })

    def state_space(self, params: SmallBowelParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
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
        })
