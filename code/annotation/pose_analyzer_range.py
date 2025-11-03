import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from motion_rep_transfer import get_arm_shoulder_angles, get_motion_rep_with_arm_angles
from motion_classifier import MotionClassifier


def get_optimal_device() -> str:
    """
    Automatically detect and return the optimal device for computation.
    
    Returns:
        Device string: 'cuda' if CUDA is available, otherwise 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class BodyKeypointPositionAnalyzer:
    """
    Windowed 1D displacement analyzer with rich, human-readable summaries.
    Reconstructed from the provided screenshots.
    """

    def __init__(self) -> None:
        # Place to stash per-group custom thresholds, e.g. {"hands": Thresholds(...)}.
        self._group_thresholds: Dict[str, Thresholds] = {}

    # --------------------------- Utilities --------------------------------- #

    def register_group_thresholds(
        self, group_name: str, slight: float, moderate: float, significant: float
    ) -> None:
        self._group_thresholds[group_name] = Thresholds(slight, moderate, significant)

    def get_thresholds_for_group(self, group_name: Optional[str]) -> Thresholds:
        """
        Return thresholds for a group if registered; otherwise reasonable defaults.
        """
        if group_name and group_name in self._group_thresholds:
            return self._group_thresholds[group_name]
        return Thresholds()

    @staticmethod
    def _axis_terms(
        axis: int,
        is_world_space: bool,
        reference_vectors: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Tuple[str, str]:
        """
        Map axis -> (positive_term, negative_term) for language templates.

        For world space we use:
            x: right / left
            y: up   / down
            z: forward / backward

        For canonical/local space, we keep the same wording by default.
        The optional reference_vectors can be used to flip semantics if
        your coordinate system is inverted; pass unit vectors for the
        subject-centric right, up, forward axes; we will infer sign flips.
        """
        # Base terms
        if axis == 0:
            pos, neg = "right", "left"
        elif axis == 1:
            pos, neg = "up", "down"
        elif axis == 2:
            pos, neg = "forward", "backward"
        else:
            raise ValueError("axis must be 0, 1, or 2")

        if is_world_space and reference_vectors is not None:
            ref = reference_vectors[axis]
            # If your world axis is opposite the reference, swap the terms.
            # A negative mean implies a flipped alignment.
            if np.mean(ref) < 0:
                pos, neg = neg, pos

        return pos, neg

    # ------------------------- Core Analyzer -------------------------------- #

    def analyze_position_window(
        self,
        positions: np.ndarray,
        group_name: Optional[str] = None,
        positive_term: str = "forward",
        negative_term: str = "backward",
    ) -> Dict[str, Any]:
        """
        Analyze a 1D position sequence within a window.

        Args:
            positions: (T,) array of scalar positions along one axis
            group_name: optional group for custom thresholds
            positive_term / negative_term: wording for + / - displacement

        Returns: dict with keys
            movement_type: str
            movement_description: str
            has_oscillation: bool
            has_bidirectional: bool
            highest_position, lowest_position, first_position, last_position: float
            forward_count, backward_count, significant_movements: int
        """
        positions = np.asarray(positions).astype(float).flatten()
        if positions.size < 2:
            return {
                "movement_type": "insufficient",
                "movement_description": "insufficient data",
                "has_oscillation": False,
                "has_bidirectional": False,
                "highest_position": float(positions[0] if positions.size else 0.0),
                "lowest_position": float(positions[0] if positions.size else 0.0),
                "first_position": float(positions[0] if positions.size else 0.0),
                "last_position": float(positions[-1] if positions.size else 0.0),
                "forward_count": 0,
                "backward_count": 0,
                "significant_movements": 0,
            }

        th = self.get_thresholds_for_group(group_name)
        slight_threshold = th.slight
        moderate_threshold = th.moderate
        significant_threshold = th.significant

        first_position = positions[0]
        highest_position = np.max(positions)
        lowest_position = np.min(positions)
        last_position = positions[-1]

        # Frame-to-frame diffs and significance mask
        diffs = np.diff(positions)
        significant_mask = np.abs(diffs) >= slight_threshold

        if not np.any(significant_mask):
            forward_count = 0
            backward_count = 0
        else:
            sig_diffs = diffs[significant_mask]
            if sig_diffs.size == 0:
                forward_count = 0
                backward_count = 0
            else:
                signs = np.sign(sig_diffs).astype(int)
                # Direction change detection: 1 -> -1 or -1 -> 1
                sign_changes = np.diff(np.concatenate(([0], signs)))
                # Forward/backward change counts
                forward_count = int(np.sum((sign_changes == 1) & (signs > 0)))
                backward_count = int(np.sum((sign_changes == -1) & (signs < 0)))

        # Oscillation heuristic: at least one in each direction
        has_oscillation = (forward_count >= 1 and backward_count >= 1) or (
            backward_count >= 1 and forward_count >= 1
        )

        # Convenience lambdas for tiered adjectives
        def tier_word(delta: float) -> Optional[str]:
            if delta >= significant_threshold:
                return "significantly"
            if delta >= moderate_threshold:
                return "moderately"
            if delta >= slight_threshold:
                return "slightly"
            if delta >= slight_threshold * 0.5:
                return "very_slightly"
            return None

        # Helper to build phrase pieces
        def describe(delta: float, pos=True) -> Optional[str]:
            w = tier_word(delta)
            if w is None:
                return None
            return w

        # Bidirectional case selection
        # We examine the ordering patterns among (first, lowest, highest, last).
        def check_movement_count() -> bool:
            return (forward_count >= 1) or (backward_count >= 1)

        significant_deviation = (highest_position - lowest_position) >= slight_threshold

        # Case 1: lowest & highest in the middle (first < highest, last < highest; first > lowest, last > lowest)
        case_one_sequential = (
            (lowest_position < first_position < highest_position)
            and (lowest_position < last_position < highest_position)
        )
        case_one_bidirectional = (
            case_one_sequential and significant_deviation
        )

        def three_term_description(t1, t2, t3, order_pos_neg: Tuple[str, str, str]) -> str:
            a, b, c = order_pos_neg
            parts = []
            if t1 is not None:
                parts.append(f"{t1} {a}")
            if t2 is not None:
                parts.append(f"then {t2} {b}")
            if t3 is not None:
                parts.append(f"then {t3} {c}")
            return " ".join(parts) if parts else "minimal movement"

        def two_term_description(t1, t2, order_pos_neg: Tuple[str, str]) -> str:
            a, b = order_pos_neg
            if t1 is not None and t2 is not None:
                return f"{t1} {a} then {t2} {b}"
            if t1 is not None:
                return f"{t1} {a}"
            if t2 is not None:
                return f"{t2} {b}"
            return "minimal movement"

        movement_type = "static"
        movement_description = "minimal movement"

        if case_one_bidirectional:
            # Determine if forward-first or backward-first by who peaks first
            if np.argmax(positions) < np.argmin(positions):
                movement_type = "bidirectional_forward_first"
                # first -> highest, highest -> lowest, lowest -> last
                t1 = describe(abs(highest_position - first_position), pos=True)
                t2 = describe(abs(highest_position - lowest_position), pos=False)
                t3 = describe(abs(last_position - lowest_position), pos=True)
                movement_description = three_term_description(
                    t1, t2, t3, (positive_term, negative_term, positive_term)
                )
            else:
                movement_type = "bidirectional_backward_first"
                # first -> lowest, lowest -> highest, highest -> last
                t1 = describe(abs(first_position - lowest_position), pos=False)
                t2 = describe(abs(highest_position - lowest_position), pos=True)
                t3 = describe(abs(highest_position - last_position), pos=False)
                movement_description = three_term_description(
                    t1, t2, t3, (negative_term, positive_term, negative_term)
                )
        else:
            # Case 2: first is lowest, highest in the middle
            case_two_seq = (first_position == lowest_position) and (
                highest_position > last_position > first_position
                or highest_position > first_position < last_position
            )
            case_two_bidirectional = (
                case_two_seq and check_movement_count() and significant_deviation
            )

            if case_two_bidirectional:
                movement_type = "bidirectional_forward_first"
                t1 = describe(abs(highest_position - first_position), pos=True)
                t2 = describe(abs(highest_position - last_position), pos=False)
                movement_description = two_term_description(
                    t1, t2, (positive_term, negative_term)
                )
            else:
                # Case 3: first is highest, lowest in the middle
                case_three_seq = (first_position == highest_position) and (
                    lowest_position < last_position < first_position
                    or lowest_position < first_position > last_position
                )
                case_three_bidirectional = (
                    case_three_seq and check_movement_count() and significant_deviation
                )

                if case_three_bidirectional:
                    movement_type = "bidirectional_backward_first"
                    t1 = describe(abs(first_position - lowest_position), pos=False)
                    t2 = describe(abs(last_position - lowest_position), pos=True)
                    movement_description = two_term_description(
                        t1, t2, (negative_term, positive_term)
                    )
                else:
                    # Case 4: last is lowest, highest in the middle
                    case_four_seq = (last_position == lowest_position) and (
                        first_position < highest_position > last_position
                    )
                    case_four_bidirectional = (
                        case_four_seq and check_movement_count() and significant_deviation
                    )

                    if case_four_bidirectional:
                        movement_type = "bidirectional_forward_first"
                        t1 = describe(abs(highest_position - first_position), pos=True)
                        t2 = describe(abs(highest_position - last_position), pos=False)
                        movement_description = two_term_description(
                            t1, t2, (positive_term, negative_term)
                        )
                    else:
                        # Case 5: last is highest, lowest in the middle
                        case_five_seq = (last_position == highest_position) and (
                            first_position > lowest_position < last_position
                        )
                        case_five_bidirectional = (
                            case_five_seq
                            and check_movement_count()
                            and significant_deviation
                        )

                        if case_five_bidirectional:
                            movement_type = "bidirectional_backward_first"
                            t1 = describe(abs(first_position - lowest_position), pos=False)
                            t2 = describe(abs(last_position - lowest_position), pos=True)
                            movement_description = two_term_description(
                                t1, t2, (negative_term, positive_term)
                            )
                        else:
                            # No bidirectional pattern determined — fallback to net displacement
                            net_disp = last_position - first_position
                            if abs(net_disp) < slight_threshold:
                                movement_type = "static"
                                movement_description = "minimal movement"
                            elif net_disp > 0:
                                movement_type = "forward"
                                movement_description = positive_term
                            else:
                                movement_type = "backward"
                                movement_description = negative_term

        result = {
            "movement_type": movement_type,
            "movement_description": movement_description,
            "has_oscillation": bool(has_oscillation),
            "has_bidirectional": movement_type.startswith("bidirectional"),
            "highest_position": float(highest_position),
            "lowest_position": float(lowest_position),
            "first_position": float(first_position),
            "last_position": float(last_position),
            "forward_count": int(forward_count),
            "backward_count": int(backward_count),
            "significant_movements": int(forward_count + backward_count),
        }
        return result