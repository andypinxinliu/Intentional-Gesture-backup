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


class PoseAnalyzer:
    """
    A class to analyze pose data, calculate arm/shoulder parameters,
    compute cumulative pose angle changes, and analyze 3D positions.
    """
    
    def __init__(self, fps: int = 30, window_size: int = 30):
        """
        Initialize the pose analyzer.
        
        Args:
            fps: Frames per second (default: 30)
            window_size: Number of frames to process in each window (default: 30)
        """
        self.fps = fps
        self.window_size = window_size
        
        # Arm and shoulder joint indices in SMPL-X (0-based)
        self.arm_joint_indices = [16, 17, 18, 19, 20, 21]
        self.arm_joint_names = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist"
        ]
        
        # Finger joint definitions for SMPL-X (0-based indices)
        # Each finger has 3 joints: MCP (knuckle), PIP (middle), DIP (tip)
        self.finger_joints = {
            'left_hand': {
                'thumb': [22, 23, 24],      # Left thumb joints
                'index': [25, 26, 27],      # Left index finger joints
                'middle': [28, 29, 30],     # Left middle finger joints
                'ring': [31, 32, 33],       # Left ring finger joints
                'pinky': [34, 35, 36]       # Left pinky finger joints
            },
            'right_hand': {
                'thumb': [37, 38, 39],      # Right thumb joints
                'index': [40, 41, 42],      # Right index finger joints
                'middle': [43, 44, 45],     # Right middle finger joints
                'ring': [46, 47, 48],       # Right ring finger joints
                'pinky': [49, 50, 51]       # Right pinky finger joints
            }
        }
        
        # Finger pose angle ranges in SMPL-X (0-based indices)
        # Each finger has 3 joints with 3 angles each (9 angles per finger)
        self.finger_pose_ranges = {
            'left_hand': {
                'thumb': [66, 67, 68, 69, 70, 71, 72, 73, 74],      # Left thumb angles
                'index': [75, 76, 77, 78, 79, 80, 81, 82, 83],      # Left index angles
                'middle': [84, 85, 86, 87, 88, 89, 90, 91, 92],     # Left middle angles
                'ring': [93, 94, 95, 96, 97, 98, 99, 100, 101],     # Left ring angles
                'pinky': [102, 103, 104, 105, 106, 107, 108, 109, 110]  # Left pinky angles
            },
            'right_hand': {
                'thumb': [111, 112, 113, 114, 115, 116, 117, 118, 119],  # Right thumb angles
                'index': [120, 121, 122, 123, 124, 125, 126, 127, 128],  # Right index angles
                'middle': [129, 130, 131, 132, 133, 134, 135, 136, 137], # Right middle angles
                'ring': [138, 139, 140, 141, 142, 143, 144, 145, 146],   # Right ring angles
                'pinky': [147, 148, 149, 150, 151, 152, 153, 154, 155]   # Right pinky angles
            }
        }
        
        # Finger grouping for analysis
        # Thumb and index are analyzed separately due to higher dynamics
        # Other fingers (middle, ring, pinky) are grouped together
        self.finger_groups = {
            'left_hand': {
                'individual': ['thumb', 'index'],
                'grouped': ['middle', 'ring', 'pinky']
            },
            'right_hand': {
                'individual': ['thumb', 'index'],
                'grouped': ['middle', 'ring', 'pinky']
            }
        }
        
        # Finger curvature thresholds (in radians)
        self.finger_thresholds = {
            'curved_threshold': 0.3,    # Threshold for considering finger curved
            'straight_threshold': 0.1,  # Threshold for considering finger straight
            'change_threshold': 0.2     # Threshold for detecting significant change
        }
        
        # Initialize motion classifier
        self.motion_classifier = MotionClassifier()
    
    def get_arm_parameters(self, poses: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract arm and shoulder parameters from pose data.
        
        Args:
            poses: Pose data of shape (T, 165)
            
        Returns:
            Dictionary containing arm parameters for each joint
        """
        return get_arm_shoulder_angles(poses)
    
    def get_finger_angles(self, poses: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract finger angles from pose data.
        
        Args:
            poses: Pose data of shape (T, 165)
            
        Returns:
            Dictionary containing finger angles for each hand and finger
        """
        T = poses.shape[0]
        finger_angles = {}
        
        for hand in ['left_hand', 'right_hand']:
            finger_angles[hand] = {}
            for finger_name, angle_indices in self.finger_pose_ranges[hand].items():
                # Extract angles for this finger (9 angles per finger)
                finger_angles[hand][finger_name] = poses[:, angle_indices]
        
        return finger_angles
    
    def _rotvec_magnitude_per_joint(self, finger_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        finger_angles: (T, 9) = [MCP(0:3), PIP(3:6), DIP(6:9)] as axis-angle (radians).
        Returns: (T,), (T,), (T,) magnitudes for MCP, PIP, DIP.
        """
        assert finger_angles.shape[1] == 9, "Expected (T, 9) for a single finger"
        rot = finger_angles.reshape(-1, 3, 3)  # (T, joints=3, comps=3)
        mags = np.linalg.norm(rot, axis=2)     # (T, 3)
        mcp_mag, pip_mag, dip_mag = mags[:, 0], mags[:, 1], mags[:, 2]
        return mcp_mag, pip_mag, dip_mag

    
    def analyze_finger_curvature(self, finger_angles: np.ndarray) -> Dict[str, float]:
        """
        Curvature ≈ flexion amount. Use geodesic angle ||rotvec|| for PIP & DIP,
        which dominate curl; optionally include MCP with small weight.
        """
        mcp_mag, pip_mag, dip_mag = self._rotvec_magnitude_per_joint(finger_angles)

        # Primary curl from PIP & DIP; MCP can abduct/adduct, so give it less weight.
        # You can tune these weights if desired.
        curvature_series = 0.1 * mcp_mag + 0.45 * pip_mag + 0.45 * dip_mag  # (T,)

        # Optional: light smoothing to reduce frame jitter
        # curvature_series = np.convolve(curvature_series, np.ones(3)/3, mode="same")

        start_curv = float(curvature_series[0])
        end_curv   = float(curvature_series[-1])
        delta      = end_curv - start_curv
        avg_curv   = float(np.mean(curvature_series))

        # Thresholds are in radians (axis-angle). Tune for your data.
        curved_th   = self.finger_thresholds.get('curved_threshold', 0.6)    # ~34°
        straight_th = self.finger_thresholds.get('straight_threshold', 0.15) # ~9°
        change_th   = self.finger_thresholds.get('change_threshold', 0.2)    # ~11°

        if end_curv > curved_th:
            state = "curved"
        elif end_curv < straight_th:
            state = "straight"
        else:
            state = "slightly_curved"

        if abs(delta) < change_th:
            change = "no_change"
        elif delta > 0:
            change = "more_curved"
        else:
            change = "more_straight"
        
        return {
            "curvature_state": state,
            "change_description": change,
            "start_curvature": start_curv,
            "end_curvature": end_curv,
            "curvature_change": delta,
            "avg_curvature": avg_curv,
        }
    
    def analyze_grouped_fingers(self, finger_angles: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute curvature per finger correctly, then average the series.
        """
        per_series = []
        for _, angles in finger_angles.items():
            mcp_mag, pip_mag, dip_mag = self._rotvec_magnitude_per_joint(angles)
            series = 0.1 * mcp_mag + 0.45 * pip_mag + 0.45 * dip_mag
            per_series.append(series)

        # Align (just in case) and average
        minT = min(s.shape[0] for s in per_series)
        per_series = [s[:minT] for s in per_series]
        group_series = np.mean(np.stack(per_series, axis=0), axis=0)  # (T,)

        start_curv = float(group_series[0])
        end_curv   = float(group_series[-1])
        delta      = end_curv - start_curv
        avg_curv   = float(np.mean(group_series))

        curved_th   = self.finger_thresholds.get('curved_threshold', 0.6)
        straight_th = self.finger_thresholds.get('straight_threshold', 0.15)
        change_th   = self.finger_thresholds.get('change_threshold', 0.2)

        if end_curv > curved_th:
            state = "curved"
        elif end_curv < straight_th:
            state = "straight"
        else:
            state = "slightly_curved"

        if abs(delta) < change_th:
            change = "no_change"
        elif delta > 0:
            change = "more_curved"
        else:
            change = "more_straightened"

        return {
            "curvature_state": state,
            "change_description": change,
            "start_curvature": start_curv,
            "end_curvature": end_curv,
            "curvature_change": delta,
            "avg_curvature": avg_curv,
            "finger_count": len(finger_angles),
            "finger_names": list(finger_angles.keys()),
        }
    
    def analyze_all_fingers(self, poses: np.ndarray) -> Dict[str, Dict[str, Dict]]:
        """
        Analyze all fingers for both hands with grouped analysis.
        Thumb and index are analyzed individually, others are grouped.
        
        Args:
            poses: Pose data of shape (T, 165)
            
        Returns:
            Dictionary containing finger analysis for both hands
        """
        finger_angles = self.get_finger_angles(poses)
        finger_analysis = {}
        
        for hand in ['left_hand', 'right_hand']:
            finger_analysis[hand] = {}
            
            # Analyze individual fingers (thumb and index)
            for finger_name in self.finger_groups[hand]['individual']:
                if finger_name in finger_angles[hand]:
                    angles = finger_angles[hand][finger_name]
                    finger_analysis[hand][finger_name] = self.analyze_finger_curvature(angles)
            
            # Analyze grouped fingers (middle, ring, pinky)
            grouped_fingers = {}
            for finger_name in self.finger_groups[hand]['grouped']:
                if finger_name in finger_angles[hand]:
                    grouped_fingers[finger_name] = finger_angles[hand][finger_name]
            
            if grouped_fingers:
                finger_analysis[hand]['other_fingers'] = self.analyze_grouped_fingers(grouped_fingers)
        
        return finger_analysis
    
    def get_finger_summary(self, finger_analysis: Dict) -> Dict[str, List[str]]:
        """
        Get a simple summary of finger states and changes.
        Thumb and index are shown individually, others are grouped.
        
        Args:
            finger_analysis: Finger analysis from analyze_all_fingers
            
        Returns:
            Dictionary containing simple finger descriptions
        """
        summary = {
            'left_hand': [],
            'right_hand': []
        }
        
        for hand in ['left_hand', 'right_hand']:
            # Add individual finger descriptions (thumb and index)
            for finger_name in self.finger_groups[hand]['individual']:
                if finger_name in finger_analysis[hand]:
                    analysis = finger_analysis[hand][finger_name]
                    state = analysis['curvature_state']
                    change = analysis['change_description']
                    
                    if change == "no_change":
                        description = f"{finger_name}: {state}"
                    else:
                        description = f"{finger_name}: {change} ({state})"
                    
                    summary[hand].append(description)
            
            # Add grouped finger description (middle, ring, pinky)
            if 'other_fingers' in finger_analysis[hand]:
                analysis = finger_analysis[hand]['other_fingers']
                state = analysis['curvature_state']
                change = analysis['change_description']
                finger_names = analysis['finger_names']
                
                if change == "no_change":
                    description = f"other fingers ({', '.join(finger_names)}): {state}"
                else:
                    description = f"other fingers ({', '.join(finger_names)}): {change} ({state})"
                
                summary[hand].append(description)
        
        return summary
    
    def calculate_cumulative_angle_change(self, angles: np.ndarray, window_size: int = None) -> np.ndarray:
        """
        Calculate direct angle change between first and last frame of each window.
        
        Args:
            angles: Angle data of shape (T, 3) for a single joint
            window_size: Number of frames to consider (default: self.window_size)
            
        Returns:
            Direct angle change for each window (last_frame - first_frame)
        """
        if window_size is None:
            window_size = self.window_size
            
        T = angles.shape[0]
        direct_changes = []
        
        for start_frame in range(0, T - window_size + 1, window_size):
            end_frame = start_frame + window_size
            window_angles = angles[start_frame:end_frame]
            
            # Calculate direct change as difference between last frame and first frame
            # This preserves the sign/direction of movement
            first_frame = window_angles[0]  # First frame in window
            last_frame = window_angles[-1]  # Last frame in window
            direct_change = last_frame - first_frame  # Direct difference preserving sign
            
            direct_changes.append(direct_change)
            
        return np.array(direct_changes)
    
    def calculate_hand_position_movement(self, joints: np.ndarray, hand_type: str = "both") -> Dict[str, np.ndarray]:
        """
        Calculate hand position movement over a window of frames using wrist joints.
        
        Args:
            joints: Joint positions of shape (T, J, 3) where J is number of joints
            hand_type: Which hand to analyze ("left", "right", or "both")
            
        Returns:
            Dictionary containing movement metrics for each hand based on wrist joint
        """
        T, J, _ = joints.shape
        
        # SMPL-X wrist joint indices - single joint representing each hand
        left_wrist_idx = 20   # Left wrist joint
        right_wrist_idx = 21  # Right wrist joint
        
        results = {}
        
        if hand_type in ["left", "both"]:
            left_wrist_positions = joints[:, left_wrist_idx, :]  # (T, 3)
            
            # Calculate movement metrics for left hand using wrist joint
            left_movement = self._calculate_position_movement(left_wrist_positions)
            results['left_hand'] = left_movement
        
        if hand_type in ["right", "both"]:
            right_wrist_positions = joints[:, right_wrist_idx, :]  # (T, 3)
            
            # Calculate movement metrics for right hand using wrist joint
            right_movement = self._calculate_position_movement(right_wrist_positions)
            results['right_hand'] = right_movement
        
        return results
    
    def _calculate_position_movement(self, positions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate various movement metrics for a sequence of 3D positions.
        
        Args:
            positions: Position data of shape (T, 3)
            
        Returns:
            Dictionary containing movement metrics
        """
        T = positions.shape[0]
        
        # Calculate frame-to-frame differences
        position_diffs = np.diff(positions, axis=0)  # (T-1, 3)
        
        # Calculate movement along each axis
        x_movement = position_diffs[:, 0]  # X-axis movement
        y_movement = position_diffs[:, 1]  # Y-axis movement  
        z_movement = position_diffs[:, 2]  # Z-axis movement
        
        # Calculate total displacement (Euclidean distance)
        total_displacement = np.linalg.norm(position_diffs, axis=1)  # (T-1,)
        
        # Calculate cumulative movement metrics
        cumulative_x = np.cumsum(np.abs(x_movement))
        cumulative_y = np.cumsum(np.abs(y_movement))
        cumulative_z = np.cumsum(np.abs(z_movement))
        cumulative_total = np.cumsum(total_displacement)
        
        # Calculate direct movement metrics (last frame - first frame)
        # This preserves the sign/direction of movement
        direct_x = positions[-1, 0] - positions[0, 0]  # Direct X movement
        direct_y = positions[-1, 1] - positions[0, 1]  # Direct Y movement
        direct_z = positions[-1, 2] - positions[0, 2]  # Direct Z movement
        
        # Calculate range of movement (max - min) for each axis
        x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
        y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])
        z_range = np.max(positions[:, 2]) - np.min(positions[:, 2])
        
        # Calculate average velocity (displacement per frame)
        avg_velocity = np.mean(total_displacement) if T > 1 else 0.0
        
        # Calculate peak velocity
        peak_velocity = np.max(total_displacement) if T > 1 else 0.0
        
        # Calculate relative position to starting position
        relative_positions = positions - positions[0]  # (T, 3)
        
        # Calculate final displacement from start
        final_displacement = np.linalg.norm(positions[-1] - positions[0])
        
        return {
            'positions': positions,  # Original positions (T, 3)
            'relative_positions': relative_positions,  # Positions relative to start (T, 3)
            'frame_differences': position_diffs,  # Frame-to-frame differences (T-1, 3)
            'x_movement': x_movement,  # X-axis movement (T-1,)
            'y_movement': y_movement,  # Y-axis movement (T-1,)
            'z_movement': z_movement,  # Z-axis movement (T-1,)
            'total_displacement': total_displacement,  # Total displacement per frame (T-1,)
            'cumulative_x': cumulative_x,  # Cumulative X movement (T-1,)
            'cumulative_y': cumulative_y,  # Cumulative Y movement (T-1,)
            'cumulative_z': cumulative_z,  # Cumulative Z movement (T-1,)
            'cumulative_total': cumulative_total,  # Cumulative total movement (T-1,)
            'direct_x': direct_x,  # Direct X movement (last - first frame)
            'direct_y': direct_y,  # Direct Y movement (last - first frame)
            'direct_z': direct_z,  # Direct Z movement (last - first frame)
            'x_range': x_range,  # Range of X movement (scalar)
            'y_range': y_range,  # Range of Y movement (scalar)
            'z_range': z_range,  # Range of Z movement (scalar)
            'avg_velocity': avg_velocity,  # Average velocity (scalar)
            'peak_velocity': peak_velocity,  # Peak velocity (scalar)
            'final_displacement': final_displacement,  # Final displacement from start (scalar)
            'start_position': positions[0],  # Starting position (3,)
            'end_position': positions[-1],  # Ending position (3,)
        }
    
    def calculate_hand_relative_positions(self, joints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate relative positions between left and right hands using wrist joints.
        
        Args:
            joints: Joint positions of shape (T, J, 3) where J is number of joints
            
        Returns:
            Dictionary containing relative position metrics based on wrist joints
        """
        T, J, _ = joints.shape
        
        # SMPL-X wrist joint indices - single joint representing each hand
        left_wrist_idx = 20   # Left wrist joint
        right_wrist_idx = 21  # Right wrist joint
        
        left_wrist_positions = joints[:, left_wrist_idx, :]  # (T, 3)
        right_wrist_positions = joints[:, right_wrist_idx, :]  # (T, 3)
        
        # Calculate relative position (right hand relative to left hand)
        relative_positions = right_wrist_positions - left_wrist_positions  # (T, 3)
        
        # Calculate distance between hands using wrist joints
        hand_distances = np.linalg.norm(relative_positions, axis=1)  # (T,)
        
        # Calculate relative movement (how the relative position changes over time)
        relative_movement = np.diff(relative_positions, axis=0)  # (T-1, 3)
        
        # Calculate relative movement along each axis
        relative_x_movement = relative_movement[:, 0]  # X-axis relative movement
        relative_y_movement = relative_movement[:, 1]  # Y-axis relative movement
        relative_z_movement = relative_movement[:, 2]  # Z-axis relative movement
        
        # Calculate cumulative relative movement
        cumulative_relative_x = np.cumsum(np.abs(relative_x_movement))
        cumulative_relative_y = np.cumsum(np.abs(relative_y_movement))
        cumulative_relative_z = np.cumsum(np.abs(relative_z_movement))
        
        # Calculate direct relative movement (last frame - first frame)
        # This preserves the sign/direction of relative movement
        direct_relative_x = relative_positions[-1, 0] - relative_positions[0, 0]  # Direct X relative movement
        direct_relative_y = relative_positions[-1, 1] - relative_positions[0, 1]  # Direct Y relative movement
        direct_relative_z = relative_positions[-1, 2] - relative_positions[0, 2]  # Direct Z relative movement
        
        # Calculate range of relative positions
        relative_x_range = np.max(relative_positions[:, 0]) - np.min(relative_positions[:, 0])
        relative_y_range = np.max(relative_positions[:, 1]) - np.min(relative_positions[:, 1])
        relative_z_range = np.max(relative_positions[:, 2]) - np.min(relative_positions[:, 2])
        
        # Calculate average and peak hand distance
        avg_hand_distance = np.mean(hand_distances)
        peak_hand_distance = np.max(hand_distances)
        min_hand_distance = np.min(hand_distances)
        
        return {
            'relative_positions': relative_positions,  # Right hand relative to left (T, 3)
            'hand_distances': hand_distances,  # Distance between hands (T,)
            'relative_movement': relative_movement,  # Relative movement (T-1, 3)
            'relative_x_movement': relative_x_movement,  # X-axis relative movement (T-1,)
            'relative_y_movement': relative_y_movement,  # Y-axis relative movement (T-1,)
            'relative_z_movement': relative_z_movement,  # Z-axis relative movement (T-1,)
            'cumulative_relative_x': cumulative_relative_x,  # Cumulative X relative movement (T-1,)
            'cumulative_relative_y': cumulative_relative_y,  # Cumulative Y relative movement (T-1,)
            'cumulative_relative_z': cumulative_relative_z,  # Cumulative Z relative movement (T-1,)
            'direct_relative_x': direct_relative_x,  # Direct X relative movement (last - first frame)
            'direct_relative_y': direct_relative_y,  # Direct Y relative movement (last - first frame)
            'direct_relative_z': direct_relative_z,  # Direct Z relative movement (last - first frame)
            'relative_x_range': relative_x_range,  # Range of X relative position (scalar)
            'relative_y_range': relative_y_range,  # Range of Y relative position (scalar)
            'relative_z_range': relative_z_range,  # Range of Z relative position (scalar)
            'avg_hand_distance': avg_hand_distance,  # Average distance between hands (scalar)
            'peak_hand_distance': peak_hand_distance,  # Maximum distance between hands (scalar)
            'min_hand_distance': min_hand_distance,  # Minimum distance between hands (scalar)
            'left_wrist_positions': left_wrist_positions,  # Left wrist positions (T, 3)
            'right_wrist_positions': right_wrist_positions,  # Right wrist positions (T, 3)
        }
    
    def generate_joint_positions(self, poses: np.ndarray, trans: np.ndarray, expressions: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate joint positions using SMPL-X.
        
        Args:
            poses: Pose data (T, 165)
            trans: Translation data (T, 3)
            expressions: Expression data (T, 100)
            
        Returns:
            Joint positions array of shape (T, J, 3) or None if failed
        """
        try:
            from motion_rep_transfer import smplx_model
            
            # Use optimal device for computation
            device = torch.device(get_optimal_device())
            smplx_model_device = smplx_model.to(device)
            
            T = poses.shape[0]
            joints_list = []
            
            for i in range(T):
                # Prepare input for SMPL-X
                pose_tensor = torch.from_numpy(poses[i:i+1]).float().to(device)
                trans_tensor = torch.from_numpy(trans[i:i+1]).float().to(device)
                betas_tensor = torch.zeros(1, 300).float().to(device)  # Neutral shape
                expressions_tensor = torch.from_numpy(expressions[i:i+1]).float().to(device)
                
                # Forward pass through SMPL-X
                with torch.no_grad():
                    output = smplx_model_device(
                        betas=betas_tensor,
                        transl=trans_tensor,
                        expression=expressions_tensor,
                        jaw_pose=pose_tensor[:, 22*3:23*3],
                        global_orient=pose_tensor[:, :3],
                        body_pose=pose_tensor[:, 3:21*3+3],
                        left_hand_pose=pose_tensor[:, 25*3:40*3],
                        right_hand_pose=pose_tensor[:, 40*3:55*3],
                        leye_pose=pose_tensor[:, 69:72],
                        reye_pose=pose_tensor[:, 72:75],
                        return_joints=True
                    )
                
                joints = output['joints'][0].cpu().numpy()  # (J, 3)
                joints_list.append(joints)
            
            # Stack all joint positions
            window_joints = np.stack(joints_list, axis=0)  # (T, J, 3)
            return window_joints
            
        except Exception as e:
            print(f"Error generating joint positions: {e}")
            return None
    
    def analyze_window(self, poses: np.ndarray, trans: np.ndarray, expressions: np.ndarray) -> Dict:
        """
        Analyze a window of pose data and return comprehensive analysis.
        
        Args:
            poses: Pose data for the window (T, 165)
            trans: Translation data for the window (T, 3)
            expressions: Expression data for the window (T, 100)
            
        Returns:
            Dictionary containing all analysis results
        """
        analysis = {
            'arm_parameters': {},
            'cumulative_changes': {},
            'hand_movement': {},
            'hand_relative_positions': {},
            'finger_analysis': {},
            'finger_summary': {},
            'motion_rep': None,
            'motion_analysis': None
        }
        
        # Calculate arm parameters for this window
        window_arm_params = self.get_arm_parameters(poses)
        analysis['arm_parameters'] = window_arm_params
        
        # Calculate direct changes for each joint (last frame - first frame)
        # Use the actual window size from the data instead of fixed self.window_size
        actual_window_size = poses.shape[0]  # Dynamic window size based on actual data
        
        for joint_name in self.arm_joint_names:
            joint_angles = window_arm_params[joint_name]
            # For single window analysis, directly calculate the change
            if len(joint_angles) > 0:
                first_frame = joint_angles[0]  # First frame in window
                last_frame = joint_angles[-1]  # Last frame in window
                direct_change = last_frame - first_frame  # Direct difference preserving sign
                analysis['cumulative_changes'][joint_name] = direct_change
            else:
                analysis['cumulative_changes'][joint_name] = np.zeros(3)

        # Generate joint positions for hand analysis
        window_joints = self.generate_joint_positions(poses, trans, expressions)
        
        if window_joints is not None:
            # Calculate hand movement metrics
            hand_movement = self.calculate_hand_position_movement(window_joints, hand_type="both")
            analysis['hand_movement'] = hand_movement
            
            # Calculate relative positions between hands
            hand_relative = self.calculate_hand_relative_positions(window_joints)
            analysis['hand_relative_positions'] = hand_relative
            
            # Store joint positions for potential use
            analysis['joint_positions'] = window_joints
        
        # Analyze finger curvature
        finger_analysis = self.analyze_all_fingers(poses)
        analysis['finger_analysis'] = finger_analysis
        
        # Get finger summary
        finger_summary = self.get_finger_summary(finger_analysis)
        analysis['finger_summary'] = finger_summary
        
        # Get full motion representation for this window
        motion_rep = get_motion_rep_with_arm_angles(poses, pose_fps=self.fps, device=get_optimal_device())
        analysis['motion_rep'] = motion_rep
        
        # Generate motion analysis and text annotations
        motion_summary = self.motion_classifier.generate_motion_summary(analysis)
        analysis['motion_analysis'] = motion_summary
        
        return analysis
    
    def get_hand_movement_summary(self, hand_movement: Dict) -> Dict:
        """
        Get summary statistics for hand movement data.
        
        Args:
            hand_movement: Hand movement data from calculate_hand_position_movement
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {}
        
        for hand_name, hand_data in hand_movement.items():
            summary[hand_name] = {
                'total_distance': hand_data['cumulative_total'][-1] if len(hand_data['cumulative_total']) > 0 else 0.0,
                'x_range': hand_data['x_range'],
                'y_range': hand_data['y_range'],
                'z_range': hand_data['z_range'],
                'avg_velocity': hand_data['avg_velocity'],
                'peak_velocity': hand_data['peak_velocity'],
                'final_displacement': hand_data['final_displacement'],
                'start_position': hand_data['start_position'].tolist(),
                'end_position': hand_data['end_position'].tolist(),
            }
        
        return summary
    
    def get_hand_relative_summary(self, hand_relative: Dict) -> Dict:
        """
        Get summary statistics for hand relative position data.
        
        Args:
            hand_relative: Hand relative position data from calculate_hand_relative_positions
            
        Returns:
            Dictionary containing summary statistics
        """
        return {
            'avg_hand_distance': hand_relative['avg_hand_distance'],
            'peak_hand_distance': hand_relative['peak_hand_distance'],
            'min_hand_distance': hand_relative['min_hand_distance'],
            'relative_x_range': hand_relative['relative_x_range'],
            'relative_y_range': hand_relative['relative_y_range'],
            'relative_z_range': hand_relative['relative_z_range'],
            'total_relative_movement': hand_relative['cumulative_relative_x'][-1] + 
                                     hand_relative['cumulative_relative_y'][-1] + 
                                     hand_relative['cumulative_relative_z'][-1] if len(hand_relative['cumulative_relative_x']) > 0 else 0.0,
        }
    
    def get_motion_classifier(self) -> MotionClassifier:
        """
        Get the motion classifier instance.
        
        Returns:
            MotionClassifier instance
        """
        return self.motion_classifier
    
    def update_motion_thresholds(self, new_thresholds: Dict) -> None:
        """
        Update motion classification thresholds.
        
        Args:
            new_thresholds: New threshold values
        """
        self.motion_classifier.update_thresholds(new_thresholds)
    
    def get_motion_thresholds(self) -> Dict:
        """
        Get current motion classification thresholds.
        
        Returns:
            Current threshold configuration
        """
        return self.motion_classifier.get_thresholds()
    
    def print_motion_analysis(self, window_data: Dict) -> None:
        """
        Print formatted motion analysis for a window.
        
        Args:
            window_data: Window data containing motion analysis
        """
        if 'motion_analysis' in window_data:
            self.motion_classifier.print_motion_analysis(window_data['motion_analysis'])
        else:
            print("No motion analysis available for this window")
    
    def print_finger_analysis(self, finger_analysis: Dict) -> None:
        """
        Print formatted finger analysis.
        Thumb and index are shown individually, others are grouped.
        
        Args:
            finger_analysis: Finger analysis from analyze_all_fingers
        """
        print("\n=== Finger Analysis ===")
        
        for hand in ['left_hand', 'right_hand']:
            print(f"\n{hand.upper().replace('_', ' ')}:")
            
            # Print individual finger analysis (thumb and index)
            for finger_name in self.finger_groups[hand]['individual']:
                if finger_name in finger_analysis[hand]:
                    analysis = finger_analysis[hand][finger_name]
                    state = analysis['curvature_state']
                    change = analysis['change_description']
                    curvature = analysis['avg_curvature']
                    
                    print(f"  {finger_name.capitalize()}:")
                    print(f"    State: {state}")
                    print(f"    Change: {change}")
                    print(f"    Average Curvature: {curvature:.3f}")
            
            # Print grouped finger analysis (middle, ring, pinky)
            if 'other_fingers' in finger_analysis[hand]:
                analysis = finger_analysis[hand]['other_fingers']
                state = analysis['curvature_state']
                change = analysis['change_description']
                curvature = analysis['avg_curvature']
                finger_names = analysis['finger_names']
                
                print(f"  Other Fingers ({', '.join(finger_names)}):")
                print(f"    State: {state}")
                print(f"    Change: {change}")
                print(f"    Average Curvature: {curvature:.3f}")
                print(f"    Finger Count: {analysis['finger_count']}")
    
    def print_finger_summary(self, finger_summary: Dict) -> None:
        """
        Print simple finger summary.
        
        Args:
            finger_summary: Finger summary from get_finger_summary
        """
        print("\n=== Finger Summary ===")
        
        for hand in ['left_hand', 'right_hand']:
            print(f"\n{hand.upper().replace('_', ' ')}:")
            for description in finger_summary[hand]:
                print(f"  - {description}")
    
    def get_finger_curvature_thresholds(self) -> Dict[str, float]:
        """
        Get current finger curvature thresholds.
        
        Returns:
            Dictionary containing current thresholds
        """
        return self.finger_thresholds.copy()
    
    def update_finger_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update finger curvature thresholds.
        
        Args:
            new_thresholds: New threshold values
        """
        self.finger_thresholds.update(new_thresholds)
        print("Finger thresholds updated")