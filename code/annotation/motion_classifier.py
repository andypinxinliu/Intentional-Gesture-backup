#!/usr/bin/env python3
"""
Motion Classifier for Pose and Hand Movement Analysis

This module provides functionality to:
1. Define motion thresholds for pose angles and hand positions
2. Classify movements as static/moving and slow/fast
3. Generate text annotations for movement patterns
4. Create and load JSON configuration files for thresholds
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum

class MovementType(Enum):
    """Enumeration for movement types."""
    STATIC = "static"
    SLOW = "slow"
    FAST = "fast"

class Axis(Enum):
    """Enumeration for coordinate axes."""
    X = "x"
    Y = "y"
    Z = "z"

class MotionClassifier:
    """
    Classifies motion based on pose angles and hand positions.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the motion classifier.
        
        Args:
            config_path: Path to JSON configuration file with thresholds
        """
        self.config_path = config_path or "motion_thresholds.json"
        self.thresholds = self._load_or_create_default_config()
        
        # Movement direction descriptions
        self.direction_descriptions = {
            'pose': {
                'x': {'positive': 'rotates right', 'negative': 'rotates left'},
                'y': {'positive': 'rotates forward', 'negative': 'rotates backward'},
                'z': {'positive': 'rotates upward', 'negative': 'rotates downward'}
            },
            'hand': {
                'x': {'positive': 'moves leftward', 'negative': 'moves rightward'},
                'y': {'positive': 'moves upward', 'negative': 'moves downward'},
                'z': {'positive': 'moves forward', 'negative': 'moves backward'}
            },
            'left_shoulder': {
                'x': {'positive': 'rotates counterclockwise', 'negative': 'rotates clockwise'},
                'y': {'positive': 'moves leftward', 'negative': 'moves rightward'},
                'z': {'positive': 'moves upward', 'negative': 'moves downward'}
            },
            'left_elbow': {
                'x': {'positive': 'rotates counterclockwise', 'negative': 'rotates clockwise'},
                'y': {'positive': 'straightens', 'negative': 'bends'},
                'z': {'positive': 'moves upward', 'negative': 'moves downward'}
            },
            'right_elbow': {
                'x': {'positive': 'rotates clockwise', 'negative': 'rotates counterclockwise'},
                'y': {'positive': 'bends', 'negative': 'straightens'},
                'z': {'positive': 'moves upward', 'negative': 'moves downward'}
            },
            'left_wrist': {
                'x': {'positive': 'moves leftward', 'negative': 'moves rightward'},
                'y': {'positive': 'moves upward', 'negative': 'moves downward'},
                'z': {'positive': 'moves forward', 'negative': 'moves backward'}
            }
        }
        
        # Movement intensity descriptions
        self.intensity_descriptions = {
            MovementType.STATIC: "remains almost static",
            MovementType.SLOW: "slightly",
            MovementType.FAST: "significantly"
        }
    
    def _load_or_create_default_config(self) -> Dict:
        """
        Load existing configuration or create default one.
        
        Returns:
            Dictionary containing motion thresholds
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Creating default configuration...")
        
        # Create default configuration
        default_config = {
            "pose_angles": {
                "joints": {
                    "left_shoulder": {
                        "x": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "y": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "z": {"static_threshold": 5.0, "slow_threshold": 15.0}
                    },
                    "right_shoulder": {
                        "x": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "y": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "z": {"static_threshold": 5.0, "slow_threshold": 15.0}
                    },
                    "left_elbow": {
                        "x": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "y": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "z": {"static_threshold": 5.0, "slow_threshold": 15.0}
                    },
                    "right_elbow": {
                        "x": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "y": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "z": {"static_threshold": 5.0, "slow_threshold": 15.0}
                    },
                    "left_wrist": {
                        "x": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "y": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "z": {"static_threshold": 5.0, "slow_threshold": 15.0}
                    },
                    "right_wrist": {
                        "x": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "y": {"static_threshold": 5.0, "slow_threshold": 15.0},
                        "z": {"static_threshold": 5.0, "slow_threshold": 15.0}
                    }
                }
            },
            "hand_positions": {
                "hands": {
                    "left_hand": {
                        "x": {"static_threshold": 0.05, "slow_threshold": 0.15},
                        "y": {"static_threshold": 0.05, "slow_threshold": 0.15},
                        "z": {"static_threshold": 0.05, "slow_threshold": 0.15}
                    },
                    "right_hand": {
                        "x": {"static_threshold": 0.05, "slow_threshold": 0.15},
                        "y": {"static_threshold": 0.05, "slow_threshold": 0.15},
                        "z": {"static_threshold": 0.05, "slow_threshold": 0.15}
                    }
                }
            },
            "relative_hand_distance": {
                "static_threshold": 0.03,  # meters
                "slow_threshold": 0.10     # meters
            }
        }
        
        # Save default configuration
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: Dict = None) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration to save (uses self.thresholds if None)
        """
        if config is None:
            config = self.thresholds
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to: {self.config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def classify_movement(self, movement_value: float, static_threshold: float, slow_threshold: float) -> MovementType:
        """
        Classify movement based on thresholds.
        
        Args:
            movement_value: The movement value to classify
            static_threshold: Threshold below which movement is considered static
            slow_threshold: Threshold between static and fast movement
            
        Returns:
            MovementType enum value
        """
        abs_value = abs(movement_value)
        
        if abs_value < static_threshold:
            return MovementType.STATIC
        elif abs_value < slow_threshold:
            return MovementType.SLOW
        else:
            return MovementType.FAST
    
    def get_movement_description(self, movement_type: MovementType, direction: str, axis: str, motion_type: str = "hand") -> str:
        """
        Generate text description for movement.
        
        Args:
            movement_type: Type of movement (static, slow, fast)
            direction: Direction of movement (positive/negative)
            axis: Axis of movement (x, y, z)
            motion_type: Type of motion (pose, hand, or joint name)
            
        Returns:
            Text description of the movement
        """
        if movement_type == MovementType.STATIC:
            return f"{motion_type} {self.intensity_descriptions[movement_type]} ({axis} axis)"
        
        # Determine the motion category based on the motion_type
        if motion_type in ["left_shoulder", "left_elbow", "left_wrist"]:
            # Use specific descriptions for left side joints
            motion_category = motion_type
        elif motion_type == "right_elbow":
            # Use specific descriptions for right elbow
            motion_category = motion_type
        elif motion_type in ["right_shoulder", "right_wrist"]:
            motion_category = "pose"
        elif motion_type in ["left_hand", "right_hand"]:
            motion_category = "hand"
        elif motion_type == "hands relative":
            motion_category = "hand"  # Use hand descriptions for relative movements
        else:
            motion_category = "pose"  # Default to pose for unknown types
        
        # Get direction description
        direction_desc = self.direction_descriptions[motion_category][axis][direction]
        
        # Get intensity description
        intensity_desc = self.intensity_descriptions[movement_type]
        
        return f"{motion_type} {direction_desc} {intensity_desc}"
    
    def analyze_pose_movements(self, cumulative_changes: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """
        Analyze pose angle movements and generate descriptions.
        
        Args:
            cumulative_changes: Dictionary of cumulative angle changes for each joint
            
        Returns:
            Dictionary mapping joint names to list of movement descriptions
        """
        
        pose_config = self.thresholds["pose_angles"]
        
        descriptions = {}
        
        for joint_name, changes in cumulative_changes.items():
            
            joint_descriptions = []
            
            # Ensure changes is a numpy array with at least 3 elements
            if not isinstance(changes, np.ndarray) or len(changes) < 3:
                print(f"Warning: Invalid changes data for {joint_name}: {changes}")
                continue
            
            # Get joint-specific thresholds
            joint_thresholds = pose_config["joints"].get(joint_name, {
                "x": {"static_threshold": 20.0, "slow_threshold": 40.0},
                "y": {"static_threshold": 20.0, "slow_threshold": 40.0},
                "z": {"static_threshold": 5.0, "slow_threshold": 15.0}
            })
            
            for i, axis in enumerate(['x', 'y', 'z']):
                if i < len(changes):
                    movement_value = changes[i]
                else:
                    movement_value = 0.0
                
                # Skip z-axis analysis for elbows
                if axis == 'z' and joint_name in ['left_elbow', 'right_elbow']:
                    continue
                
                # Get axis-specific thresholds
                axis_thresholds = joint_thresholds.get(axis, {"static_threshold": 5.0, "slow_threshold": 15.0})
                static_threshold = axis_thresholds["static_threshold"]
                slow_threshold = axis_thresholds["slow_threshold"]
                
                # Classify movement
                movement_type = self.classify_movement(movement_value, static_threshold, slow_threshold)
                
                # Determine direction
                direction = "positive" if movement_value > 0 else "negative"
                
                # Generate description
                description = self.get_movement_description(movement_type, direction, axis, joint_name)
                joint_descriptions.append(description)
            
            descriptions[joint_name] = joint_descriptions
                
        
        return descriptions
    
    def analyze_hand_movements(self, hand_movement: Dict[str, Dict]) -> Dict[str, List[str]]:
        """
        Analyze hand position movements and generate descriptions.
        
        Args:
            hand_movement: Dictionary of hand movement data
            
        Returns:
            Dictionary mapping hand names to list of movement descriptions
        """

        hand_config = self.thresholds["hand_positions"]
        
        descriptions = {}
        
        for hand_name, hand_data in hand_movement.items():
            hand_descriptions = []
            
            # Get hand-specific thresholds
            hand_thresholds = hand_config["hands"].get(hand_name, {
                "x": {"static_threshold": 0.05, "slow_threshold": 0.15},
                "y": {"static_threshold": 0.05, "slow_threshold": 0.15},
                "z": {"static_threshold": 0.05, "slow_threshold": 0.15}
            })
            
            # Analyze direct movement for each axis (last frame - first frame)
            for axis in ['x', 'y', 'z']:
                direct_key = f'direct_{axis}'
                if direct_key in hand_data:
                    movement_value = hand_data[direct_key]
                else:
                    # Fallback to cumulative if direct not available
                    cumulative_key = f'cumulative_{axis}'
                    if cumulative_key in hand_data and len(hand_data[cumulative_key]) > 0:
                        movement_value = hand_data[cumulative_key][-1]
                    else:
                        movement_value = 0.0
                
                # Get axis-specific thresholds
                axis_thresholds = hand_thresholds.get(axis, {"static_threshold": 0.05, "slow_threshold": 0.15})
                static_threshold = axis_thresholds["static_threshold"]
                slow_threshold = axis_thresholds["slow_threshold"]
                
                # Classify movement
                movement_type = self.classify_movement(movement_value, static_threshold, slow_threshold)
                
                # Determine direction
                direction = "positive" if movement_value > 0 else "negative"
                
                # Generate description
                description = self.get_movement_description(movement_type, direction, axis, hand_name)
                hand_descriptions.append(description)
            
            descriptions[hand_name] = hand_descriptions

        return descriptions
        
        
    
    def analyze_relative_hand_movements(self, hand_relative: Dict) -> List[str]:
        """
        Analyze relative hand movements and generate descriptions.
        
        Args:
            hand_relative: Dictionary of relative hand position data
            
        Returns:
            List of movement descriptions for relative hand positions
        """
        # Disabled for now - return empty list
        return []
            
       
    
    def generate_motion_summary(self, window_data: Dict) -> Dict[str, any]:
        """
        Generate comprehensive motion summary for a window.
        
        Args:
            window_data: Window data containing pose and hand movement information
            
        Returns:
            Dictionary containing motion analysis and descriptions
        """
        
        summary = {
            "pose_movements": {},
            "hand_movements": {},
            "relative_hand_movements": [],
            "overall_motion_level": "static"
        }
        
        # Analyze pose movements
        if 'cumulative_changes' in window_data and window_data['cumulative_changes']:
            try:
                summary["pose_movements"] = self.analyze_pose_movements(window_data['cumulative_changes'])
            except Exception as e:
                print(f"Error analyzing pose movements: {e}")
                summary["pose_movements"] = {}
        
        # Analyze hand movements
        if 'hand_movement' in window_data and window_data['hand_movement']:
            try:
                summary["hand_movements"] = self.analyze_hand_movements(window_data['hand_movement'])
            except Exception as e:
                print(f"Error analyzing hand movements: {e}")
                summary["hand_movements"] = {}
        
        # Analyze relative hand movements
        if 'hand_relative_positions' in window_data and window_data['hand_relative_positions']:
            try:
                summary["relative_hand_movements"] = self.analyze_relative_hand_movements(window_data['hand_relative_positions'])
            except Exception as e:
                print(f"Error analyzing relative hand movements: {e}")
                summary["relative_hand_movements"] = []
        
        # Determine overall motion level
        try:
            summary["overall_motion_level"] = self._determine_overall_motion_level(summary)
        except Exception as e:
            print(f"Error determining overall motion level: {e}")
            summary["overall_motion_level"] = "error"
        
        return summary
            
    
    
    def _determine_overall_motion_level(self, summary: Dict) -> str:
        """
        Determine overall motion level based on all movements.
        
        Args:
            summary: Motion summary dictionary
            
        Returns:
            Overall motion level description
        """
        # Count different movement types
        static_count = 0
        slow_count = 0
        fast_count = 0
        
        # Count from pose movements
        for joint_descriptions in summary["pose_movements"].values():
            for desc in joint_descriptions:
                if "remains almost static" in desc:
                    static_count += 1
                elif "slightly" in desc:
                    slow_count += 1
                elif "significantly" in desc:
                    fast_count += 1
        
        # Count from hand movements
        for hand_descriptions in summary["hand_movements"].values():
            for desc in hand_descriptions:
                if "remains almost static" in desc:
                    static_count += 1
                elif "slightly" in desc:
                    slow_count += 1
                elif "significantly" in desc:
                    fast_count += 1
        
        # Count from relative hand movements
        for desc in summary["relative_hand_movements"]:
            if "remains almost static" in desc:
                static_count += 1
            elif "slightly" in desc:
                slow_count += 1
            elif "significantly" in desc:
                fast_count += 1
        
        total_movements = static_count + slow_count + fast_count
        
        if total_movements == 0:
            return "no motion data"
        
        # Determine overall level
        if fast_count > 0:
            return "high activity"
        elif slow_count > 0:
            return "moderate activity"
        else:
            return "low activity"
    
    def print_motion_analysis(self, summary: Dict) -> None:
        """
        Print formatted motion analysis.
        
        Args:
            summary: Motion summary dictionary
        """
        print("\n=== Motion Analysis ===")
        print(f"Overall Motion Level: {summary['overall_motion_level']}")
        
        # Print pose movements
        if summary["pose_movements"]:
            print("\nPose Movements:")
            for joint_name, descriptions in summary["pose_movements"].items():
                print(f"  {joint_name}:")
                for desc in descriptions:
                    print(f"    - {desc}")
        
        # Print hand movements
        if summary["hand_movements"]:
            print("\nHand Movements:")
            for hand_name, descriptions in summary["hand_movements"].items():
                print(f"  {hand_name}:")
                for desc in descriptions:
                    print(f"    - {desc}")
        
        # Print relative hand movements
        if summary["relative_hand_movements"]:
            print("\nRelative Hand Movements:")
            for desc in summary["relative_hand_movements"]:
                print(f"  - {desc}")
    
    def update_thresholds(self, new_thresholds: Dict) -> None:
        """
        Update motion thresholds.
        
        Args:
            new_thresholds: New threshold values
        """
        self.thresholds.update(new_thresholds)
        self.save_config()
        print("Thresholds updated and saved")
    
    def get_thresholds(self) -> Dict:
        """
        Get current thresholds.
        
        Returns:
            Current threshold configuration
        """
        return self.thresholds.copy()
