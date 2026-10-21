#!/usr/bin/env python3
"""
FAMOS: Feed-Forward 3D Articulation Modeling from Sparse Observations

A command-line tool for reconstructing articulated 3D models from sparse
observations using a feed-forward inference approach. This tool processes
sparse 3D point observations (e.g., from depth sensors, motion capture,
or structure-from-motion) and reconstructs articulated structures with
joints, bones, and kinematic chains.

Architecture Overview:
    The tool implements a feed-forward pipeline that:
    1. Ingests sparse 3D observations from various input formats
    2. Performs spatial clustering to identify rigid body parts
    3. Infers articulation points (joints) between connected components
    4. Constructs a kinematic tree representation
    5. Outputs the articulated model in standard formats

Use Cases:
    - Reconstructing articulated objects from partial scans
    - Inferring joint locations from motion capture data
    - Building kinematic models from sparse point clouds
    - Analyzing articulation patterns in 3D observations

Author: FAMOS Development Team
License: MIT
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import csv
import struct


# ============================================================================
# Constants and Configuration
# ============================================================================

VERSION = "1.0.0"
DEFAULT_CLUSTER_RADIUS = 0.5
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_JOINT_THRESHOLD = 0.3
EPSILON = 1e-10


class JointType(Enum):
    """Types of articulation joints supported by FAMOS."""
    REVOLUTE = "revolute"      # Single-axis rotation (hinge)
    PRISMATIC = "prismatic"    # Single-axis translation (slider)
    SPHERICAL = "spherical"    # Ball-and-socket (3 DOF rotation)
    FIXED = "fixed"            # Rigid connection
    UNKNOWN = "unknown"        # Unclassified joint


class OutputFormat(Enum):
    """Supported output formats for articulated models."""
    JSON = "json"
    URDF = "urdf"
    YAML = "yaml"
    GRAPHVIZ = "dot"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Vector3:
    """3D vector representation with basic operations."""
    x: float
    y: float
    z: float

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> 'Vector3':
        if abs(scalar) < EPSILON:
            raise ValueError("Division by near-zero scalar")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: 'Vector3') -> float:
        """Compute dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3') -> 'Vector3':
        """Compute cross product with another vector."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self) -> float:
        """Compute vector magnitude (length)."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> 'Vector3':
        """Return unit vector in same direction."""
        mag = self.magnitude()
        if mag < EPSILON:
            return Vector3(0.0, 0.0, 0.0)
        return self / mag

    def distance_to(self, other: 'Vector3') -> float:
        """Compute Euclidean distance to another point."""
        return (self - other).magnitude()

    def to_list(self) -> List[float]:
        """Convert to list representation."""
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, values: List[float]) -> 'Vector3':
        """Create Vector3 from list of coordinates."""
        if len(values) < 3:
            raise ValueError(f"Need at least 3 coordinates, got {len(values)}")
        return cls(float(values[0]), float(values[1]), float(values[2]))

    @classmethod
    def zero(cls) -> 'Vector3':
        """Return zero vector."""
        return cls(0.0, 0.0, 0.0)


@dataclass
class Observation:
    """A single sparse 3D observation point with optional metadata."""
    position: Vector3
    timestamp: float = 0.0
    confidence: float = 1.0
    label: Optional[str] = None
    frame_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary."""
        return {
            'position': self.position.to_list(),
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'label': self.label,
            'frame_id': self.frame_id
        }


@dataclass
class RigidPart:
    """A rigid body part identified from clustered observations."""
    part_id: int
    points: List[Vector3] = field(default_factory=list)
    centroid: Vector3 = field(default_factory=Vector3.zero)
    bounding_box_min: Vector3 = field(default_factory=Vector3.zero)
    bounding_box_max: Vector3 = field(default_factory=Vector3.zero)
    principal_axes: List[Vector3] = field(default_factory=list)

    def compute_properties(self) -> None:
        """Compute geometric properties of the rigid part."""
        if not self.points:
            return

        # Compute centroid
        sum_x = sum(p.x for p in self.points)
        sum_y = sum(p.y for p in self.points)
        sum_z = sum(p.z for p in self.points)
        n = len(self.points)
        self.centroid = Vector3(sum_x / n, sum_y / n, sum_z / n)

        # Compute bounding box
        self.bounding_box_min = Vector3(
            min(p.x for p in self.points),
            min(p.y for p in self.points),
            min(p.z for p in self.points)
        )
        self.bounding_box_max = Vector3(
            max(p.x for p in self.points),
            max(p.y for p in self.points),
            max(p.z for p in self.points)
        )

        # Compute principal axes using covariance analysis
        self.principal_axes = self._compute_principal_axes()

    def _compute_principal_axes(self) -> List[Vector3]:
        """
        Compute principal axes of the point distribution.
        Uses a simplified PCA approach with power iteration.
        """
        if len(self.points) < 3:
            return [Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)]

        # Compute covariance matrix elements
        n = len(self.points)
        cxx = sum((p.x - self.centroid.x)**2 for p in self.points) / n
        cyy = sum((p.y - self.centroid.y)**2 for p in self.points) / n
        czz = sum((p.z - self.centroid.z)**2 for p in self.points) / n
        cxy = sum((p.x - self.centroid.x)*(p.y - self.centroid.y) for p in self.points) / n
        cxz = sum((p.x - self.centroid.x)*(p.z - self.centroid.z) for p in self.points) / n
        cyz = sum((p.y - self.centroid.y)*(p.z - self.centroid.z) for p in self.points) / n

        # Power iteration to find dominant eigenvector
        def power_iteration(matrix: List[List[float]], iterations: int = 50) -> Vector3:
            v = [1.0, 0.0, 0.0]
            for _ in range(iterations):
                new_v = [
                    matrix[0][0]*v[0] + matrix[0][1]*v[1] + matrix[0][2]*v[2],
                    matrix[1][0]*v[0] + matrix[1][1]*v[1] + matrix[1][2]*v[2],
                    matrix[2][0]*v[0] + matrix[2][1]*v[1] + matrix[2][2]*v[2]
                ]
                mag = math.sqrt(sum(x*x for x in new_v))
                if mag < EPSILON:
                    break
                v = [x/mag for x in new_v]
            return Vector3(v[0], v[1], v[2])

        cov_matrix = [
            [cxx, cxy, cxz],
            [cxy, cyy, cyz],
            [cxz, cyz, czz]
        ]

        # Get first principal axis
        axis1 = power_iteration(cov_matrix)

        # Deflate and get second axis
        lambda1 = (axis1.x * (cxx*axis1.x + cxy*axis1.y + cxz*axis1.z) +
                   axis1.y * (cxy*axis1.x + cyy*axis1.y + cyz*axis1.z) +
                   axis1.z * (cxz*axis1.x + cyz*axis1.y + czz*axis1.z))

        deflated = [[cov_matrix[i][j] - lambda1 * (axis1.to_list()[i] if i == j else 0) *
                     axis1.to_list()[i] * axis1.to_list()[j]
                     for j in range(3)] for i in range(3)]

        axis2 = power_iteration(deflated)

        # Third axis is cross product
        axis3 = axis1.cross(axis2)

        return [axis1, axis2, axis3]

    def to_dict(self) -> Dict[str, Any]:
        """Convert rigid part to dictionary."""
        return {
            'part_id': self.part_id,
            'num_points': len(self.points),
            'centroid': self.centroid.to_list(),
            'bounding_box': {
                'min': self.bounding_box_min.to_list(),
                'max': self.bounding_box_max.to_list()
            },
            'principal_axes': [axis.to_list() for axis in self.principal_axes]
        }


@dataclass
class Joint:
    """An articulation joint connecting two rigid parts."""
    joint_id: int
    joint_type: JointType
    parent_part_id: int
    child_part_id: int
    position: Vector3
    axis: Vector3 = field(default_factory=lambda: Vector3(0, 0, 1))
    limits: Tuple[float, float] = (-math.pi, math.pi)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert joint to dictionary."""
        return {
            'joint_id': self.joint_id,
            'joint_type': self.joint_type.value,
            'parent_part_id': self.parent_part_id,
            'child_part_id': self.child_part_id,
            'position': self.position.to_list(),
            'axis': self.axis.to_list(),
            'limits': list(self.limits),
            'confidence': self.confidence
        }


@dataclass
class ArticulatedModel:
    """Complete articulated model with parts and joints."""
    name: str
    parts: List[RigidPart] = field(default_factory=list)
    joints: List[Joint] = field(default_factory=list)
    root_part_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_kinematic_tree(self) -> Dict[int, List[int]]:
        """
        Build adjacency list representation of kinematic tree.
        Returns dict mapping parent part ID to list of child part IDs.
        """
        tree = defaultdict(list)
        for joint in self.joints:
            tree[joint.parent_part_id].append(joint.child_part_id)
        return dict(tree)

    def get_part_by_id(self, part_id: int) -> Optional[RigidPart]:
        """Retrieve a part by its ID."""
        for part in self.parts:
            if part.part_id == part_id:
                return part
        return None

    def compute_dof(self) -> int:
        """Compute total degrees of freedom of the articulated model."""
        dof = 0
        for joint in self.joints:
            if joint.joint_type == JointType.REVOLUTE:
                dof += 1
            elif joint.joint_type == JointType.PRISMATIC:
                dof += 1
            elif joint.joint_type == JointType.SPHERICAL:
                dof += 3
        return dof

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            'name': self.name,
            'num_parts': len(self.parts),
            'num_joints': len(self.joints),
            'total_dof': self.compute_dof(),
            'root_part_id': self.root_part_id,
            'parts': [part.to_dict() for part in self.parts],
            'joints': [joint.to_dict() for joint in self.joints],
            'kinematic_tree': {str(k): v for k, v in self.get_kinematic_tree().items()},
            'metadata': self.metadata
        }


# ============================================================================
# Input/Output Handlers
# ============================================================================

class ObservationLoader:
    """Load sparse observations from various file formats."""

    @staticmethod
    def load(filepath: str) -> List[Observation]:
        """
        Load observations from file, auto-detecting format.

        Supported formats:
            - CSV: x,y,z[,timestamp,confidence,label]
            - JSON: array of objects with position/xyz fields
            - PLY: ASCII point cloud format
            - XYZ: simple space-separated coordinates

        Args:
            filepath: Path to input file

        Returns:
            List of Observation objects

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If file format is unsupported or malformed
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")

        suffix = path.suffix.lower()

        if suffix == '.csv':
            return ObservationLoader._load_csv(filepath)
        elif suffix == '.json':
            return ObservationLoader._load_json(filepath)
        elif suffix == '.ply':
            return ObservationLoader._load_ply(filepath)
        elif suffix in ('.xyz', '.txt', '.pts'):
            return ObservationLoader._load_xyz(filepath)
        else:
            # Try to auto-detect based on content
            return ObservationLoader._auto_detect(filepath)

    @staticmethod
    def _load_csv(filepath: str) -> List[Observation]:
        """Load observations from CSV file."""
        observations = []

        with open(filepath, 'r', newline='') as f:
            # Try to detect if there's a header
            sample = f.read(1024)
            f.seek(0)
            has_header = csv.Sniffer().has_header(sample) if sample else False

            reader = csv.reader(f)
            if has_header:
                header = next(reader)
                # Map column names to indices
                col_map = {}
                for i, col in enumerate(header):
                    col_lower = col.lower().strip()
                    if col_lower in ('x', 'px', 'pos_x'):
                        col_map['x'] = i
                    elif col_lower in ('y', 'py', 'pos_y'):
                        col_map['y'] = i
                    elif col_lower in ('z', 'pz', 'pos_z'):
                        col_map['z'] = i
                    elif col_lower in ('t', 'time', 'timestamp'):
                        col_map['timestamp'] = i
                    elif col_lower in ('c', 'conf', 'confidence'):
                        col_map['confidence'] = i
                    elif col_lower in ('label', 'name', 'id'):
                        col_map['label'] = i

                if 'x' not in col_map or 'y' not in col_map or 'z' not in col_map:
                    # Fall back to positional columns
                    col_map = {'x': 0, 'y': 1, 'z': 2}
            else:
                col_map = {'x': 0, 'y': 1, 'z': 2}

            for row_num, row in enumerate(reader, start=2 if has_header else 1):
                if not row or all(not cell.strip() for cell in row):
                    continue

                try:
                    x = float(row[col_map['x']])
                    y = float(row[col_map['y']])
                    z = float(row[col_map['z']])

                    timestamp = 0.0
                    confidence = 1.0
                    label = None

                    if 'timestamp' in col_map and col_map['timestamp'] < len(row):
                        timestamp = float(row[col_map['timestamp']])
                    if 'confidence' in col_map and col_map['confidence'] < len(row):
                        confidence = float(row[col_map['confidence']])
                    if 'label' in col_map and col_map['label'] < len(row):
                        label = row[col_map['label']].strip() or None

                    observations.append(Observation(
                        position=Vector3(x, y, z),
                        timestamp=timestamp,
                        confidence=confidence,
                        label=label,
                        frame_id=0
                    ))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed row {row_num}: {e}",
                          file=sys.stderr)

        return observations

    @staticmethod
    def _load_json(filepath: str) -> List[Observation]:
        """Load observations from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        observations = []

        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if 'observations' in data:
                items = data['observations']
            elif 'points' in data:
                items = data['points']
            elif 'frames' in data:
                # Multi-frame format
                items = []
                for frame_idx, frame in enumerate(data['frames']):
                    for pt in frame.get('points', []):
                        pt['frame_id'] = frame_idx
                        items.append(pt)
            else:
                raise ValueError("JSON file must contain 'observations', 'points', or 'frames' array")
        else:
            raise ValueError("Invalid JSON structure")

        for idx, item in enumerate(items):
            try:
                if isinstance(item, list):
                    pos = Vector3.from_list(item)
                    observations.append(Observation(position=pos, frame_id=idx))
                elif isinstance(item, dict):
                    # Extract position
                    if 'position' in item:
                        pos = Vector3.from_list(item['position'])
                    elif 'xyz' in item:
                        pos = Vector3.from_list(item['xyz'])
                    elif 'x' in item and 'y' in item and 'z' in item:
                        pos = Vector3(item['x'], item['y'], item['z'])
                    else:
                        continue

                    observations.append(Observation(
                        position=pos,
                        timestamp=item.get('timestamp', item.get('t', 0.0)),
                        confidence=item.get('confidence', item.get('c', 1.0)),
                        label=item.get('label', item.get('name')),
                        frame_id=item.get('frame_id', 0)
                    ))
            except (ValueError, KeyError, TypeError) as e:
                print(f"Warning: Skipping malformed item {idx}: {e}", file=sys.stderr)

        return observations

    @staticmethod
    def _load_ply(filepath: str) -> List[Observation]:
        """Load observations from ASCII PLY point cloud file."""
        observations = []

        with open(filepath, 'r') as f:
            # Parse header
            vertex_count = 0
            properties = []
            in_header = True

            for line in f:
                line = line.strip()
                if in_header:
                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line.startswith('property'):
                        parts = line.split()
                        if len(parts) >= 3:
                            properties.append(parts[-1])
                    elif line == 'end_header':
                        in_header = False
                        break

            # Find x, y, z property indices
            try:
                x_idx = properties.index('x')
                y_idx = properties.index('y')
                z_idx = properties.index('z')
            except ValueError:
                x_idx, y_idx, z_idx = 0, 1, 2

            # Read vertex data
            for i, line in enumerate(f):
                if i >= vertex_count:
                    break

                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[x_idx])
                        y = float(parts[y_idx])
                        z = float(parts[z_idx])
                        observations.append(Observation(
                            position=Vector3(x, y, z),
                            frame_id=i
                        ))
                    except ValueError:
                        continue

        return observations

    @staticmethod
    def _load_xyz(filepath: str) -> List[Observation]:
        """Load observations from simple XYZ format (space/tab separated)."""
        observations = []

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Try different separators
                parts = line.replace(',', ' ').replace('\t', ' ').split()

                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        timestamp = float(parts[3]) if len(parts) > 3 else 0.0
                        observations.append(Observation(
                            position=Vector3(x, y, z),
                            timestamp=timestamp,
                            frame_id=line_num - 1
                        ))
                    except ValueError as e:
                        print(f"Warning: Skipping line {line_num}: {e}", file=sys.stderr)

        return observations

    @staticmethod
    def _auto_detect(filepath: str) -> List[Observation]:
        """Auto-detect file format based on content."""
        with open(filepath, 'r') as f:
            first_lines = [f.readline() for _ in range(5)]

        content = ''.join(first_lines)

        if content.strip().startswith('{') or content.strip().startswith('['):
            return ObservationLoader._load_json(filepath)
        elif 'ply' in content.lower():
            return ObservationLoader._load_ply(filepath)
        elif ',' in first_lines[0]:
            return ObservationLoader._load_csv(filepath)
        else:
            return ObservationLoader._load_xyz(filepath)


class ModelExporter:
    """Export articulated models to various formats."""

    @staticmethod
    def export(model: ArticulatedModel, filepath: str,
               format_type: OutputFormat) -> None:
        """
        Export articulated model to file.

        Args:
            model: The articulated model to export
            filepath: Output file path
            format_type: Output format enum value
        """
        if format_type == OutputFormat.JSON:
            ModelExporter._export_json(model, filepath)
        elif format_type == OutputFormat.URDF:
            ModelExporter._export_urdf(model, filepath)
        elif format_type == OutputFormat.YAML:
            ModelExporter._export_yaml(model, filepath)
        elif format_type == OutputFormat.GRAPHVIZ:
            ModelExporter._export_graphviz(model, filepath)
        else:
            raise ValueError(f"Unsupported output format: {format_type}")

    @staticmethod
    def _export_json(model: ArticulatedModel, filepath: str) -> None:
        """Export model to JSON format."""
        with open(filepath, 'w') as f:
            json.dump(model.to_dict(), f, indent=2)

    @staticmethod
    def _export_urdf(model: ArticulatedModel, filepath: str) -> None:
        """Export model to URDF (Unified Robot Description Format)."""
        lines = [
            '<?xml version="1.0"?>',
            f'<!-- Generated by FAMOS v{VERSION} -->',
            f'<robot name="{model.name}">',
            ''
        ]

        # Export links (parts)
        for part in model.parts:
            size = Vector3(
                part.bounding_box_max.x - part.bounding_box_min.x,
                part.bounding_box_max.y - part.bounding_box_min.y,
                part.bounding_box_max.z - part.bounding_box_min.z
            )
            lines.extend([
                f'  <link name="part_{part.part_id}">',
                f'    <visual>',
                f'      <origin xyz="{part.centroid.x:.6f} {part.centroid.y:.6f} {part.centroid.z:.6f}" rpy="0 0 0"/>',
                f'      <geometry>',
                f'        <box size="{max(size.x, 0.01):.6f} {max(size.y, 0.01):.6f} {max(size.z, 0.01):.6f}"/>',
                f'      </geometry>',
                f'    </visual>',
                f'    <inertial>',
                f'      <mass value="1.0"/>',
                f'      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>',
                f'    </inertial>',
                f'  </link>',
                ''
            ])

        # Export joints
        for joint in model.joints:
            joint_type_map = {
                JointType.REVOLUTE: 'revolute',
                JointType.PRISMATIC: 'prismatic',
                JointType.SPHERICAL: 'ball',
                JointType.FIXED: 'fixed',
                JointType.UNKNOWN: 'floating'
            }
            urdf_type = joint_type_map.get(joint.joint_type, 'fixed')

            lines.extend([
                f'  <joint name="joint_{joint.joint_id}" type="{urdf_type}">',
                f'    <parent link="part_{joint.parent_part_id}"/>',
                f'    <child link="part_{joint.child_part_id}"/>',
                f'    <origin xyz="{joint.position.x:.6f} {joint.position.y:.6f} {joint.position.z:.6f}" rpy="0 0 0"/>',
            ])

            if joint.joint_type in (JointType.REVOLUTE, JointType.PRISMATIC):
                lines.append(f'    <axis xyz="{joint.axis.x:.6f} {joint.axis.y:.6f} {joint.axis.z:.6f}"/>')
                lines.append(f'    <limit lower="{joint.limits[0]:.6f}" upper="{joint.limits[1]:.6f}" effort="100" velocity="1"/>')

            lines.extend([
                f'  </joint>',