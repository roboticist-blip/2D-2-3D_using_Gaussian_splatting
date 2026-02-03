"""
PLY File Analysis and Validation Utility
Analyze 3D Gaussian Splatting PLY outputs for research and debugging
"""

import numpy as np
from plyfile import PlyData
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GaussianSplattingAnalyzer:
    """Analyze PLY files from 3D Gaussian Splatting"""
    
    def __init__(self, ply_path: str):
        """
        Initialize analyzer with PLY file
        
        Args:
            ply_path: Path to PLY file
        """
        self.ply_path = Path(ply_path)
        if not self.ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
        
        print(f"Loading PLY file: {self.ply_path}")
        self.plydata = PlyData.read(str(self.ply_path))
        self.vertices = self.plydata['vertex']
        
    def get_basic_stats(self) -> dict:
        """Get basic statistics about the model"""
        num_gaussians = len(self.vertices)
        file_size_mb = self.ply_path.stat().st_size / (1024 * 1024)
        
        # Extract positions
        positions = np.stack([
            self.vertices['x'],
            self.vertices['y'],
            self.vertices['z']
        ], axis=1)
        
        # Bounding box
        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)
        bbox_extent = bbox_max - bbox_min
        scene_extent = np.linalg.norm(bbox_extent)
        
        # Center of mass
        center = positions.mean(axis=0)
        
        stats = {
            'num_gaussians': int(num_gaussians),
            'file_size_mb': float(file_size_mb),
            'memory_per_gaussian_bytes': file_size_mb * 1024 * 1024 / num_gaussians,
            'bounding_box': {
                'min': bbox_min.tolist(),
                'max': bbox_max.tolist(),
                'extent': bbox_extent.tolist(),
                'scene_extent': float(scene_extent)
            },
            'center_of_mass': center.tolist(),
            'properties': list(self.vertices.data.dtype.names)
        }
        
        return stats
    
    def analyze_opacity(self) -> dict:
        """Analyze opacity distribution"""
        if 'opacity' in self.vertices.data.dtype.names:
            opacity = np.array(self.vertices['opacity'])
        elif 'alpha' in self.vertices.data.dtype.names:
            opacity = np.array(self.vertices['alpha'])
        else:
            return {'error': 'No opacity field found'}
        
        # Convert to actual opacity (sigmoid)
        opacity_values = 1 / (1 + np.exp(-opacity))
        
        stats = {
            'mean': float(opacity_values.mean()),
            'std': float(opacity_values.std()),
            'min': float(opacity_values.min()),
            'max': float(opacity_values.max()),
            'median': float(np.median(opacity_values)),
            'num_transparent': int(np.sum(opacity_values < 0.1)),
            'num_opaque': int(np.sum(opacity_values > 0.9))
        }
        
        return stats
    
    def analyze_scale(self) -> dict:
        """Analyze Gaussian scale distribution"""
        scale_fields = [f for f in self.vertices.data.dtype.names if f.startswith('scale')]
        
        if not scale_fields:
            return {'error': 'No scale fields found'}
        
        scales = []
        for field in scale_fields:
            scales.append(np.array(self.vertices[field]))
        
        scales = np.stack(scales, axis=1)
        
        # Convert to actual scale (exp)
        scale_values = np.exp(scales)
        
        # Compute average scale per Gaussian
        avg_scale = scale_values.mean(axis=1)
        
        stats = {
            'mean_scale': float(avg_scale.mean()),
            'std_scale': float(avg_scale.std()),
            'min_scale': float(avg_scale.min()),
            'max_scale': float(avg_scale.max()),
            'median_scale': float(np.median(avg_scale)),
            'scale_fields': scale_fields
        }
        
        return stats
    
    def analyze_colors(self) -> dict:
        """Analyze color distribution"""
        # Look for spherical harmonic DC components (base color)
        color_fields = [f for f in self.vertices.data.dtype.names if 'f_dc' in f]
        
        if len(color_fields) < 3:
            return {'error': 'Color information not found'}
        
        # Get DC components (base RGB)
        colors = np.stack([
            self.vertices[color_fields[0]],
            self.vertices[color_fields[1]],
            self.vertices[color_fields[2]]
        ], axis=1)
        
        # Convert from SH to RGB (approximate)
        # DC component relates to RGB through: RGB = 0.5 + SH_C0 * dc
        SH_C0 = 0.28209479177387814
        rgb = 0.5 + SH_C0 * colors
        rgb = np.clip(rgb, 0, 1)
        
        stats = {
            'mean_rgb': rgb.mean(axis=0).tolist(),
            'std_rgb': rgb.std(axis=0).tolist(),
            'color_fields': color_fields,
            'num_sh_components': len([f for f in self.vertices.data.dtype.names if 'f_dc' in f or 'f_rest' in f])
        }
        
        return stats
    
    def create_visualizations(self, output_dir: str):
        """Create visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        positions = np.stack([
            self.vertices['x'],
            self.vertices['y'],
            self.vertices['z']
        ], axis=1)
        
        # 1. 3D scatter plot of Gaussian positions
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample for visualization (to avoid memory issues)
        sample_size = min(10000, len(positions))
        indices = np.random.choice(len(positions), sample_size, replace=False)
        
        ax.scatter(
            positions[indices, 0],
            positions[indices, 1],
            positions[indices, 2],
            c='blue',
            alpha=0.1,
            s=1
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Gaussian Distribution (sampled {sample_size} points)')
        
        plt.savefig(output_path / '3d_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Opacity histogram
        if 'opacity' in self.vertices.data.dtype.names:
            opacity = np.array(self.vertices['opacity'])
            opacity_values = 1 / (1 + np.exp(-opacity))
            
            plt.figure(figsize=(10, 6))
            plt.hist(opacity_values, bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel('Opacity')
            plt.ylabel('Count')
            plt.title('Opacity Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path / 'opacity_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Scale distribution
        scale_fields = [f for f in self.vertices.data.dtype.names if f.startswith('scale')]
        if scale_fields:
            scales = []
            for field in scale_fields:
                scales.append(np.array(self.vertices[field]))
            scales = np.stack(scales, axis=1)
            scale_values = np.exp(scales)
            avg_scale = scale_values.mean(axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.hist(np.log10(avg_scale), bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel('Log10(Average Scale)')
            plt.ylabel('Count')
            plt.title('Gaussian Scale Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path / 'scale_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\nVisualizations saved to: {output_path}")
    
    def generate_report(self, output_path: str = None):
        """Generate comprehensive analysis report"""
        report = {
            'file_info': {
                'path': str(self.ply_path),
                'name': self.ply_path.name
            },
            'basic_stats': self.get_basic_stats(),
            'opacity_analysis': self.analyze_opacity(),
            'scale_analysis': self.analyze_scale(),
            'color_analysis': self.analyze_colors()
        }
        
        if output_path:
            output_file = Path(output_path)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {output_file}")
        
        return report
    
    def print_summary(self):
        """Print a human-readable summary"""
        stats = self.get_basic_stats()
        opacity_stats = self.analyze_opacity()
        scale_stats = self.analyze_scale()
        color_stats = self.analyze_colors()
        
        print("\n" + "=" * 70)
        print("3D GAUSSIAN SPLATTING MODEL ANALYSIS")
        print("=" * 70)
        
        print(f"\n📁 File: {self.ply_path.name}")
        print(f"💾 Size: {stats['file_size_mb']:.2f} MB")
        print(f"🎯 Gaussians: {stats['num_gaussians']:,}")
        print(f"📊 Memory per Gaussian: {stats['memory_per_gaussian_bytes']:.2f} bytes")
        
        print(f"\n📐 Bounding Box:")
        print(f"   Min: [{stats['bounding_box']['min'][0]:.3f}, {stats['bounding_box']['min'][1]:.3f}, {stats['bounding_box']['min'][2]:.3f}]")
        print(f"   Max: [{stats['bounding_box']['max'][0]:.3f}, {stats['bounding_box']['max'][1]:.3f}, {stats['bounding_box']['max'][2]:.3f}]")
        print(f"   Extent: {stats['bounding_box']['scene_extent']:.3f} units")
        
        if 'error' not in opacity_stats:
            print(f"\n👁️  Opacity:")
            print(f"   Mean: {opacity_stats['mean']:.4f}")
            print(f"   Range: [{opacity_stats['min']:.4f}, {opacity_stats['max']:.4f}]")
            print(f"   Transparent (<0.1): {opacity_stats['num_transparent']:,} ({opacity_stats['num_transparent']/stats['num_gaussians']*100:.1f}%)")
            print(f"   Opaque (>0.9): {opacity_stats['num_opaque']:,} ({opacity_stats['num_opaque']/stats['num_gaussians']*100:.1f}%)")
        
        if 'error' not in scale_stats:
            print(f"\n📏 Scale:")
            print(f"   Mean: {scale_stats['mean_scale']:.4f}")
            print(f"   Range: [{scale_stats['min_scale']:.4f}, {scale_stats['max_scale']:.4f}]")
        
        if 'error' not in color_stats:
            print(f"\n🎨 Colors:")
            print(f"   Mean RGB: [{color_stats['mean_rgb'][0]:.3f}, {color_stats['mean_rgb'][1]:.3f}, {color_stats['mean_rgb'][2]:.3f}]")
            print(f"   SH Components: {color_stats['num_sh_components']}")
        
        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze 3D Gaussian Splatting PLY files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python analyze_ply.py model.ply
    
    # Generate full report with visualizations
    python analyze_ply.py model.ply --report report.json --visualize ./viz
    
    # Just print summary
    python analyze_ply.py model.ply --summary-only
        """
    )
    
    parser.add_argument('ply_file', type=str, help='Path to PLY file')
    parser.add_argument('--report', type=str, help='Output path for JSON report')
    parser.add_argument('--visualize', type=str, help='Directory for visualization plots')
    parser.add_argument('--summary-only', action='store_true', help='Only print summary')
    
    args = parser.parse_args()
    
    try:
        analyzer = GaussianSplattingAnalyzer(args.ply_file)
        
        # Always print summary
        analyzer.print_summary()
        
        # Generate report if requested
        if args.report and not args.summary_only:
            analyzer.generate_report(args.report)
        
        # Create visualizations if requested
        if args.visualize and not args.summary_only:
            analyzer.create_visualizations(args.visualize)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
