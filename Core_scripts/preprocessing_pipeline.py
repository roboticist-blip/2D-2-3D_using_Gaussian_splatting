"""
3D Gaussian Splatting Preprocessing Pipeline
Author: [Your Name]
Description: Professional pipeline for converting video data to 3D Gaussian Splatting format
"""

import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """
    Extract frames from video with various sampling strategies.
    Supports multiple quality and sampling configurations for research purposes.
    """
    
    def __init__(self, video_path: str, output_dir: str, sampling_strategy: str = 'uniform'):
        """
        Initialize the frame extractor.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            sampling_strategy: Strategy for frame extraction ('uniform', 'keyframe', 'adaptive')
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.sampling_strategy = sampling_strategy
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata = {
            'video_path': str(self.video_path),
            'extraction_time': datetime.now().isoformat(),
            'sampling_strategy': sampling_strategy,
            'frames': []
        }
        
    def extract_frames(self, 
                      fps: float = None, 
                      max_frames: int = None,
                      resolution: Tuple[int, int] = None,
                      quality: int = 95) -> int:
        """
        Extract frames from video.
        
        Args:
            fps: Target frames per second (None for original fps)
            max_frames: Maximum number of frames to extract
            resolution: Target resolution (width, height). None keeps original
            quality: JPEG quality (1-100)
            
        Returns:
            Number of frames extracted
        """
        logger.info(f"Starting frame extraction from {self.video_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {original_fps} fps, {total_frames} frames, {original_width}x{original_height}")
        
        # Store video metadata
        self.metadata['video_properties'] = {
            'fps': original_fps,
            'total_frames': total_frames,
            'width': original_width,
            'height': original_height
        }
        
        # Calculate sampling interval
        if fps is None:
            fps = original_fps
        
        frame_interval = int(original_fps / fps) if fps < original_fps else 1
        
        # Determine frames to extract
        if max_frames:
            frame_interval = max(frame_interval, total_frames // max_frames)
        
        logger.info(f"Extracting every {frame_interval} frame(s)")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Resize if needed
                if resolution:
                    frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_LANCZOS4)
                
                # Save frame
                frame_filename = f"frame_{extracted_count:06d}.jpg"
                frame_path = self.output_dir / frame_filename
                
                cv2.imwrite(
                    str(frame_path), 
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                
                # Store frame metadata
                self.metadata['frames'].append({
                    'filename': frame_filename,
                    'original_frame_index': frame_count,
                    'timestamp': frame_count / original_fps
                })
                
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    logger.info(f"Reached maximum frame limit: {max_frames}")
                    break
            
            frame_count += 1
            
            # Progress logging
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames, extracted {extracted_count}")
        
        cap.release()
        
        # Save metadata
        metadata_path = self.output_dir / "extraction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Extraction complete: {extracted_count} frames saved to {self.output_dir}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return extracted_count


class CameraParameterEstimator:
    """
    Estimate camera parameters from extracted frames using COLMAP-compatible format.
    Prepares data for Structure-from-Motion (SfM) processing.
    """
    
    def __init__(self, images_dir: str, output_dir: str):
        """
        Initialize camera parameter estimator.
        
        Args:
            images_dir: Directory containing extracted frames
            output_dir: Directory to save camera parameter data
        """
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_image_quality(self) -> Dict:
        """
        Analyze image quality metrics for the dataset.
        
        Returns:
            Dictionary containing quality metrics
        """
        logger.info("Analyzing image quality metrics")
        
        image_files = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
        
        if not image_files:
            raise ValueError(f"No images found in {self.images_dir}")
        
        metrics = {
            'num_images': len(image_files),
            'resolutions': [],
            'blur_scores': [],
            'brightness_scores': []
        }
        
        for img_path in image_files[:min(50, len(image_files))]:  # Sample for speed
            img = cv2.imread(str(img_path))
            
            # Resolution
            h, w = img.shape[:2]
            metrics['resolutions'].append((w, h))
            
            # Blur detection (Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['blur_scores'].append(blur_score)
            
            # Brightness
            brightness = np.mean(gray)
            metrics['brightness_scores'].append(brightness)
        
        # Summary statistics
        summary = {
            'total_images': metrics['num_images'],
            'avg_blur_score': np.mean(metrics['blur_scores']),
            'avg_brightness': np.mean(metrics['brightness_scores']),
            'resolution_mode': max(set(metrics['resolutions']), key=metrics['resolutions'].count)
        }
        
        logger.info(f"Quality analysis: {summary}")
        
        # Save quality report
        report_path = self.output_dir / "quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def create_colmap_structure(self) -> str:
        """
        Create directory structure compatible with COLMAP for SfM processing.
        
        Returns:
            Path to COLMAP input directory
        """
        colmap_dir = self.output_dir / "colmap_input"
        colmap_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (colmap_dir / "images").mkdir(exist_ok=True)
        
        # Copy or symlink images
        import shutil
        image_files = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
        
        logger.info(f"Preparing {len(image_files)} images for COLMAP")
        
        for img_path in image_files:
            dest = colmap_dir / "images" / img_path.name
            shutil.copy(str(img_path), str(dest))
        
        logger.info(f"COLMAP structure created at {colmap_dir}")
        
        return str(colmap_dir)


class GaussianSplattingDataPrep:
    """
    Prepare data in the format required for 3D Gaussian Splatting training.
    Creates the necessary directory structure and configuration files.
    """
    
    def __init__(self, source_dir: str, output_dir: str):
        """
        Initialize data preparation.
        
        Args:
            source_dir: Directory containing processed images
            output_dir: Output directory for training data
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_training_structure(self, colmap_dir: str = None) -> Dict[str, str]:
        """
        Create the complete training data structure.
        
        Args:
            colmap_dir: Optional path to COLMAP sparse reconstruction
            
        Returns:
            Dictionary mapping structure components to their paths
        """
        structure = {
            'images': self.output_dir / 'images',
            'sparse': self.output_dir / 'sparse' / '0',
            'config': self.output_dir / 'config.json'
        }
        
        for key, path in structure.items():
            if key != 'config':
                path.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        import shutil
        image_files = sorted(self.source_dir.glob("*.jpg")) + sorted(self.source_dir.glob("*.png"))
        
        for img_path in image_files:
            dest = structure['images'] / img_path.name
            if not dest.exists():
                shutil.copy(str(img_path), str(dest))
        
        # Create configuration
        config = {
            'dataset_name': self.output_dir.name,
            'source_path': str(self.output_dir),
            'model_path': str(self.output_dir / 'output'),
            'images': 'images',
            'sparse_reconstruction': 'sparse/0',
            'created_at': datetime.now().isoformat()
        }
        
        with open(structure['config'], 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training structure created at {self.output_dir}")
        logger.info(f"Next steps: Run COLMAP on {structure['images']} to generate sparse reconstruction")
        
        return {k: str(v) for k, v in structure.items()}


def main():
    """Main execution pipeline"""
    parser = argparse.ArgumentParser(
        description='3D Gaussian Splatting Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Extract 300 frames at 2 FPS
    python preprocessing_pipeline.py --video input.mp4 --output ./data --fps 2 --max-frames 300
    
    # Extract with custom resolution
    python preprocessing_pipeline.py --video input.mp4 --output ./data --resolution 1920 1080
        """
    )
    
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--fps', type=float, default=None, help='Target FPS for extraction')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames')
    parser.add_argument('--resolution', type=int, nargs=2, help='Target resolution (width height)')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality (1-100)')
    parser.add_argument('--skip-quality-check', action='store_true', help='Skip image quality analysis')
    
    args = parser.parse_args()
    
    # Create output structure
    output_dir = Path(args.output)
    frames_dir = output_dir / 'extracted_frames'
    processed_dir = output_dir / 'processed'
    training_dir = output_dir / 'training_data'
    
    # Step 1: Extract frames
    logger.info("=" * 80)
    logger.info("STEP 1: Frame Extraction")
    logger.info("=" * 80)
    
    extractor = VideoFrameExtractor(args.video, str(frames_dir))
    resolution = tuple(args.resolution) if args.resolution else None
    
    num_frames = extractor.extract_frames(
        fps=args.fps,
        max_frames=args.max_frames,
        resolution=resolution,
        quality=args.quality
    )
    
    # Step 2: Quality analysis
    if not args.skip_quality_check:
        logger.info("=" * 80)
        logger.info("STEP 2: Quality Analysis")
        logger.info("=" * 80)
        
        estimator = CameraParameterEstimator(str(frames_dir), str(processed_dir))
        quality_metrics = estimator.analyze_image_quality()
    
    # Step 3: Prepare COLMAP structure
    logger.info("=" * 80)
    logger.info("STEP 3: COLMAP Structure Preparation")
    logger.info("=" * 80)
    
    estimator = CameraParameterEstimator(str(frames_dir), str(processed_dir))
    colmap_dir = estimator.create_colmap_structure()
    
    # Step 4: Create training data structure
    logger.info("=" * 80)
    logger.info("STEP 4: Training Data Structure")
    logger.info("=" * 80)
    
    data_prep = GaussianSplattingDataPrep(str(frames_dir), str(training_dir))
    structure = data_prep.create_training_structure()
    
    # Final summary
    logger.info("=" * 80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Frames extracted: {num_frames}")
    logger.info(f"Data directory: {training_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run COLMAP SfM on the images:")
    logger.info(f"   colmap_dir: {colmap_dir}")
    logger.info("2. Upload training data to Google Colab for Gaussian Splatting training")
    logger.info("3. Use the generated PLY file for visualization")


if __name__ == "__main__":
    main()
