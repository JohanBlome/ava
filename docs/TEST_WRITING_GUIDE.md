# AVA Test Writing Guide

This guide explains how to write new tests for the AVA video codec testing framework.

## Overview

The AVA testing framework follows a three-step process:
1. **Data Collection**: Generate or collect appropriate test video sources
2. **Data Transformation**: Run encoding tests and collect results
3. **Analysis and Evaluation**: Assess quality and generate reports

## Test Structure

### File Naming Convention
- Test files must start with `test_` (e.g., `test_bitrate_mode.py`)
- Test functions must start with `test_` (e.g., `test_bitrate_mode_support()`)
- This creates a hierarchical test structure that the framework can discover automatically

### Directory Organization
```
python/
├── behavior/          # Tests for encoder behavior patterns
├── bugs/             # Tests for specific bug fixes
├── functionality/    # Tests for core functionality
└── tools/           # Utility modules
```

## Writing a New Test

### Basic Test Structure

```python
from pathlib import Path
import ava_common
import ava_sources
import ava_quality
import encapp

def test_your_feature(device, input_file, workdir, test_data):
    """
    Test description explaining what this test does.
    
    Args:
        device: Device information dictionary
        input_file: Path to input video file
        workdir: Working directory for test files
        test_data: Dictionary to store test results
        
    Returns:
        Dictionary with test results
    """
    # Initialize test data
    test_data = ava_common.initialize_testdata()
    
    # Your test implementation here
    # ...
    
    return {
        "success": True,
        "output_files": ["encoded_video.mp4"],
        "test_data": test_data
    }
```

### Test Parameters

Each test function receives these parameters:

- **`device`**: Dictionary containing:
  - `serial`: Device serial number
  - `device_workdir`: Device working directory
  - `encoder`: Encoder name
  - `encoder_info`: Detailed encoder information
  - `codecs`: Available codecs on device

- **`input_file`**: Path to input video file

- **`workdir`**: Working directory for test files

- **`test_data`**: Dictionary for storing test results and metadata

## Video Source Generation

### Using Predefined Sources

The framework can generate appropriate video sources for your test:

```python
from ava_sources import VideoSourceGenerator

def test_with_generated_sources(device, input_file, workdir, test_data):
    # Generate sources specific to your test
    generator = VideoSourceGenerator(workdir)
    
    if "framerate" in test_data.get("test_name", ""):
        sources = generator.generate_framerate_test_sources()
    elif "motion" in test_data.get("test_name", ""):
        sources = generator.generate_motion_test_sources()
    else:
        sources = generator.generate_all_standard_sources()
    
    # Use the generated sources
    for source in sources:
        # Run your test with this source
        pass
```

### Creating Custom Sources

For specific test requirements, create custom video sources:

```python
from ava_sources import VideoSpec, MotionType, ComplexityProfile

def test_custom_feature(device, input_file, workdir, test_data):
    # Create custom video specification
    spec = VideoSpec(
        width=1920,
        height=1080,
        framerate=60,
        duration=10.0,
        motion_type=MotionType.COMPLEX,
        complexity_profile=ComplexityProfile.HIGH_SPATIAL_HIGH_TEMPORAL
    )
    
    # Generate the video
    generator = VideoSourceGenerator(workdir)
    custom_source = generator.generate_test_pattern(spec)
    
    # Use custom_source in your test
    pass
```

## Example Test Patterns

Here are some common test patterns you might want to implement:

### 1. Framerate Handling Test

Test how the encoder handles different framerates and frame drops:

```python
def test_framerate_handling(device, input_file, workdir, test_data):
    """Test encoder behavior with framerate changes and frame drops"""
    
    # Generate 60fps complex source
    generator = VideoSourceGenerator(workdir)
    sources = generator.generate_framerate_test_sources()
    
    # Test with different framerate settings
    framerates = [60, 30, 15]
    
    for source in sources:
        for target_framerate in framerates:
            # Configure encoder for target framerate
            test = encapp.tests_definitions.Test()
            test.configure.codec = device["encoder"]
            test.configure.framerate = target_framerate
            test.configure.bitrate = "1Mbps"
            
            # Run encoding test
            result = ava_common.run_capture(
                device, source, workdir, "1Mbps", "vbr", 
                f"framerate_{target_framerate}", test
            )
            
            # Analyze results
            # Check if framerate was handled correctly
            pass
```

### 2. Motion Handling Test

Test macroblock handling for different motion patterns:

```python
def test_motion_handling(device, input_file, workdir, test_data):
    """Test macroblock handling for lateral vs horizontal movement"""
    
    # Generate motion-specific sources
    generator = VideoSourceGenerator(workdir)
    sources = generator.generate_motion_test_sources()
    
    for source in sources:
        # Run encoding with different motion settings
        test = encapp.tests_definitions.Test()
        test.configure.codec = device["encoder"]
        test.configure.bitrate = "2Mbps"
        
        # Test horizontal motion
        test.configure.motion_vectors = "horizontal"
        result_h = ava_common.run_capture(
            device, source, workdir, "2Mbps", "vbr", 
            "motion_horizontal", test
        )
        
        # Test lateral motion
        test.configure.motion_vectors = "lateral"
        result_l = ava_common.run_capture(
            device, source, workdir, "2Mbps", "vbr", 
            "motion_lateral", test
        )
        
        # Compare results
        pass
```

### 3. Resolution Change Test

Test transcoding with changing resolution:

```python
def test_resolution_change(device, input_file, workdir, test_data):
    """Test transcoding with changing resolution"""
    
    # Generate resolution change sources
    generator = VideoSourceGenerator(workdir)
    sources = generator.generate_resolution_change_sources()
    
    for source in sources:
        # Test with different resolution settings
        resolutions = [(1280, 720), (1920, 1080), (2560, 1440)]
        
        for width, height in resolutions:
            test = encapp.tests_definitions.Test()
            test.configure.codec = device["encoder"]
            test.configure.resolution = f"{width}x{height}"
            test.configure.bitrate = "3Mbps"
            
            result = ava_common.run_capture(
                device, source, workdir, "3Mbps", "vbr", 
                f"resolution_{width}x{height}", test
            )
            
            # Analyze resolution handling
            pass
```

## Quality Assessment

### Running Quality Metrics

```python
def test_with_quality_assessment(device, input_file, workdir, test_data):
    """Test with quality assessment"""
    
    # Run encoding test
    result = ava_common.run_capture(
        device, input_file, workdir, "1Mbps", "vbr", 
        "quality_test"
    )
    
    # Assess quality
    quality_assessor = ava_quality.QualityAssessment()
    metrics = quality_assessor.assess_quality(
        input_file,  # reference
        result["output_files"][0],  # encoded
        workdir
    )
    
    # Store quality metrics
    test_data["quality_metrics"] = {
        "psnr": metrics.psnr,
        "ssim": metrics.ssim,
        "vmaf": metrics.vmaf
    }
    
    return result
```

## Error Handling

### Proper Error Handling

```python
def test_with_error_handling(device, input_file, workdir, test_data):
    """Test with proper error handling"""
    
    try:
        # Your test code here
        result = ava_common.run_capture(device, input_file, workdir, "1Mbps", "vbr", "test")
        
        return {
            "success": True,
            "output_files": result["output_files"],
            "test_data": test_data
        }
        
    except Exception as e:
        # Log the error
        test_data["error"] = str(e)
        
        return {
            "success": False,
            "output_files": [],
            "test_data": test_data,
            "error_message": str(e)
        }
```

## Test Data Management

### Storing Test Results

```python
def test_with_data_storage(device, input_file, workdir, test_data):
    """Test with proper data storage"""
    
    # Initialize test data
    test_data = ava_common.initialize_testdata()
    
    # Add test-specific data
    test_data["test_parameters"] = {
        "bitrate": "1Mbps",
        "resolution": "1920x1080",
        "framerate": 30
    }
    
    # Run test
    result = ava_common.run_capture(device, input_file, workdir, "1Mbps", "vbr", "test")
    
    # Store results
    test_data["encoding_result"] = result
    test_data["output_file"] = result["output_files"][0]
    
    return {
        "success": True,
        "output_files": result["output_files"],
        "test_data": test_data
    }
```

## Running Tests

### Command Line Interface

```bash
# List all available tests
python3 ava_new.py --test-list

# Run a specific test
python3 ava_new.py --test test_bitrate_mode --encoder c2.qti.hevc.encoder -s device_serial

# Run all tests
python3 ava_new.py --encoder c2.qti.hevc.encoder

# Run with specific input files
python3 ava_new.py --test test_framerate --input-file video1.mp4 --input-file video2.mp4

# Run with debug output
python3 ava_new.py --test test_motion --debug
```

### Test Configuration

Tests can be configured through command line arguments:

- `--encoder`: Specify encoder name (partial or full match)
- `--serial`: Specify device serial numbers
- `--input-file`: Specify input video files
- `--output-dir`: Specify output directory
- `--workdir`: Specify working directory
- `--max-workers`: Specify number of parallel workers
- `--debug`: Enable debug output

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Clean up temporary files
- Don't rely on state from other tests

### 2. Error Handling
- Always handle exceptions gracefully
- Provide meaningful error messages
- Log errors for debugging

### 3. Resource Management
- Use appropriate video sources for your test
- Don't generate unnecessarily large files
- Clean up temporary files

### 4. Documentation
- Write clear docstrings
- Explain what the test is testing
- Document any special requirements

### 5. Performance
- Consider test execution time
- Use appropriate video durations
- Optimize for your testing needs

## Example: Complete Test

Here's a complete example of a test that follows all best practices:

```python
from pathlib import Path
import ava_common
import ava_sources
import ava_quality
import encapp

def test_bitrate_ladder_quality(device, input_file, workdir, test_data):
    """
    Test encoder quality across a range of bitrates.
    
    This test encodes the same video at different bitrates and measures
    quality metrics to create a rate-distortion curve.
    """
    try:
        # Initialize test data
        test_data = ava_common.initialize_testdata()
        test_data["test_type"] = "bitrate_ladder"
        
        # Generate appropriate source if needed
        generator = ava_sources.VideoSourceGenerator(workdir)
        if "complex" in str(input_file).lower():
            sources = [input_file]  # Use provided complex source
        else:
            # Generate a complex source for bitrate testing
            spec = ava_sources.VideoSpec(
                width=1920, height=1080, framerate=30, duration=10.0,
                motion_type=ava_sources.MotionType.COMPLEX,
                complexity_profile=ava_sources.ComplexityProfile.HIGH_SPATIAL_HIGH_TEMPORAL
            )
            sources = [generator.generate_test_pattern(spec)]
        
        # Define bitrate ladder
        bitrates = ["500k", "1M", "2M", "4M", "8M"]
        quality_results = []
        
        for bitrate in bitrates:
            # Configure test
            test = encapp.tests_definitions.Test()
            test.configure.codec = device["encoder"]
            test.configure.bitrate = f"{bitrate}bps"
            test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr
            
            # Run encoding
            result = ava_common.run_capture(
                device, sources[0], workdir, f"{bitrate}bps", "vbr", 
                f"bitrate_{bitrate}", test
            )
            
            if result["success"]:
                # Assess quality
                quality_assessor = ava_quality.QualityAssessment()
                metrics = quality_assessor.assess_quality(
                    sources[0],  # reference
                    result["output_files"][0],  # encoded
                    workdir
                )
                
                quality_results.append({
                    "bitrate": bitrate,
                    "psnr": metrics.psnr,
                    "ssim": metrics.ssim,
                    "vmaf": metrics.vmaf,
                    "file_size": metrics.file_size
                })
        
        # Store results
        test_data["quality_results"] = quality_results
        test_data["bitrate_ladder"] = bitrates
        
        return {
            "success": True,
            "output_files": [r["output_file"] for r in quality_results if "output_file" in r],
            "test_data": test_data
        }
        
    except Exception as e:
        test_data["error"] = str(e)
        return {
            "success": False,
            "output_files": [],
            "test_data": test_data,
            "error_message": str(e)
        }
```

This test demonstrates:
- Proper error handling
- Source generation
- Quality assessment
- Data storage
- Clear documentation
- Appropriate test structure

## Conclusion

This guide provides the foundation for writing effective tests in the AVA framework. Follow these patterns and best practices to create robust, maintainable tests that provide valuable insights into video codec performance.
