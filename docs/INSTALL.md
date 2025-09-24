# AVA Installation Guide

This guide explains how to install and use the AVA video codec testing framework.

## Installation Options

### Option 1: Direct Execution (No Installation)

If you just want to run AVA without installing it:

```bash
# Navigate to the ava directory
cd ava

# Run directly
python3 ava.py --test-list
python3 ava.py --test qp_bounds --encoder c2.qti.hevc.encoder -s <serial_number>
```

### Option 2: Module Execution (PYTHONPATH)

To use `python3 -m ava`, add the ava directory to your PYTHONPATH:

```bash
# Add to PYTHONPATH
export PYTHONPATH="/path/to/ava:$PYTHONPATH"

# Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export PYTHONPATH="/path/to/ava:$PYTHONPATH"' >> ~/.bashrc

# Then use as module
python3 -m ava --test-list
python3 -m ava --test qp_bounds --encoder c2.qti.hevc.encoder -s <serial_number>
```

### Option 3: Package Installation (Recommended)

For the best experience, install AVA as a Python package:

```bash
# Navigate to the ava directory
cd ava

# Install in development mode (editable)
pip install -e .

# Or install normally
pip install .

# Then use the ava command directly
ava --test-list
ava --test qp_bounds --encoder c2.qti.hevc.encoder -s <serial_number>
```

## Dependencies

### Required Dependencies
- Python 3.7+
- FFmpeg (for video processing)
- Android SDK/ADB (for device communication)
- encapp (included in lib/encapp)

### Optional Dependencies
Install these for enhanced functionality:

```bash
pip install -r requirements.txt
```

This installs:
- `plotly` - For interactive plotting and reporting
- `pandas` - For data analysis
- `numpy` - For numerical computations

## Verification

Test that everything works:

```bash
# Test module import
python3 test_module.py

# Test basic functionality
python3 -m ava --help
python3 -m ava --test-list
python3 -m ava --scenarios
```

## Troubleshooting

### Import Errors
If you get import errors, make sure:
1. The python directory is in your PYTHONPATH
2. All required dependencies are installed
3. The encapp library is built and available

### Device Connection Issues
If device discovery fails:
1. Ensure ADB is installed and in your PATH
2. Enable USB debugging on your Android device
3. Check device connection with `adb devices`

### Video Processing Issues
If video generation/processing fails:
1. Ensure FFmpeg is installed and in your PATH
2. Check available disk space
3. Verify input video files are valid

## Usage Examples

```bash
# List all available tests
python3 -m ava --test-list

# Run a specific test
python3 -m ava --test test_bitrate_mode --encoder c2.qti.hevc.encoder -s device_serial

# Run all tests with a specific encoder
python3 -m ava --encoder hevc

# Generate video sources
python3 -m ava --generate-sources

# List test scenarios
python3 -m ava --scenarios

# Run with debug output
python3 -m ava --test test_motion --debug

# Run with specific input files
python3 -m ava --test test_quality --input-file video1.mp4 --input-file video2.mp4
```

## Development

For development, install in editable mode:

```bash
cd python
pip install -e .
```

This allows you to modify the code and see changes immediately without reinstalling.
