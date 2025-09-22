# ava: Tool to Understand Mobile Video Codecs

# 1. Introduction

ava is a tool that allows understanding mobile video codecs (both encoders and decoders). It works on both hardware and software encoders.

Ava can do:
* List advertised encoder features.
* Check actual support for different encoder features (either advertised or not) by analyzing the bitstream structure.
* Calculate RD-curves, which measure encoder quality as a function of bitrate (based on a dataset).

The ideal use case for ava is to design a test suite

Some ava goals:
* (1) easy to run: We should be able to define the test or test suite we want to run using a very simple set of instructions. This would allow directing a third-party to run the tests.
* (2) reproducibility
* (3) reliability: Ava should not crash without a clear message that can be reported easily.


# 2. Operation

(1) Get source code.
The project needs submodules to build.

```
$ git clone  --recurse-submodules https://github.com/chemag/ava.git
```

(2) Build the libraries.
```
$ cd ava
$ mkdir build
$ cmake build
$ make
```

(3) Install Python dependencies (optional, for enhanced functionality).
```
$ cd ava
$ pip install -r requirements.txt
```

(4) Run tests.

**Option A: Direct execution**
```
# List available tests
python3 ava/ava.py --test-list

# Run a specific test
python3 ava/ava.py --test qp_bounds --encoder c2.qti.hevc.encoder -s <serial_number>

# Run all tests with a specific encoder
python3 ava/ava.py --encoder c2.qti.hevc.encoder

# Generate video sources
python3 ava/ava.py --generate-sources

```

**Option B: Module execution (recommended)**
```
# List available tests
python3 -m ava --test-list

# Run a specific test
python3 -m ava --test qp_bounds --encoder c2.qti.hevc.encoder -s <serial_number>

# Run all tests with a specific encoder
python3 -m ava --encoder c2.qti.hevc.encoder

# Generate video sources
python3 -m ava --generate-sources

```

**Option C: Installed package (after pip install)**
```
# Install the package
cd ava
pip install -e .

# Then use the ava command directly
ava --test-list
ava --test qp_bounds --encoder c2.qti.hevc.encoder -s <serial_number>
```

# 3. New Testing Framework

The AVA testing framework has been redesigned to follow a three-step process:

1. **Data Collection**: Generate appropriate video sources for testing
2. **Data Transformation**: Run encoding tests and collect results  
3. **Analysis and Evaluation**: Assess quality and generate reports

## Key Features

- **Hierarchical Test Discovery**: Automatically finds tests following `test_*.py` and `test_*()` naming convention
- **Video Source Generation**: Creates test videos with specific characteristics (motion patterns, framerates, etc.)
- **Device Discovery**: Automatically detects connected devices and available encoders
- **Quality Assessment**: Integrates encapp_quality.py for comprehensive quality metrics
- **Interactive Reporting**: Generates Plotly-based interactive reports and static HTML reports
- **Progress Tracking**: Real-time progress display with ETA and success/failure counts

## Writing New Tests

See `python/TEST_WRITING_GUIDE.md` for detailed instructions on writing new tests.
