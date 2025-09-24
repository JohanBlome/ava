def test_simple(device, input_file, workdir, test_data):
    """Simple test to verify basic functionality"""
    print(f"test_simple called with input_file: {input_file}")
    return {
        "success": True,
        "test_data": test_data,
        "output_files": []
    }
