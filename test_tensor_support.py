#!/usr/bin/env python3
"""
Test script for MS-SWIFT tensor support functionality.
This script tests the tensor loading and processing capabilities.
"""

import tempfile
import numpy as np
import torch
import os
from swift.llm.template.template_inputs import StdTemplateInputs, InferRequest
from swift.llm.template.vision_utils import load_tensor_npy, load_tensor_pt

def create_test_tensors():
    """Create test tensor files (.npy and .pt)"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create test numpy array and save as .npy
    test_array = np.random.rand(10, 5, 3).astype(np.float32)
    npy_path = os.path.join(temp_dir, "test_tensor.npy")
    np.save(npy_path, test_array)
    
    # Create test torch tensor and save as .pt
    test_tensor = torch.randn(8, 4, 2)
    pt_path = os.path.join(temp_dir, "test_tensor.pt")
    torch.save(test_tensor, pt_path)
    
    return npy_path, pt_path, test_array, test_tensor

def test_tensor_loading():
    """Test tensor loading functions"""
    print("Testing tensor loading functions...")
    
    npy_path, pt_path, original_array, original_tensor = create_test_tensors()
    
    # Test .npy loading
    loaded_array = load_tensor_npy(npy_path)
    assert np.allclose(loaded_array, original_array), "NPY loading failed"
    print("✓ NPY tensor loading works correctly")
    
    # Test .pt loading
    loaded_tensor = load_tensor_pt(pt_path)
    assert torch.allclose(loaded_tensor, original_tensor), "PT loading failed"
    print("✓ PT tensor loading works correctly")
    
    # Clean up
    os.remove(npy_path)
    os.remove(pt_path)
    os.rmdir(os.path.dirname(npy_path))
    
    return npy_path, pt_path

def test_template_inputs():
    """Test StdTemplateInputs and InferRequest with tensors"""
    print("\nTesting template inputs with tensors...")
    
    # Create test tensor paths
    npy_path, pt_path, _, _ = create_test_tensors()
    
    # Test StdTemplateInputs
    inputs = StdTemplateInputs(
        messages=[{"role": "user", "content": "<tensor>What can you tell me about this tensor?"}],
        tensors=[npy_path, pt_path]
    )
    
    assert inputs.tensors == [npy_path, pt_path], "StdTemplateInputs tensor field failed"
    assert inputs.is_multimodal == True, "is_multimodal should be True with tensors"
    assert inputs.tensor_idx == 0, "tensor_idx should be initialized to 0"
    print("✓ StdTemplateInputs tensor support works")
    
    # Test InferRequest
    request = InferRequest(
        messages=[{"role": "user", "content": "<tensor><tensor>Compare these two tensors"}],
        tensors=[npy_path, pt_path]
    )
    
    assert request.tensors == [npy_path, pt_path], "InferRequest tensor field failed"
    assert request.is_multimodal == True, "InferRequest is_multimodal should be True with tensors"
    print("✓ InferRequest tensor support works")
    
    # Clean up
    os.remove(npy_path)
    os.remove(pt_path)
    os.rmdir(os.path.dirname(npy_path))

def test_message_processing():
    """Test message processing with tensor tags"""
    print("\nTesting message processing with tensor tags...")
    
    # Create test data
    npy_path, pt_path, _, _ = create_test_tensors()
    
    inputs = StdTemplateInputs(
        messages=[
            {
                "role": "user", 
                "content": "Here is my tensor data: <tensor>"
            },
            {
                "role": "assistant",
                "content": "I can see the tensor data you provided."
            }
        ],
        tensors=[npy_path]
    )
    
    # Test that multimodal detection works
    assert inputs.is_multimodal, "Should detect tensor as multimodal"
    print("✓ Tensor messages detected as multimodal")
    
    # Clean up
    os.remove(npy_path)
    os.remove(pt_path)
    os.rmdir(os.path.dirname(npy_path))

def test_video_to_tensor_scenario():
    """Test scenario where video is converted to tensor format"""
    print("\nTesting video-to-tensor conversion scenario...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Simulate video feature extraction to tensor
    # This simulates extracting features from an MP4 video into a tensor
    # For example: video frames -> feature vectors -> tensor
    
    # Create a tensor that represents extracted video features
    # Shape: (num_frames, feature_dim) - common for video feature extraction
    num_frames = 30  # 1 second of 30fps video
    feature_dim = 512  # common feature dimension
    video_features = np.random.rand(num_frames, feature_dim).astype(np.float32)
    
    # Save as both .npy and .pt formats
    npy_path = os.path.join(temp_dir, "video_features.npy")
    pt_path = os.path.join(temp_dir, "video_features.pt")
    
    np.save(npy_path, video_features)
    torch.save(torch.from_numpy(video_features), pt_path)
    
    # Test loading the video-derived tensors
    loaded_npy = load_tensor_npy(npy_path)
    loaded_pt = load_tensor_pt(pt_path)
    
    assert np.allclose(loaded_npy, video_features), "Video-derived NPY tensor loading failed"
    assert torch.allclose(loaded_pt, torch.from_numpy(video_features)), "Video-derived PT tensor loading failed"
    print("✓ Video-derived tensor loading works correctly")
    
    # Test using these tensors in template inputs
    video_tensor_inputs = StdTemplateInputs(
        messages=[
            {
                "role": "user",
                "content": "I have extracted features from a video and saved them as tensors: <tensor><tensor>. Can you analyze these video features?"
            }
        ],
        tensors=[npy_path, pt_path]
    )
    
    assert video_tensor_inputs.is_multimodal, "Video-derived tensors should be detected as multimodal"
    assert len(video_tensor_inputs.tensors) == 2, "Should have 2 tensor files"
    print("✓ Video-derived tensors work with template inputs")
    
    # Test with InferRequest as well
    video_infer_request = InferRequest(
        messages=[
            {
                "role": "user",
                "content": "Analyze this video feature tensor: <tensor>"
            }
        ],
        tensors=[npy_path]
    )
    
    assert video_infer_request.is_multimodal, "Video-derived tensor should be detected as multimodal in InferRequest"
    print("✓ Video-derived tensors work with InferRequest")
    
    # Simulate a realistic video processing workflow
    print("✓ Video-to-tensor conversion workflow simulation:")
    print(f"  - Simulated video features shape: {video_features.shape}")
    print(f"  - Saved as .npy file: {os.path.basename(npy_path)}")
    print(f"  - Saved as .pt file: {os.path.basename(pt_path)}")
    print("  - Successfully loaded and processed by MS-SWIFT tensor system")
    
    # Clean up
    os.remove(npy_path)
    os.remove(pt_path)
    os.rmdir(temp_dir)

def main():
    """Run all tests"""
    print("Running MS-SWIFT Tensor Support Tests")
    print("=" * 40)
    
    try:
        test_tensor_loading()
        test_template_inputs()
        test_message_processing()
        test_video_to_tensor_scenario()
        
        print("\n" + "=" * 40)
        print("🎉 All tests passed! Tensor support is working correctly.")
        print("\nKey features tested:")
        print("- ✓ Loading .npy and .pt tensor files")
        print("- ✓ StdTemplateInputs tensor field support")
        print("- ✓ InferRequest tensor field support")
        print("- ✓ Multimodal detection with tensors")
        print("- ✓ Message processing with <tensor> tags")
        print("- ✓ Video-to-tensor conversion workflow")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
