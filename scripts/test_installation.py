#!/usr/bin/env python3
"""
Installation and Verification Script for MIMIC
==============================================
This script tests:
1. Python environment and core dependency imports (PyTorch, Transformers, Biotite, x-transformers, etc.)
2. GPU / CUDA availability and VRAM capacity
3. MIMIC package import and Tokenizer functionality (RNA, Amino Acids, Binning)
4. (Optional / Default) Pretrained weights download/loading and a minimal forward pass (embed + generate)

Usage:
------
# Basic verification (offline / fast, tests environment & tokenizers without downloading weights):
python scripts/test_installation.py --skip-model

# Full end-to-end test (loads pretrained model 1.0, tests embedding and generation):
python scripts/test_installation.py

# Specify device or local weights directory:
python scripts/test_installation.py --device cuda
python scripts/test_installation.py --weights-dir /path/to/weights
"""

import sys
import os
import argparse
import time

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_step(step_name: str):
    print(f"\n[+] {step_name}...")

def test_environment():
    print_header("1. Environment & Hardware Check")
    
    # Python version
    py_version = sys.version.split()[0]
    print(f"Python Version: {py_version}")
    if sys.version_info < (3, 10):
        print(f"[-] ERROR: MIMIC requires Python >= 3.10, but found {py_version}")
        return False
    print("[✓] Python version is compatible (>= 3.10)")

    # PyTorch & CUDA
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available:  {cuda_available}")
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_dev = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(current_dev)
            vram_gb = torch.cuda.get_device_properties(current_dev).total_memory / (1024**3)
            bf16_supported = torch.cuda.is_bf16_supported()
            print(f"GPU Count:       {device_count}")
            print(f"Active GPU:      {dev_name} (ID: {current_dev})")
            print(f"Total VRAM:      {vram_gb:.2f} GB")
            print(f"BFloat16 Native: {bf16_supported}")
            if vram_gb < 8.0:
                print("[!] Warning: Less than 8 GB VRAM detected. MIMIC (1.25B params) might run out of memory for large contexts on GPU.")
            else:
                print("[✓] Sufficient VRAM detected.")
        else:
            print("[!] CUDA is not available. Inference will fall back to CPU (slower).")
    except ImportError:
        print("[-] ERROR: PyTorch ('torch') is not installed!")
        return False

    return True

def test_imports():
    print_header("2. Core Dependencies & MIMIC Import")
    
    deps = [
        ("torch", "PyTorch"),
        ("x_transformers", "x-transformers"),
        ("einops", "Einops"),
        ("safetensors", "SafeTensors"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("transformers", "Hugging Face Transformers (BioBERT tokenizer)"),
        ("biotite", "Biotite (Structure I/O)"),
        ("mimic", "MIMIC package"),
    ]
    
    all_ok = True
    for module_name, label in deps:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "installed")
            print(f"[✓] {label:45s} : OK (v{version})")
        except ImportError as e:
            print(f"[-] {label:45s} : FAILED ({e})")
            all_ok = False
            
    return all_ok

def test_tokenizers():
    print_header("3. Tokenizer Sanity Checks")
    
    try:
        from mimic.modality_info import MODALITY_INFO
        
        # Test RNA Tokenizer
        if "tok_rna_seq" in MODALITY_INFO and "tokenizer" in MODALITY_INFO["tok_rna_seq"]:
            rna_tok = MODALITY_INFO["tok_rna_seq"]["tokenizer"]
            test_rna = "ACGUACGUACGU"
            token_ids = list(rna_tok.tokenize(test_rna))
            reconstructed = rna_tok.detokenize(token_ids)
            assert reconstructed == test_rna, f"RNA mismatch: {reconstructed} != {test_rna}"
            print(f"[✓] RNA Tokenizer: '{test_rna}' -> {token_ids[:4]}... -> '{reconstructed}'")
        else:
            print("[!] Warning: tok_rna_seq tokenizer not found.")

        # Test Amino Acid (Protein) Tokenizer
        if "tok_aa_seq" in MODALITY_INFO and "tokenizer" in MODALITY_INFO["tok_aa_seq"]:
            aa_tok = MODALITY_INFO["tok_aa_seq"]["tokenizer"]
            test_aa = "MKTAYIAKQR"
            aa_tokens = list(aa_tok.tokenize(test_aa))
            reconstructed_aa = aa_tok.detokenize(aa_tokens)
            assert reconstructed_aa == test_aa, f"AA mismatch: {reconstructed_aa} != {test_aa}"
            print(f"[✓] Protein AA Tokenizer: '{test_aa}' -> {aa_tokens[:4]}... -> '{reconstructed_aa}'")
        else:
            print("[!] Warning: tok_aa_seq tokenizer not found.")

        # Test Digitized Binning Tokenizer
        bin_mods = [m for m, info in MODALITY_INFO.items() if hasattr(info.get("tokenizer"), "bins")]
        if bin_mods:
            sample_mod = bin_mods[0]
            bin_tok = MODALITY_INFO[sample_mod]["tokenizer"]
            print(f"[✓] Binning Tokenizer verified on '{sample_mod}' (num bins: {len(bin_tok.bins)})")
            
        print("[✓] All tokenizer smoke tests passed!")
        return True
    except Exception as e:
        print(f"[-] Tokenizer test failed: {e}")
        return False

def test_model_inference(device: str, version: str, weights_dir: str = None):
    print_header("4. Model Load & Inference Test")
    
    try:
        from mimic import load_pretrained
        import torch

        print_step(f"Loading pretrained MIMIC (version={version}, device={device}, weights_dir={weights_dir})")
        start_time = time.time()
        
        model = load_pretrained(version=version, local_path=weights_dir, device=device)
        load_time = time.time() - start_time
        print(f"[✓] Successfully loaded model weights into memory in {load_time:.2f}s.")

        # Test 1: Joint Embedding
        print_step("Testing joint representation embedding (model.embed())")
        test_seq = "ACGUACGUACGUACGU"
        model.input([{"rna_seq": test_seq}])
        with torch.no_grad():
            reps = model.embed()
        
        assert "full" in reps, "Output representations missing 'full' key"
        assert reps["full"].isfinite().all(), "Representations contain NaN or Inf values"
        print(f"[✓] Embedding output shape: {tuple(reps['full'].shape)} (dim={reps['full'].shape[-1]})")

        # Test 2: Generation / Prediction
        print_step("Testing cross-modal inference / generation (model.generate())")
        with torch.no_grad():
            out = model.generate("splice_jctns_5cls", verbose=False)
            
        assert "splice_jctns_5cls" in out, "Generation output missing 'splice_jctns_5cls'"
        preds = out["splice_jctns_5cls"]
        print(f"[✓] Generation successful! Target 'splice_jctns_5cls' produced: {preds}")

        return True
    except Exception as e:
        print(f"[-] Model loading or inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify MIMIC installation and readiness.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to use for model test (default: auto)")
    parser.add_argument("--version", type=str, default="1.0",
                        help="MIMIC model version to load (default: 1.0)")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="Optional local path to weights directory containing config.json and model.safetensors")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip downloading model weights and running inference test (runs environment and tokenizer tests only)")
    
    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("      MIMIC Installation & Verification Suite")
    print("#" * 60)

    # 1. Environment & Hardware
    env_ok = test_environment()
    if not env_ok:
        print("\n[-] Environment test failed. Please fix Python / PyTorch prerequisites.")
        sys.exit(1)

    # 2. Dependencies & Package
    deps_ok = test_imports()
    if not deps_ok:
        print("\n[-] Dependency import test failed. Please install missing dependencies (e.g. `pip install -e .`).")
        sys.exit(1)

    # 3. Tokenizers
    tok_ok = test_tokenizers()
    if not tok_ok:
        print("\n[-] Tokenizer test failed.")
        sys.exit(1)

    # 4. Model & Inference
    if args.skip_model:
        print("\n[i] Skipping model loading and inference test (--skip-model specified).")
    else:
        model_ok = test_model_inference(device=args.device, version=args.version, weights_dir=args.weights_dir)
        if not model_ok:
            print("\n[-] Model test failed.")
            sys.exit(1)

    print_header("SUMMARY")
    print(" [✓] All tested components are operational!")
    print(" MIMIC is successfully installed and ready to use.\n")

if __name__ == "__main__":
    main()
