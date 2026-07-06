"""
standalone training script — runs outside MCP timeout
Usage: python train_sdctft.py
"""
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, '/Users/sgi/julia-mcp')
import sdctft

losses = sdctft.build_and_finetune(
    checkpoint_path = '/Users/sgi/julia-mcp/sam_vit_b_01ec64.pth',
    data_root       = '/Users/sgi/julia-mcp/data',
    n_select        = 700,
    delta_ratio     = 0.7,
    max_blocks      = 12,
    n_steps         = 1000,
    max_samples     = 50,
    lr              = 1e-3,
    save_path       = '/Users/sgi/julia-mcp/sam_vit_b_sdctft_kvasir_1000steps.pth',
)

import numpy as np
print(f"\n=== 학습 완료 ===")
print(f"초반(1-10)    평균 손실: {np.mean(losses[:10]):.4f}")
print(f"후반(991-1000) 평균 손실: {np.mean(losses[990:]):.4f}")
print("DONE")
