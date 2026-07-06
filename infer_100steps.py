"""
Inference with 100-step fine-tuned model
"""
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, '/Users/sgi/julia-mcp')
import sdctft

results, mean_iou = sdctft.run_inference(
    orig_checkpoint      = '/Users/sgi/julia-mcp/sam_vit_b_01ec64.pth',
    finetuned_checkpoint = '/Users/sgi/julia-mcp/sam_vit_b_sdctft_kvasir_1000steps.pth',
    data_root            = '/Users/sgi/julia-mcp/data',
    output_dir           = '/Users/sgi/julia-mcp/outputs/inference_1000steps',
    n_images             = 10,
    bbox_scale           = 0.75,
    n_select             = 700,
    delta_ratio          = 0.7,
    max_blocks           = 12,
    train_offset         = 50,
)

print(f"\n=== 추론 완료 ===")
print(f"평균 IoU: {mean_iou:.3f}")
for r in results:
    print(f"  {r['file']}: IoU={r['iou']:.3f}")
print("DONE")
