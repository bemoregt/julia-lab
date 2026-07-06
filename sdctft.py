"""
sDCTFT for SAM — arXiv:2410.09103
Dataset: Kvasir-SEG (polyp segmentation, 1000 images, ~44 MB)
"""
import os, zipfile, ssl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dctn, idctn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ─── 1. Differentiable iDCT via custom autograd ───────────────────────────────

class _SDCTFTFunc(torch.autograd.Function):
    """
    Forward : delta_coeff (sparse DCT 계수) → iDCT → ΔW → W_base + ΔW
    Backward: grad_W → DCT(grad_W) → selected indices 로 grad_delta_coeff 반환
    """
    @staticmethod
    def forward(ctx, delta_coeff, W_base, sel_np, M, N):
        ctx.save_for_backward(torch.tensor(sel_np, dtype=torch.long))
        ctx.M, ctx.N = M, N
        ctx.in_device = delta_coeff.device

        delta_F = np.zeros(M * N, dtype=np.float64)
        delta_F[sel_np] = delta_coeff.detach().cpu().numpy().astype(np.float64)
        delta_W = torch.tensor(
            idctn(delta_F.reshape(M, N), norm='ortho'), dtype=torch.float32
        )
        return W_base.cpu() + delta_W   # CPU tensor (moved to device later)

    @staticmethod
    def backward(ctx, grad_W):
        sel, = ctx.saved_tensors
        grad_W_F = dctn(
            grad_W.detach().cpu().numpy().astype(np.float64), norm='ortho'
        )
        grad_flat = torch.tensor(grad_W_F.flatten(), dtype=torch.float32)
        grad_delta = grad_flat[sel].to(ctx.in_device)
        return grad_delta, None, None, None, None


# ─── 2. Drop-in sDCTFT replacement for nn.Linear ──────────────────────────────

class SDCTFTLinear(nn.Module):
    def __init__(self, linear: nn.Linear, n_select: int, delta_ratio: float):
        super().__init__()
        W = linear.weight.detach().float().cpu()
        self.M, self.N = W.shape
        self.bias = linear.bias   # keeps original bias (frozen)

        W_F = dctn(W.numpy().astype(np.float64), norm='ortho')
        sel = self._select(W_F, n_select, delta_ratio)
        self.sel_np = np.array(sel, dtype=np.int64)

        self.register_buffer('W_base', W)
        self.delta_coeff = nn.Parameter(torch.zeros(len(sel), dtype=torch.float32))

    def _select(self, W_F, n_select, delta_ratio):
        M, N   = W_F.shape
        d_max  = np.sqrt((M-1)**2 + (N-1)**2) + 1e-8
        ii     = np.arange(M)[:, None];  jj = np.arange(N)[None, :]
        dist   = np.sqrt(ii**2 + jj**2).flatten()
        idx    = np.arange(M * N)
        W_flat = np.abs(W_F.flatten())
        n_per  = max(1, n_select // 3)
        parts  = []
        for part in [idx[dist <= d_max/3],
                     idx[(dist > d_max/3) & (dist <= 2*d_max/3)],
                     idx[dist > 2*d_max/3]]:
            if not len(part): continue
            n_p   = min(len(part), n_per)
            n_e   = int(n_p * delta_ratio)
            order = np.argsort(W_flat[part])[::-1]
            e_sel = part[order[:n_e]]
            mask  = np.ones(len(part), dtype=bool); mask[order[:n_e]] = False
            rem   = part[mask]
            n_r   = min(n_p - n_e, len(rem))
            r_sel = np.random.choice(rem, n_r, replace=False) if n_r > 0 else np.array([], dtype=int)
            parts += [e_sel, r_sel]
        combined = np.concatenate(parts).astype(int)
        _, keep  = np.unique(combined, return_index=True)
        return combined[np.sort(keep)][:n_select].tolist()

    def forward(self, x):
        W_eff = _SDCTFTFunc.apply(
            self.delta_coeff, self.W_base, self.sel_np, self.M, self.N
        )
        return F.linear(x, W_eff.to(x.device), self.bias)


# ─── 3. Apply sDCTFT to SAM image encoder ─────────────────────────────────────

def apply_sdctft(sam, n_select=700, delta_ratio=0.7, max_blocks=12):
    for p in sam.parameters():
        p.requires_grad_(False)

    replaced = 0
    for b_idx, block in enumerate(list(sam.image_encoder.blocks)[:max_blocks]):
        for attr in ['attn.qkv', 'attn.proj', 'mlp.lin1', 'mlp.lin2']:
            parent = block
            parts  = attr.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p, None)
                if parent is None: break
            if parent is None: continue
            child = getattr(parent, parts[-1], None)
            if not isinstance(child, nn.Linear): continue
            if min(child.weight.shape) < 16: continue
            setattr(parent, parts[-1], SDCTFTLinear(child, n_select, delta_ratio))
            replaced += 1

    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    total_sam = sum(p.numel() for p in sam.parameters())
    print(f"sDCTFT 레이어: {replaced}개 | 학습 파라미터: {trainable:,} "
          f"({100*trainable/total_sam:.4f}% of SAM)")
    return sam


# ─── 4. Kvasir-SEG dataset ────────────────────────────────────────────────────

def download_kvasir(root='./data'):
    url     = 'https://datasets.simula.no/downloads/kvasir-seg.zip'
    seg_dir = os.path.join(root, 'Kvasir-SEG')
    if os.path.isdir(seg_dir):
        print(f"Kvasir-SEG 이미 존재: {seg_dir}")
        return root
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, 'kvasir-seg.zip')
    print(f"Kvasir-SEG 다운로드 중...")
    import urllib.request
    ctx = ssl.create_default_context()
    ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as r, open(zip_path, 'wb') as f:
        total = int(r.headers.get('content-length', 0))
        done  = 0
        while chunk := r.read(1 << 20):
            f.write(chunk); done += len(chunk)
            if total:
                print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB", end='', flush=True)
    print(f"\n압축 해제 중...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(root)
    os.remove(zip_path)
    print(f"완료: {seg_dir}")
    return root


class KvasirSEG(Dataset):
    IMG_MEAN = torch.tensor([123.675, 116.28,  103.53 ]).view(3,1,1)
    IMG_STD  = torch.tensor([ 58.395,  57.12,   57.375]).view(3,1,1)

    def __init__(self, root, img_size=1024, max_samples=None):
        from segment_anything.utils.transforms import ResizeLongestSide
        self.rsz      = ResizeLongestSide(img_size)
        self.img_size = img_size

        img_dir  = os.path.join(root, 'Kvasir-SEG', 'images')
        mask_dir = os.path.join(root, 'Kvasir-SEG', 'masks')

        imgs  = sorted(f for f in os.listdir(img_dir)  if f.lower().endswith(('.jpg','.png')))
        masks = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg','.png')))
        if max_samples:
            imgs  = imgs[:max_samples]
            masks = masks[:max_samples]

        self.samples = [(os.path.join(img_dir, i), os.path.join(mask_dir, m))
                        for i, m in zip(imgs, masks)]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_p, msk_p = self.samples[idx]
        image = np.array(Image.open(img_p).convert('RGB'))
        mask  = np.array(Image.open(msk_p).convert('L'))

        # SAM 전처리
        img_r  = self.rsz.apply_image(image)
        img_t  = torch.as_tensor(img_r).permute(2,0,1).float()
        h, w   = img_t.shape[-2:]
        img_t  = F.pad(img_t, (0, self.img_size-w, 0, self.img_size-h))
        img_t  = (img_t - self.IMG_MEAN) / self.IMG_STD    # (3, 1024, 1024)

        # 마스크 → 256×256 (mask decoder 출력 크기)
        msk_pil = Image.fromarray(mask)
        msk_256 = transforms.functional.resize(
            msk_pil, (256, 256), transforms.InterpolationMode.NEAREST)
        msk_t = (transforms.functional.to_tensor(msk_256) > 0.5).float()  # (1, 256, 256)

        # GT bounding box (prompt)
        msk_1k = np.array(transforms.functional.resize(
            msk_pil, (self.img_size, self.img_size),
            transforms.InterpolationMode.NEAREST)) > 127
        if msk_1k.any():
            rows, cols = np.where(msk_1k)
            bbox = torch.tensor(
                [cols.min(), rows.min(), cols.max(), rows.max()], dtype=torch.float32)
        else:
            bbox = torch.tensor([0, 0, self.img_size-1, self.img_size-1], dtype=torch.float32)

        return img_t, msk_t, bbox


# ─── 5. Loss ──────────────────────────────────────────────────────────────────

def dice_loss(pred, target, eps=1e-6):
    p     = torch.sigmoid(pred)
    inter = (p * target).sum(dim=(-2,-1))
    denom = p.sum(dim=(-2,-1)) + target.sum(dim=(-2,-1)) + eps
    return (1 - 2*inter/denom).mean()

def seg_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target) + dice_loss(pred, target)


# ─── 6. Main pipeline ─────────────────────────────────────────────────────────

def build_and_finetune(
    checkpoint_path,
    data_root   = '/Users/sgi/julia-mcp/data',
    n_select    = 700,
    delta_ratio = 0.7,
    max_blocks  = 12,
    n_steps     = 10,
    max_samples = 50,
    lr          = 1e-3,
    save_path   = None,
):
    from segment_anything import sam_model_registry

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"디바이스: {device}\n")

    # ── 데이터셋 ──
    data_root = download_kvasir(data_root)
    dataset   = KvasirSEG(data_root, img_size=1024, max_samples=max_samples)
    loader    = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"Kvasir-SEG: {len(dataset)}개 샘플 로드\n")

    # ── 모델 ──
    print("SAM ViT-B 로딩...")
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    print("sDCTFT 적용 중...")
    sam = apply_sdctft(sam, n_select, delta_ratio, max_blocks)
    sam.to(device).train()

    optimizer = torch.optim.AdamW(
        [p for p in sam.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # ── 학습 루프 (multi-epoch: n_steps 달성까지 데이터셋 순환) ──
    print(f"\n{'─'*55}")
    print(f"파인튜닝 시작 | steps={n_steps} | lr={lr}")
    print(f"{'─'*55}")

    losses, step = [], 0
    while step < n_steps:
        for images, gt_masks, bboxes in loader:
            if step >= n_steps: break
            images   = images.to(device)
            gt_masks = gt_masks.to(device)
            bboxes   = bboxes.to(device)

            optimizer.zero_grad()

            img_emb = sam.image_encoder(images)
            sparse, dense = sam.prompt_encoder(
                points=None, boxes=bboxes.unsqueeze(1), masks=None)
            low_masks, iou_pred = sam.mask_decoder(
                image_embeddings         = img_emb,
                image_pe                 = sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings = sparse,
                dense_prompt_embeddings  = dense,
                multimask_output         = False,
            )

            loss = seg_loss(low_masks, gt_masks)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            losses.append(loss.item())
            if step % 10 == 0 or step == 1:
                print(f"  Step {step:>3}/{n_steps} | loss={loss.item():.4f} "
                      f"| iou={iou_pred.mean().item():.4f}")

    print(f"{'─'*55}")
    print(f"평균 손실: {np.mean(losses):.4f}  최소: {min(losses):.4f}")

    # ── 저장 ──
    sam.eval()
    if save_path is None:
        save_path = checkpoint_path.replace('.pth', '_sdctft_kvasir.pth')
    torch.save(sam.state_dict(), save_path)
    print(f"\n모델 저장 완료: {save_path}")
    print(f"파일 크기: {os.path.getsize(save_path)/1e6:.1f} MB")
    return losses


# ─── 7. Inference with 75% bbox prompt ────────────────────────────────────────

def run_inference(
    orig_checkpoint,
    finetuned_checkpoint,
    data_root    = '/Users/sgi/julia-mcp/data',
    output_dir   = '/Users/sgi/julia-mcp/outputs/inference',
    n_images     = 10,
    bbox_scale   = 0.75,
    n_select     = 700,
    delta_ratio  = 0.7,
    max_blocks   = 12,
    train_offset = 50,   # training set size to skip
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    os.makedirs(output_dir, exist_ok=True)

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"Device: {device}")

    # ── 파인튜닝 모델 로드 ──
    print("Loading fine-tuned SAM …")
    sam = sam_model_registry["vit_b"](checkpoint=orig_checkpoint)
    sam = apply_sdctft(sam, n_select, delta_ratio, max_blocks)
    state = torch.load(finetuned_checkpoint, map_location='cpu')
    sam.load_state_dict(state)
    sam.to(device).eval()
    print("Model ready.\n")

    # ── 테스트 이미지 목록 (학습 구간 이후) ──
    img_dir  = os.path.join(data_root, 'Kvasir-SEG', 'images')
    msk_dir  = os.path.join(data_root, 'Kvasir-SEG', 'masks')
    all_imgs  = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png')))
    all_masks = sorted(f for f in os.listdir(msk_dir) if f.lower().endswith(('.jpg','.png')))
    test_imgs  = all_imgs [train_offset : train_offset + n_images]
    test_masks = all_masks[train_offset : train_offset + n_images]

    rsz      = ResizeLongestSide(1024)
    IMG_MEAN = torch.tensor([123.675, 116.28,  103.53]).view(3,1,1)
    IMG_STD  = torch.tensor([ 58.395,  57.12,   57.375]).view(3,1,1)
    results  = []

    for i, (fname, mname) in enumerate(zip(test_imgs, test_masks)):
        img_np  = np.array(Image.open(os.path.join(img_dir, fname)).convert('RGB'))
        mask_np = np.array(Image.open(os.path.join(msk_dir, mname)).convert('L'))
        H, W    = img_np.shape[:2]

        # SAM 전처리
        img_r  = rsz.apply_image(img_np)
        h_r, w_r = img_r.shape[:2]
        img_t  = torch.as_tensor(img_r).permute(2,0,1).float()
        img_t  = F.pad(img_t, (0, 1024-w_r, 0, 1024-h_r))
        img_t  = ((img_t - IMG_MEAN) / IMG_STD).unsqueeze(0).to(device)

        # GT bbox (1024 좌표계)
        msk_1k = np.array(Image.fromarray(mask_np).resize((1024,1024), Image.NEAREST)) > 127
        if msk_1k.any():
            ry, rx = np.where(msk_1k)
            gt_bbox = np.array([rx.min(), ry.min(), rx.max(), ry.max()], dtype=np.float32)
        else:
            gt_bbox = np.array([0., 0., 1023., 1023.], dtype=np.float32)

        # 75% 바운딩박스: 중심 유지, 가로·세로 75% 축소
        cx = (gt_bbox[0] + gt_bbox[2]) / 2
        cy = (gt_bbox[1] + gt_bbox[3]) / 2
        bw = (gt_bbox[2] - gt_bbox[0]) * bbox_scale
        bh = (gt_bbox[3] - gt_bbox[1]) * bbox_scale
        prompt_bbox = np.array([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], dtype=np.float32)

        # SAM 추론
        bbox_t = torch.tensor(prompt_bbox).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = sam.image_encoder(img_t)
            sparse, dense = sam.prompt_encoder(points=None, boxes=bbox_t, masks=None)
            low_masks, iou_pred = sam.mask_decoder(
                image_embeddings         = img_emb,
                image_pe                 = sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings = sparse,
                dense_prompt_embeddings  = dense,
                multimask_output         = False,
            )

        # 원본 크기로 복원
        pred = sam.postprocess_masks(low_masks, input_size=(h_r, w_r), original_size=(H, W))
        pred_mask = (torch.sigmoid(pred[0,0]) > 0.5).cpu().numpy()
        gt_mask   = mask_np > 127

        # IoU 계산
        inter = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        iou   = float(inter) / (float(union) + 1e-6)

        # 1024 → 원본 스케일 변환 계수
        sx, sy = W / 1024, H / 1024

        # ── 시각화 (3-panel) ──
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#1a1a2e')

        def show_mask(ax, mask, color):
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = color
            ax.imshow(overlay)

        def draw_rect(ax, bbox_1024, color, lw=2, ls='-'):
            x0,y0,x1,y1 = bbox_1024
            r = patches.Rectangle(
                (x0*sx, y0*sy), (x1-x0)*sx, (y1-y0)*sy,
                linewidth=lw, edgecolor=color, facecolor='none', linestyle=ls)
            ax.add_patch(r)

        # Panel 1 — GT
        axes[0].imshow(img_np)
        show_mask(axes[0], gt_mask,   [0.2, 0.5, 1.0, 0.45])
        draw_rect(axes[0], gt_bbox,   'deepskyblue', lw=2)
        axes[0].set_title('GT mask + GT BBox (100%)', color='white', fontsize=10)
        axes[0].axis('off')

        # Panel 2 — Prediction
        axes[1].imshow(img_np)
        show_mask(axes[1], pred_mask, [0.1, 0.9, 0.3, 0.45])
        draw_rect(axes[1], gt_bbox,   'deepskyblue', lw=1, ls='--')   # GT 참고용
        draw_rect(axes[1], prompt_bbox,'orange',     lw=2)
        axes[1].set_title(f'Pred mask + 75% BBox prompt  (IoU={iou:.3f})',
                          color='white', fontsize=10)
        axes[1].axis('off')

        # Panel 3 — TP/FP/FN
        axes[2].imshow(img_np)
        tp_fp_fn = np.zeros((*gt_mask.shape, 4))
        tp_fp_fn[ gt_mask &  pred_mask] = [0.0, 1.0, 0.3, 0.55]   # TP green
        tp_fp_fn[~gt_mask &  pred_mask] = [1.0, 0.2, 0.2, 0.55]   # FP red
        tp_fp_fn[ gt_mask & ~pred_mask] = [0.2, 0.4, 1.0, 0.55]   # FN blue
        axes[2].imshow(tp_fp_fn)
        axes[2].set_title('TP(green) / FP(red) / FN(blue)', color='white', fontsize=10)
        axes[2].axis('off')

        for ax in axes:
            ax.set_facecolor('#1a1a2e')
        fig.suptitle(
            f'[{i+1:02d}] {fname}   IoU={iou:.3f}   SAM IoU pred={iou_pred.item():.3f}',
            color='white', fontsize=11, y=1.01
        )
        plt.tight_layout()

        stem = os.path.splitext(fname)[0]
        out  = os.path.join(output_dir, f'{i+1:02d}_{stem}_result.png')
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        results.append({'file': fname, 'iou': iou, 'iou_pred': iou_pred.item()})
        print(f"  [{i+1:02d}] {fname:<40s} IoU={iou:.3f}  SAM_iou={iou_pred.item():.3f}  -> {os.path.basename(out)}")

    mean_iou = float(np.mean([r['iou'] for r in results]))
    print(f"\n{'─'*55}")
    print(f"저장 위치 : {output_dir}")
    print(f"평균 IoU  : {mean_iou:.3f}")
    return results, mean_iou
