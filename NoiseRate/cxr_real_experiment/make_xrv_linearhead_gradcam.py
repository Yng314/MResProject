#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import torchxrayvision as xrv

from cxr_real_full_train_eval_cleanlab_xrv12 import LABEL_NAMES, create_model


def load_processed_image(image_root: Path, rel_image_path: str, image_size: int) -> tuple[np.ndarray, torch.Tensor]:
    image = Image.open(image_root / rel_image_path).convert("L")
    x = np.asarray(image)
    if x.max() <= 1.0:
        x = (x * 255).astype(np.uint8)
    if x.ndim == 2:
        x = x[np.newaxis, :, :]

    center_crop = xrv.datasets.XRayCenterCrop()
    resizer = xrv.datasets.XRayResizer(image_size)
    x = center_crop(x)
    x = resizer(x)
    x = xrv.datasets.normalize(x, maxval=255)
    if x.ndim == 2:
        x = x[np.newaxis, :, :]

    processed = x[0].astype(np.float32)
    tensor = torch.from_numpy(x).float().unsqueeze(0)
    return processed, tensor


def compute_gradcam(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_idx: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output.detach()

    def backward_hook(_module, _grad_inputs, grad_outputs):
        gradients["value"] = grad_outputs[0].detach()

    handle_fwd = model.base_model.features.register_forward_hook(forward_hook)
    handle_bwd = model.base_model.features.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x.to(device))
        score = logits[0, target_idx]
        score.backward()

        acts = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=False))[0]
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        prob = torch.sigmoid(score.detach()).item()
        return cam, prob
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def resize_cam(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, resample=Image.BILINEAR)
    return np.asarray(cam_img).astype(np.float32) / 255.0


def save_panel(
    image_gray: np.ndarray,
    cam: np.ndarray,
    label_name: str,
    prob: float,
    output_path: Path,
    caption: str | None,
) -> None:
    overlay = plt.cm.jet(cam)[..., :3]
    base = np.stack([image_gray, image_gray, image_gray], axis=-1)
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    blended = 0.55 * base + 0.45 * overlay
    blended = np.clip(blended, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(image_gray, cmap="gray")
    axes[0].set_title("Processed image")
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(blended)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")

    title = f"{label_name} | predicted prob = {prob:.4f}"
    if caption:
        title = f"{title}\n{caption}"
    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--label-name", type=str, required=True, choices=LABEL_NAMES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--xrv-cache-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        model_backbone=ckpt_args["model_backbone"],
        xrv_weights=checkpoint["xrv_weights"],
        xrv_cache_dir=args.xrv_cache_dir,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_gray, x = load_processed_image(args.image_root, args.image_path, args.image_size)
    cam_small, prob = compute_gradcam(model, x, LABEL_NAMES.index(args.label_name), device=device)
    cam = resize_cam(cam_small, size=(image_gray.shape[1], image_gray.shape[0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_panel(image_gray=image_gray, cam=cam, label_name=args.label_name, prob=prob, output_path=args.output, caption=args.caption)
    print(f"Saved Grad-CAM: {args.output}")


if __name__ == "__main__":
    main()
