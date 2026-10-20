"""Signed metric-aware parameter-edit score (paper Eqs. 18–19).

Only activated-opacity edits are implemented. All derivatives are first-order
VJPs through the ordinary rasterizer, using torch.autograd.grad.
"""
import math
from functools import lru_cache
import torch


def add_score_arguments(parser):
    parser.add_argument('--score_metric', choices=('mse', 'lpips', 'ssim', 'clip'), default='mse')
    parser.add_argument('--num_probes', type=int, default=64)
    parser.add_argument('--score_seed', type=int, default=0)
    parser.add_argument('--target_opacity', type=float, default=1e-4)
    parser.add_argument('--scale_log_delta', type=float, default=-3.)
    parser.add_argument('--prune_only', action='store_true', help='Save after the first prune without finetuning')


def validate_pruning_schedule(args, first_iter, iterations):
    if len(args.prune_iterations) != len(args.prune_percent):
        raise ValueError('Provide one prune_percent for each prune iteration')
    if not args.prune_iterations or args.prune_iterations != sorted(set(args.prune_iterations)):
        raise ValueError('prune_iterations must be nonempty, unique and increasing')
    if any(not first_iter < i <= iterations for i in args.prune_iterations):
        raise ValueError('Pruning iterations must follow the input model and be within iterations')
    if any(not 0 <= value < 1 for value in args.prune_percent):
        raise ValueError('prune_percent must be in [0, 1)')
    if args.num_probes < 1 or not 0 <= args.target_opacity < 1 or not math.isfinite(args.scale_log_delta):
        raise ValueError('Invalid sketch dimension or pruning displacement')


@lru_cache(maxsize=2)
def _perceptual(device):
    from lpipsPyTorch.modules.lpips import LPIPS
    return LPIPS('vgg').to(device).eval().requires_grad_(False)


def feature_residual(image, target, metric):
    if metric == 'mse':
        return image - target
    if metric == 'ssim':
        from utils.loss_utils import ssim
        return 1 - ssim(image, target, return_map=True)
    if metric == 'lpips':
        criterion = _perceptual(str(image.device))
        features = criterion.net(image[None])
        with torch.no_grad():
            reference = criterion.net(target[None])
        return torch.cat([lin(x-y).flatten() for x, y, lin in zip(features, reference, criterion.lin)])
    if metric == 'clip':
        from clip_score import clip_margin_map
        feature = clip_margin_map(image)
        with torch.no_grad():
            reference = clip_margin_map(target)
        return feature - reference
    raise ValueError('Unknown metric: ' + metric)


def sketched_edit_score(residual, parameters, deltas, num_probes, generator):
    """2 r^T J delta + ||S J delta||^2, evaluated per Gaussian.

    Stream normalized Rademacher probes to bound memory. Preserve signed scores
    and existing optimizer .grad buffers; no full Jacobian or double backward.
    """
    def response(vector, retain_graph):
        gradients = torch.autograd.grad(residual, parameters, grad_outputs=vector,
                                        retain_graph=retain_graph, create_graph=False)
        return sum((gradient * delta).flatten(1).sum(1)
                   for gradient, delta in zip(gradients, deltas))
    scores = response(2 * residual.detach(), True).detach()
    for index in range(num_probes):
        probe = torch.empty_like(residual).bernoulli_(.5, generator=generator)
        probe.mul_(2).sub_(1).div_(math.sqrt(num_probes))
        scores.add_(response(probe, index + 1 < num_probes).detach().square())
    return scores


@torch.enable_grad()
def compute_pruning_scores(gaussians, cameras, pipe, background, args):
    from gaussian_renderer import render
    from tqdm import tqdm
    cameras = list(cameras)
    if not cameras:
        raise ValueError('At least one training view is required')
    device = gaussians.get_xyz.device
    generator = torch.Generator(device=device).manual_seed(args.score_seed)
    scores = torch.zeros(len(gaussians.get_xyz), device=device)
    for camera in tqdm(cameras, desc='Metric-aware scoring (' + args.score_metric + ')'):
        # A leaf in activated-opacity space avoids dividing by a saturated
        # sigmoid derivative. The original Gaussian parameters stay unchanged.
        alpha = gaussians.get_opacity.detach().requires_grad_(True)
        image = render(camera, gaussians, pipe, background, override_opacity=alpha)['render']
        target = camera.original_image.to(device=device, dtype=image.dtype).detach()
        residual = feature_residual(image, target, args.score_metric)
        deltas = (args.target_opacity - alpha.detach(),
                  torch.full_like(gaussians._scaling, args.scale_log_delta))
        scores.add_(sketched_edit_score(residual, (alpha, gaussians._scaling),
                                       deltas, args.num_probes, generator))
    scores.div_(len(cameras))
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite metric-aware pruning scores')
    return scores
