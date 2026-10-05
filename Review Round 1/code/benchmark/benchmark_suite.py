"""Baseline architectures, collocation and physical objectives."""

from __future__ import annotations

import argparse

import csv

import json

import math

import os

import time

from dataclasses import asdict, dataclass, fields

from pathlib import Path

from typing import Callable

import numpy as np

import torch

from torch import nn

DTYPE = torch.float32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L = 0.5

R = 0.1

E = 1.333

NU = 0.3333

LAM = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

MU = E / (2.0 * (1.0 + NU))

DOMAIN_AREA = (2.0 * L) ** 2 - math.pi * R**2

@dataclass
class SampleSet:
    domain: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    top: torch.Tensor
    bottom: torch.Tensor
    hole: torch.Tensor
    interface_v_low: torch.Tensor
    interface_v_high: torch.Tensor
    interface_h_left: torch.Tensor
    interface_h_right: torch.Tensor
    quad_domain: torch.Tensor
    quad_domain_weights: torch.Tensor
    quad_left: torch.Tensor
    quad_right: torch.Tensor
    quad_top: torch.Tensor
    quad_boundary_weights: torch.Tensor

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _rand_uniform(n: int, low: float, high: float, generator: torch.Generator) -> torch.Tensor:
    return torch.rand((n, 1), generator=generator, device=DEVICE, dtype=DTYPE) * (high - low) + low

def generate_samples(n_domain: int, n_boundary: int, seed: int) -> SampleSet:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(100_000 + seed)

    accepted = []
    remaining = n_domain
    while remaining > 0:
        candidate = (torch.rand((max(remaining * 2, 512), 2), generator=generator, device=DEVICE) * 2.0 - 1.0) * L
        keep = torch.linalg.vector_norm(candidate, dim=1) > R
        batch = candidate[keep][:remaining]
        accepted.append(batch)
        remaining -= batch.shape[0]
    domain = torch.cat(accepted, dim=0)

    y = _rand_uniform(n_boundary, -L, L, generator)
    left = torch.cat([torch.full_like(y, -L), y], dim=1)
    y = _rand_uniform(n_boundary, -L, L, generator)
    right = torch.cat([torch.full_like(y, L), y], dim=1)
    x = _rand_uniform(n_boundary, -L, L, generator)
    top = torch.cat([x, torch.full_like(x, L)], dim=1)
    x = _rand_uniform(n_boundary, -L, L, generator)
    bottom = torch.cat([x, torch.full_like(x, -L)], dim=1)

    theta = torch.linspace(0.0, 2.0 * math.pi, 2 * n_boundary + 1, device=DEVICE, dtype=DTYPE)[:-1]
    hole = torch.stack([R * torch.cos(theta), R * torch.sin(theta)], dim=1)

    # Interfaces exclude the circular void. Each segment receives n_boundary/2 points.
    n_half = max(n_boundary // 2, 2)
    y_low = _rand_uniform(n_half, -L, -R, generator)
    y_high = _rand_uniform(n_half, R, L, generator)
    x_left = _rand_uniform(n_half, -L, -R, generator)
    x_right = _rand_uniform(n_half, R, L, generator)
    zeros_y = torch.zeros_like(y_low)
    zeros_x = torch.zeros_like(x_left)

    # Tensor-product Gauss-Legendre rule for DEM. Its order gives approximately
    # the same number of in-domain integration points as n_domain.
    order = max(12, int(round(math.sqrt(n_domain / (1.0 - math.pi * R**2 / (2.0 * L) ** 2)))))
    nodes_np, weights_np = np.polynomial.legendre.leggauss(order)
    nodes = torch.tensor(nodes_np * L, device=DEVICE, dtype=DTYPE)
    weights_1d = torch.tensor(weights_np * L, device=DEVICE, dtype=torch.float64)
    gx, gy = torch.meshgrid(nodes, nodes, indexing="ij")
    wx, wy = torch.meshgrid(weights_1d, weights_1d, indexing="ij")
    quad_domain = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    quad_domain_weights = (wx * wy).reshape(-1)
    outside = torch.linalg.vector_norm(quad_domain, dim=1) > R
    quad_domain = quad_domain[outside]
    quad_domain_weights = quad_domain_weights[outside]
    quad_left = torch.stack([torch.full_like(nodes, -L), nodes], dim=1)
    quad_right = torch.stack([torch.full_like(nodes, L), nodes], dim=1)
    quad_top = torch.stack([nodes, torch.full_like(nodes, L)], dim=1)
    return SampleSet(
        domain=domain,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        hole=hole,
        interface_v_low=torch.cat([zeros_y, y_low], dim=1),
        interface_v_high=torch.cat([torch.zeros_like(y_high), y_high], dim=1),
        interface_h_left=torch.cat([x_left, zeros_x], dim=1),
        interface_h_right=torch.cat([x_right, torch.zeros_like(x_right)], dim=1),
        quad_domain=quad_domain,
        quad_domain_weights=quad_domain_weights,
        quad_left=quad_left,
        quad_right=quad_right,
        quad_top=quad_top,
        quad_boundary_weights=weights_1d,
    )

def mlp(in_dim: int, widths: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for width in widths:
        layers.extend([nn.Linear(last, width), nn.Tanh()])
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)

class VanillaMixed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = mlp(2, [200, 200, 200], 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AnchoredMixed(nn.Module):
    def __init__(self, n_features: int = 1000, w_max: float = 20.0) -> None:
        super().__init__()
        centers = torch.empty((n_features, 2), dtype=DTYPE).uniform_(-L, L)
        weights = torch.empty((n_features, 2), dtype=DTYPE).uniform_(-w_max, w_max)
        bias = -(weights * centers).sum(dim=1)
        self.register_buffer("weights", weights)
        self.register_buffer("bias", bias)
        self.net = mlp(n_features, [100, 100], 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.tanh(x @ self.weights.T + self.bias)
        return self.net(features)

class FourierMixed(nn.Module):
    def __init__(self, n_frequencies: int = 500, sigma: float = 20.0 / (2.0 * math.pi)) -> None:
        super().__init__()
        # Standard Gaussian random Fourier features. The 2*pi factor makes the
        # characteristic derivative scale comparable to W_max=20 in AnchoredMixed.
        frequencies = torch.randn((n_frequencies, 2), dtype=DTYPE) * sigma
        self.register_buffer("frequencies", frequencies)
        self.net = mlp(2 * n_frequencies, [100, 100], 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * (x @ self.frequencies.T)
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        return self.net(features)

class XPINNMixed(nn.Module):
    """Four-quadrant XPINN; subdomain IDs are LL, LR, UL, UR."""

    def __init__(self) -> None:
        super().__init__()
        self.nets = nn.ModuleList([mlp(2, [150, 150], 5) for _ in range(4)])

    @staticmethod
    def subdomain_id(x: torch.Tensor) -> torch.Tensor:
        return (x[:, 0] >= 0).long() + 2 * (x[:, 1] >= 0).long()

    def forward_subdomain(self, index: int, x: torch.Tensor) -> torch.Tensor:
        return self.nets[index](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        all_outputs = torch.stack([net(x) for net in self.nets], dim=0)
        row = self.subdomain_id(x)
        col = torch.arange(x.shape[0], device=x.device)
        return all_outputs[row, col]

class DEMNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = mlp(2, [200, 200, 200], 2)

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw(x)
        anchor = torch.tensor([[0.0, -L]], device=x.device, dtype=x.dtype)
        anchor_u = self.raw(anchor)[0, 0]
        u = raw[:, 0:1] - anchor_u
        v = ((x[:, 1:2] + L) / (2.0 * L)) * raw[:, 1:2]
        return torch.cat([u, v], dim=1)

def gradients(field: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(field.sum(), coordinates, create_graph=True)[0]

def mixed_loss(
    model: nn.Module,
    samples: SampleSet,
    p_lateral: float,
    p_top: float,
    include_xpinn_interfaces: bool = False,
) -> torch.Tensor:
    xy = samples.domain.detach().clone().requires_grad_(True)
    out = model(xy)
    u, v = out[:, 0:1], out[:, 1:2]
    sxx, syy, sxy = out[:, 2:3], out[:, 3:4], out[:, 4:5]
    grad_u = gradients(u, xy)
    grad_v = gradients(v, xy)
    exx, eyy = grad_u[:, 0:1], grad_v[:, 1:2]
    exy = 0.5 * (grad_u[:, 1:2] + grad_v[:, 0:1])
    trace = exx + eyy
    constitutive = (
        (sxx - (LAM * trace + 2.0 * MU * exx)).square()
        + (syy - (LAM * trace + 2.0 * MU * eyy)).square()
        + (sxy - 2.0 * MU * exy).square()
    ).mean()
    grad_sxx = gradients(sxx, xy)
    grad_syy = gradients(syy, xy)
    grad_sxy = gradients(sxy, xy)
    equilibrium = (grad_sxx[:, 0:1] + grad_sxy[:, 1:2]).square().mean()
    equilibrium = equilibrium + (grad_sxy[:, 0:1] + grad_syy[:, 1:2]).square().mean()

    out_l = model(samples.left)
    out_r = model(samples.right)
    out_t = model(samples.top)
    out_b = model(samples.bottom)
    out_h = model(samples.hole)
    traction = (
        ((out_l[:, 2] - p_lateral).square() + out_l[:, 4].square()).mean()
        + ((out_r[:, 2] - p_lateral).square() + out_r[:, 4].square()).mean()
        + ((out_t[:, 3] - p_top).square() + out_t[:, 4].square()).mean()
        + out_b[:, 4].square().mean()
    )
    nx = -samples.hole[:, 0] / R
    ny = -samples.hole[:, 1] / R
    tx = out_h[:, 2] * nx + out_h[:, 4] * ny
    ty = out_h[:, 4] * nx + out_h[:, 3] * ny
    traction = traction + (tx.square() + ty.square()).mean()

    anchor = torch.tensor([[0.0, -L]], device=DEVICE, dtype=DTYPE)
    displacement_bc = out_b[:, 1].square().mean() + model(anchor)[0, 0].square()
    total = equilibrium + constitutive + 10.0 * traction + 100.0 * displacement_bc

    if include_xpinn_interfaces:
        assert isinstance(model, XPINNMixed)
        pairs = [
            (samples.interface_v_low, 0, 1),
            (samples.interface_v_high, 2, 3),
            (samples.interface_h_left, 0, 2),
            (samples.interface_h_right, 1, 3),
        ]
        interface = torch.zeros((), device=DEVICE, dtype=DTYPE)
        for points, first, second in pairs:
            a = model.forward_subdomain(first, points)
            b = model.forward_subdomain(second, points)
            # Displacement and traction/stress continuity for the mixed system.
            interface = interface + (a - b).square().mean()
        total = total + 10.0 * interface
    return total

def dem_loss(model: DEMNet, samples: SampleSet, p_lateral: float, p_top: float) -> torch.Tensor:
    xy = samples.quad_domain.detach().clone().requires_grad_(True)
    uv = model(xy)
    grad_u = gradients(uv[:, 0:1], xy)
    grad_v = gradients(uv[:, 1:2], xy)
    exx, eyy = grad_u[:, 0:1], grad_v[:, 1:2]
    exy = 0.5 * (grad_u[:, 1:2] + grad_v[:, 0:1])
    trace = exx + eyy
    energy_density = 0.5 * LAM * trace.square() + MU * (exx.square() + eyy.square() + 2.0 * exy.square())
    internal_energy = (samples.quad_domain_weights[:, None] * energy_density.double()).sum()

    # Boundary length is one for each outer side. Sign follows traction=sigma*n.
    u_left = model(samples.quad_left)[:, 0].double()
    u_right = model(samples.quad_right)[:, 0].double()
    v_top = model(samples.quad_top)[:, 1].double()
    wb = samples.quad_boundary_weights
    external_work = (
        (-p_lateral * wb * u_left).sum()
        + (p_lateral * wb * u_right).sum()
        + (p_top * wb * v_top).sum()
    )
    return internal_energy - external_work

