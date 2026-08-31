# SPDX-License-Identifier: Apache-2.0

"""Current physical-contact feedback observations."""

import torch
from torch import Tensor


def compute_body_contact_feedback_obs(
    rigid_body_contacts: Tensor,
    current_contact_force_magnitudes: Tensor,
    body_ids: list[int] | None = None,
    force_scale: float = 100.0,
) -> Tensor:
    if body_ids is not None:
        rigid_body_contacts = rigid_body_contacts[:, body_ids]
        current_contact_force_magnitudes = current_contact_force_magnitudes[:, body_ids]
    contacts = rigid_body_contacts.to(current_contact_force_magnitudes.dtype)
    normalized_force = torch.log1p(current_contact_force_magnitudes.clamp_min(0.0))
    normalized_force = (normalized_force / torch.log1p(
        current_contact_force_magnitudes.new_tensor(force_scale)
    )).clamp(max=2.0)
    return torch.stack([contacts, normalized_force], dim=-1).flatten(start_dim=1)


def compute_reference_contact_obs(rigid_body_contacts: Tensor) -> Tensor:
    """Privileged intended-contact signal used by the critic only."""
    return rigid_body_contacts.to(torch.float32)
