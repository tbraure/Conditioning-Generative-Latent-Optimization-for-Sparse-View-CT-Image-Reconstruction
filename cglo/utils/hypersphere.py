import torch
from numpy import pi
from torch.distributions.multivariate_normal import MultivariateNormal

from cglo.utils.misc import export

def spherical_to_cartesian(radius, angles):
    '''
        Takes:
            - radius [N, 1]
            - angles [N, D-1]
        Returns cartesian [N, D]
    '''
    # converts to double: gives 1e-8 precision when applying |x - f⁻¹(f(x))|
    angles = angles.to(torch.double)
    radius = radius.to(torch.double)

    # transform to cartesian with cumulative product of cos and sin
    unit_vector = torch.ones(angles.shape[0], 1)
    unit_vector = export(unit_vector, a.device, grad=a.requires_grad)
    unit_vector = unit_vector.to(torch.double)
    sin_product = torch.cat([unit_vector, angles.sin().cumprod(-1)], -1)
    cos_product = torch.cat([angles.cos(), unit_vector], -1)

    cartesian = radius * sin_product * cos_product
    cartesian = cartesian.to(torch.float)

    return cartesian

def cartesian_to_spherical(cartesian):

    # converts to double: gives 1e-8 precision when applying |x - f⁻¹(f(x))|
    cartesian = cartesian.to(torch.double)

    # keep track of points in the last negative hyper-hemisphere
    negative_indices = cartesian[..., -1] < 0

    # compute angles and radius
    cumulative_roots = cartesian.flip(-1).square().cumsum(-1).sqrt()
    radius = cumulative_roots[..., [-1]].to(torch.float)
    angles = (cartesian[..., :-1] / cumulative_roots.flip(-1)[..., :-1]).acos()

    # correction for acos on negative_indices
    if negative_indices.any():
        angles[negative_indices, -1] = 2*pi - angles[negative_indices, -1]
    angles = angles.to(torch.float)

    return radius, angles

def diffused_normal_sampling(sample_size, latent_points, diffusion_radius):

    multi_standard_normal = MultivariateNormal(
        latent_points.to(torch.device('cpu')),
        torch.stack([
            diffusion_radius * torch.eye(latent_points.shape[1])
        ]*latent_points.shape[0])
    )

    sampled_points = multi_standard_normal.sample(torch.tensor([sample_size]))

    projected_points = projector(sampled_points.view(
        sample_size * latent_points.shape[0],
        latent_points.shape[1]
    ))

    return projected_points

def projected_normal_sampling(output_shape, cartesian_output=True):

    multi_standard_normal = MultivariateNormal(
        torch.zeros(output_shape[1]),
        torch.eye(output_shape[1])
    )

    projected_points = projector(
        multi_standard_normal.sample(torch.tensor([output_shape[0]]))
    )

    if cartesian_output:
        result = projected_points
    else:
        result = cartesian_to_spherical(projected_points)[1]

    return result

def uniformity_metric(latent_points, norm=2, factor=2):
    '''
        Compute Gaussian Pairwise Potential
        (A metric of uniformity on the hypersphere)

        Formula: E[exp(|x - y|²)], for (x, y) on the hypersphere
    '''
    return torch.pdist(latent_points, p=norm).square().mul(-factor).exp().mean()

def projector(latent_points, center=False, norm=2, tolerance=1e-8):

    # Centered projection (substracting barycenter) might be detrimental
    # Ex: Few points, Clustered points
    if center:
        new_latent_points = latent_points - latent_points.mean(dim=0)
    else:
        new_latent_points = latent_points

    latent_points_norm = new_latent_points.norm(p=norm, dim=1, keepdim=True)

    # Control for below tolerance norms to avoid null-divide numerical errors
    control_norm = torch.clip(latent_points_norm, min=tolerance)

    return new_latent_points / control_norm

