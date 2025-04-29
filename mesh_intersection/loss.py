# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_circumcircle(triangles, edge_cross_prod, idx=None):
    ''' Calculate the circumscribed circle for the given triangles

        Args:
            - triangles (torch.tensor BxTx3x3): The tensor that contains the
              coordinates of the triangle vertices
            - edge_cross_prod (torch.tensor BxCx3): Contains the unnormalized
              perpendicular vector to the surface of the triangle.
        Returns:
            - circumradius (torch.tensor BxTx1): The radius of the
              circumscribed circle
            - circumcenter (torch.tensor BxTx3): The center of the
              circumscribed circel
    '''

    alpha = triangles[:, :, 0] - triangles[:, :, 2]
    beta = triangles[:, :, 1] - triangles[:, :, 2]

    # Calculate the radius of the circumscribed circle
    # Should be BxF
    circumradius = (torch.norm(alpha - beta, dim=2, keepdim=True) /
                    (2 * torch.norm(edge_cross_prod, dim=2, keepdim=True)) *
                    torch.norm(alpha, dim=2, keepdim=True) *
                    torch.norm(beta, dim=2, keepdim=True))

    # Calculate the coordinates of the circumcenter of each triangle
    # Should BxFx3
    circumcenter = torch.cross(
        torch.sum(alpha ** 2, dim=2, keepdim=True) * beta -
        torch.sum(beta ** 2, dim=2, keepdim=True) * alpha,
        torch.cross(alpha, beta, dim=-1), dim=2)
    circumcenter /= (2 * torch.sum(edge_cross_prod ** 2, dim=2, keepdim=True))

    return circumradius, circumcenter + triangles[:, :, 2]


def repulsion_intensity(x, sigma=0.5, penalize_outside=True, linear_max=1000):
    ''' Penalizer function '''
    quad_penalty = (-(1.0 - 2.0 * sigma) / (4.0 * sigma ** 2) *
                    x ** 2 - 1 / (2.0 * sigma) * x +
                    0.25 * (3 - 2 * sigma))
    linear_region_mask = (x.le(-sigma) * x.gt(-linear_max)).to(dtype=x.dtype)
    if penalize_outside:
        quad_region_mask = (x.gt(-sigma) * x.lt(sigma)).to(dtype=x.dtype)
    else:
        quad_region_mask = (x.gt(-sigma) * x.lt(0)).to(dtype=x.dtype)

    return (linear_region_mask * (-x + 1 - sigma) +
            quad_region_mask * quad_penalty)


def dist_to_cone_axis(points_rel, dot_prod, cone_axis, cone_radius,
                      sigma=0.5, epsilon=1e-6, vectorized=True):
    ''' Computes the distance of each point to the axis

        This function projects the points on the plane of the base of the cone
        and computes the distance to the axis. This is subsequently normalized
        by the radius of the cone at the height level of the point, so that
        points with distance < 1 are in the code, distance == 1 means that the
        point is on the surface and distance > 1 means that the point is
        outside the cone.

        Args:
            - points_rel (torch.tensor BxCxNx3): The coordinates of the points
              relative to the center of the cone
            - dot_prod (torch.tensor BxCxN): The dot product of the points (in
              relative coordinates with respect to the cone center) with the
              axis of the cone
            - cone_axis (torch.tensor BxCx3): The axis of the cone
            - cone_radius (torch.tensor BxCx1): The radius of the cone
        Keyword args:
            - sigma (float = 0.5): The height of the cone
            - epsilon (float = 1e-6): Numerical stability constant for the
              float division
            - vectorized (bool = True): Whether to use an iterative or a
              vectorized version of the function
    '''

    if vectorized:
        batch_size, num_collisions = cone_radius.shape[:2]
        numerator = torch.norm(points_rel - dot_prod.unsqueeze(dim=-1) *
                               cone_axis.unsqueeze(dim=-2),
                               p=2, dim=-1)
        denominator = -cone_radius / sigma * dot_prod + cone_radius
    else:
        batch_size, num_collisions = cone_radius.shape[:2]
        numerator = torch.norm(points_rel - dot_prod.unsqueeze(-1) * cone_axis,
                               p=2, dim=-1)
        denominator = -cone_radius.view(batch_size, num_collisions) / sigma * \
            dot_prod + cone_radius.view(batch_size, num_collisions)

    return numerator / (denominator + epsilon)

def point_triangle_distance(point: torch.Tensor,
                            tri: torch.Tensor) -> torch.Tensor:
    """
    Compute the shortest Euclidean distance from each point to its
    corresponding triangle.

    Args
    ----
    point : (B, 3) tensor
        3-D coordinates of the query points.
    tri   : (B, 3, 3) tensor
        Triangle vertices ordered as (v0, v1, v2) for each batch item.

    Returns
    -------
    dist : (B,) tensor
        The point-to-triangle distance for every batch element.
    """
    EPS = 1e-12 

    # Split triangle vertices for readability
    v0, v1, v2 = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]

    # Edge vectors and vector from v0 to the point
    e0 = v1 - v0                  # edge v0→v1
    e1 = v2 - v0                  # edge v0→v2
    w  = point - v0               # v0→p

    # Pre-compute dot products (all shape = (B,))
    a = (e0 * e0).sum(1).clamp(min=EPS)  # |e0|^2
    b = (e0 * e1).sum(1)          # e0·e1
    c = (e1 * e1).sum(1).clamp(min=EPS)  # |e1|^2
    d = (e0 * w ).sum(1)          # e0·w
    e = (e1 * w ).sum(1)          # e1·w
    f = (w  * w ).sum(1)          # |w|^2

    det = a * c - b * b           # 2D parallelogram area squared
    det = det.clamp(min=EPS)     # numerical safety
    s   = b * e - c * d           # barycentric test numerator for v1
    t   = b * d - a * e           # barycentric test numerator for v2

    # Region masks ----------------------------------------------------------
    in_face = (s >= 0) & (t >= 0) & (s + t <= det)   # inside triangle

    # Edge v0–v1
    s01 = (d / a).clamp(0, 1)
    dist2_e01 = (w - s01.unsqueeze(1) * e0).pow(2).sum(1)

    # Edge v0–v2
    t02 = (e / c).clamp(0, 1)
    dist2_e02 = (w - t02.unsqueeze(1) * e1).pow(2).sum(1)

    # Edge v1–v2
    w2  = point - v1
    e2  = v2 - v1
    a2  = (e2 * e2).sum(1).clamp(min=EPS)  # |e2|^2
    d2  = (e2 * w2).sum(1)
    u12 = (d2 / a2).clamp(0, 1)
    dist2_e12 = (w2 - u12.unsqueeze(1) * e2).pow(2).sum(1)

    # Interior of the triangle
    proj = (d * (c) - e * (b))[..., None] * e0 + (e * (a) - d * (b))[..., None] * e1
    dist2_face = (f * det - (d * s + e * t)) / det
    dist2_face = dist2_face.clamp(min=0)             # numerical safety

    # Pick distance² from the correct region
    dist2 = torch.where(in_face, dist2_face,
                        torch.min(torch.min(dist2_e01, dist2_e02),
                                  dist2_e12))

    return torch.sqrt(dist2)         # final Euclidean distance (B,)

def edge_edge_distance(p1: torch.Tensor, q1: torch.Tensor,
                       p2: torch.Tensor, q2: torch.Tensor,
                       eps: float = 1e-12) -> torch.Tensor:
    """
    Compute the minimum Euclidean distance between two 3-D segments
    for every batch item.

    Args
    ----
    p1, p2 : (B, 3) tensor
        3-D coordinates of the first segment end-points.
    q1, q2 : (B, 3) tensor
        3-D coordinates of the second segment end-points.
    eps  : float
        Parallel-line threshold.

    Returns
    -------
    dist : (B,) tensor
        Shortest point-to-point distance between the two segments.
    """

    d1 = q1 - p1                           
    d2 = q2 - p2                           
    r  = p1 - p2                           # vector between origins

    a = (d1 * d1).sum(1)                   # squared length of d1
    e = (d2 * d2).sum(1)                   # squared length of d2
    f = (d2 * r ).sum(1)

    # Assume segments are not degenerate (length > 0); handle elsewhere if needed
    b = (d1 * d2).sum(1)
    c = (d1 * r ).sum(1)
    denom = a * e - b * b                 # always >= 0

    # Parallel segments: force denominator to 1 to avoid divide-by-0,
    # we'll correct the result via clamping later.
    parallel = denom < eps
    denom_adjusted = torch.where(parallel, torch.ones_like(denom), denom)

    s = (b * f - c * e) / denom_adjusted
    t = (a * f - b * c) / denom_adjusted

    # For nearly parallel segments use fallback:
    # project one segment endpoint onto the other segment, set the other
    # parameter to 0, then clamp.
    s = torch.where(parallel, torch.zeros_like(s), s)

    # Clamp s, t into [0,1] and recompute the opposite parameter when out of range
    s_clamped = s.clamp(0.0, 1.0)
    t_clamped = t.clamp(0.0, 1.0)

    # If clamping occurred on s, recompute t for the new s value
    recompute_t = (s_clamped != s) | parallel
    r_new = r + s_clamped.unsqueeze(1) * d1
    t_new = ((r_new * d2).sum(1) / e).clamp(0.0, 1.0)
    t_final = torch.where(recompute_t, t_new, t_clamped)

    # If t was clamped (and s not), recompute s for the new t value
    recompute_s = (t_clamped != t) & ~parallel & (s_clamped == s)
    r_new2 = r - t_final.unsqueeze(1) * d2
    s_new = ((-r_new2) * d1).sum(1) / a
    s_final = torch.where(recompute_s, s_new.clamp(0.0, 1.0), s_clamped)

    # Closest points on each segment
    c1 = p1 + s_final.unsqueeze(1) * d1
    c2 = p2 + t_final.unsqueeze(1) * d2

    # Return Euclidean distance
    return (c1 - c2).norm(dim=1)


def triangle_distance(triangle_points, other_triangle_points):
    ''' Computes the distance between two triangles

        Args:
            - triangle_points (torch.tensor Nx3x3): The coordinates of the
              points of the triangles of pair I
            - other_triangle_points (torch.tensor Nx3x3): The coordinates of
              the points of the triangles of pair II
        Returns:
            - (torch.tensor N): The distance between the triangles
    '''
    min_dist = torch.full((triangle_points.shape[0],),
                          float('inf'), dtype=triangle_points.dtype,
                          device=triangle_points.device)
    for i in range(3):
        min_dist = torch.min(min_dist, point_triangle_distance(triangle_points[:, i],
                                                               other_triangle_points))
        min_dist = torch.min(min_dist, point_triangle_distance(other_triangle_points[:, i],
                                                               triangle_points))
    for i in range(3):
        for j in range(3):
            min_dist = torch.min(min_dist, edge_edge_distance(triangle_points[:, i],
                                                                triangle_points[:, (i + 1) % 3],
                                                                other_triangle_points[:, j],
                                                                other_triangle_points[:, (j + 1) % 3]))
    return min_dist

def conical_distance_field(triangle_points, cone_center, cone_radius,
                           cone_axis, sigma=0.5, vectorized=True,
                           penalize_outside=True, linear_max=1000):
    ''' Distance field calculation for a cone

        Args:
            - triangle_points (torch.tensor (BxCxNx3): Contains
            the points whose distance from the cone we want to calculate.
            - cone_center (torch.tensor (BxCx3)): The coordinates of the center
              of the cone
            - cone_radius (torch.tensor (BxC)): The radius of the base of the
              cone
            - cone_axis (torch.tensor(BxCx3)): The unit vector that represents
              the axis of the cone
        Keyword Arguments
            - sigma (float = 0.5): The float value of the height of the cone
            - vectorized (bool = True): Whether to use an iterative or a
              vectorized version of the function
        Returns:
            - (torch.tensor BxCxN): The distance field values at the N points
              for the cone
    '''

    if vectorized:
        # Calculate the coordinates of the points relative to the center of
        # the cone
        points_rel = triangle_points - cone_center.unsqueeze(dim=-2)
        # Calculate the dot product between the relative point coordinates and
        # the axis (normal) of the cone. Essentially, it is the length of the
        # projection of the relative vector on the axis of the cone
        dot_prod = torch.sum(points_rel * cone_axis.unsqueeze(dim=-2), dim=-1)

        # Calculate the distance of the projections of the points on the cone
        # base plane to the center of cone, normalized by the height
        axis_dist = dist_to_cone_axis(points_rel, dot_prod,
                                      cone_axis, cone_radius,
                                      sigma=sigma, vectorized=True)

        circumcenter_dist = repulsion_intensity(
            dot_prod, sigma=sigma, penalize_outside=penalize_outside,
            linear_max=linear_max)

        # Ignore the points with axis_dist > 1, since they are out of the cone
        mask = axis_dist.lt(1).to(dtype=triangle_points.dtype)

        distance_field = mask * ((1 - axis_dist) * circumcenter_dist).pow(2)
    else:
        batch_size, num_collisions, num_points = triangle_points.shape[:3]
        distance_field = torch.zeros([batch_size, num_collisions, 3],
                                     dtype=triangle_points.dtype,
                                     device=triangle_points.device)
        for idx in range(num_points):
            # The relative coordinates of each point to the center of the cone
            # BxCx3
            points_rel = triangle_points[:, :, idx, :] - cone_center

            # Calculate the dot product between the relative point coordinates
            # and the axis (normal) of the cone. Essentially, it is the length
            # of the projection of the relative vector on the axis of the cone
            dot_prod = torch.sum(points_rel * cone_axis, dim=-1)

            axis_dist = dist_to_cone_axis(points_rel, dot_prod,
                                          cone_axis, cone_radius,
                                          sigma=sigma,
                                          vectorized=False)

            circumcenter_dist = repulsion_intensity(
                dot_prod, sigma=sigma, penalize_outside=penalize_outside)
            mask = (axis_dist < 1).to(dtype=triangle_points.dtype)

            distance_field[:, :, idx] = (1 - axis_dist) * mask * \
                circumcenter_dist

    return torch.pow(distance_field, 2)


class DistanceFieldPenetrationLoss(nn.Module):
    def __init__(self, sigma=0.5, point2plane=False, vectorized=True,
                 penalize_outside=True, linear_max=1000):
        super(DistanceFieldPenetrationLoss, self).__init__()
        self.sigma = sigma
        self.point2plane = point2plane
        self.vectorized = vectorized
        self.penalize_outside = penalize_outside
        self.linear_max = linear_max

    def forward(self, triangles, close_idxs):
        '''
        Args:
            - triangles: A torch tensor of size BxFx3x3 that contains the
                coordinates of the triangle vertices
            - close_idxs: A torch tensor of size Bx(-1)x2 that contains the
              indices of the close pairs
        Returns:
            A tensor with size B that contains the self penetration loss for
            each mesh in the batch
        '''

        clo_idxs = close_idxs[:, :, 0].ge(0).nonzero()
        if len(clo_idxs) < 1:
            return torch.zeros([triangles.shape[0]],
                               dtype=triangles.dtype,
                               device=triangles.device,
                               requires_grad=triangles.requires_grad)

        receiver_faces = close_idxs[clo_idxs[:, 0], clo_idxs[:, 1], 0]
        intruder_faces = close_idxs[clo_idxs[:, 0], clo_idxs[:, 1], 1]

        batch_idxs = clo_idxs[:, 0]
        num_closes = clo_idxs.shape[0]

        batch_size = triangles.shape[0]

        if len(intruder_faces) < 1:
            return torch.tensor(0.0, dtype=triangles.dtype,
                                device=triangles.device,
                                requires_grad=triangles.requires_grad)
        # Calculate the edges of the triangles
        # Size: BxFx3
        edge0 = triangles[:, :, 1] - triangles[:, :, 0]
        edge1 = triangles[:, :, 2] - triangles[:, :, 0]

        recv_triangles = triangles[batch_idxs, receiver_faces]
        intr_triangles = triangles[batch_idxs, intruder_faces]

        distace = triangle_distance(recv_triangles, intr_triangles)
        eps = 1e-3
        loss = torch.max(eps - distace, torch.tensor(0.0,
                                                      device=triangles.device,
                                                      dtype=triangles.dtype)).mean()

        return loss
