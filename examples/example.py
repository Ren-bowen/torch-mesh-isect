from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss
import torch

triangles = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1.001, 0], [1.001, 0, 0], [1, 1, 0]]], dtype=torch.float32, device='cuda')
print(triangles.shape)
triangles = triangles.unsqueeze(0)
search_tree = BVH(max_collisions=8)
pen_distance = collisions_loss.DistanceFieldPenetrationLoss()
collision_idx = search_tree(triangles)
print(collision_idx)
collision_loss = pen_distance(triangles, collision_idx)
print(collision_loss)