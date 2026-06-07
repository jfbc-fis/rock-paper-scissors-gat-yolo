import torch

# Un tensor simple (1 dimension)
mi_tensor = torch.tensor([1.0, 2.0, 3.0])
print(mi_tensor)
print(type(mi_tensor))
print(mi_tensor.shape)

# Un tensor 2D (como una tabla)
tabla = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])
print(tabla.shape)

# Moverlo a la GPU
tabla_gpu = tabla.cuda()
print(tabla_gpu.device)

