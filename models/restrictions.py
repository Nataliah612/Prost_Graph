import torch 
import numpy as np 


def log_barrier(z, t=5):
    """
    Calcula la pérdida de barrera logarítmica para una restricción z <= 0.
    
    Args:
        z (torch.Tensor): Valor de la restricción.
        t (float): Parámetro de suavizado.
    
    Returns:
        torch.Tensor: Pérdida de barrera logarítmica.
    """
    log_barrier_loss = torch.tensor(0).cuda().float()
    for i in np.arange(0, z.shape[0]):
        zi = z[i, 0]
        if zi <= - 1 / t ** 2: # Restriccion cumplida
            log_barrier_loss += - torch.log(-zi) / t
        else:               # Restriccion no cumplida
            log_barrier_loss += t * zi + -np.log(1 / (t ** 2)) + 1/t #(Articulo)#/ t + 1 / t
    return log_barrier_loss

def loss_restrictions(out_patches, alpha_pe=0.1, alpha_pc=1, t_pe=15, t_pc=5, null_classes=False, device=str() ):
    """
    Calcula la pérdida con restricciones para las salidas a nivel de WSI y de los parches.
    
    Args:
        out_patch (torch.Tensor): Salidas del modelo a nivel parche (shape [batch_size, num_patches, num_classes]).
        t_pe (float): Parámetro de suavizado para la pérdida Positive Expansion.
        t_pc (float): Parámetro de suavizado para la pérdida Proportion Constraint.
    
    Returns:
        torch.Tensor: Pérdida de restricciones.
    """
    # Building matrix restrictions
    log_barrier_loss_pe, log_barrier_loss_pc = 0.0, 0.0
    
    for i in range(out_patches.shape[0]):

        if null_classes==True:
            out_patch =  out_patches[i,1:] # Ignorar clase 0 (no cancer)
        else: 
            out_patch =  out_patches[i,:]

        nRestrictions = 4 #out_patch.shape
        indexes = torch.argsort(out_patch, descending=True) # Ordena las proporciones de mayor a menor
        
        # Restriction 1: Positive expantions
        o_pe = torch.zeros((nRestrictions, len(out_patch)),device=device)# Matriz de clases activas 
        for i in torch.arange(0, nRestrictions):
            o_pe[i, indexes[i]] = -1  # marca la clase más grande (primaria)
        z1 = torch.matmul(o_pe, out_patch.unsqueeze(-1))
        log_barrier_loss_pe +=  log_barrier(z1, t=t_pe)

        # Restriction 2: Proportion Constraint
        o_pc = torch.zeros((nRestrictions, len(out_patch)), device=device) # Matriz de orden realtivo (-1)
        for i in torch.arange(0, nRestrictions-1):
            o_pc[i, indexes[i]] = -1 # marca la clase más grande (primaria)
            o_pc[i, indexes[i + 1]] = 1 # marca la clase segunda más grande (secundaria)
        z2 = torch.matmul(o_pc, out_patch.unsqueeze(-1)) 
        log_barrier_loss_pc += log_barrier(z2, t=t_pc)
    
    return alpha_pe*log_barrier_loss_pe + alpha_pc*log_barrier_loss_pc