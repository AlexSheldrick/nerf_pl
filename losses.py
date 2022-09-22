import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.MSEloss = nn.MSELoss(reduction='mean')
        self.GNNLoss = nn.GaussianNLLLoss(eps=1e-3, reduction='mean')

    def forward(self, results, targets):
        depth_mask = targets['depth'].squeeze()  > 0
        rgb_loss = self.MSEloss(results['rgb_coarse'], targets['rgb'])       

        #orientation_loss = self.orientation_loss(results, targets, 'coarse')
        orientation_loss = 0.

        #DSNerf found it was better to supervise only fine network with depth
        #depth_loss = self.MSEloss(results['depth_coarse'], targets['depth']) *0.1
        depth_loss = 0.
        #distortion_loss = self.lossfun_distortion(results['z_vals'], results['weights_fine'])
        #distortion_loss = torch.mean(results['distortion'])
        distortion_loss = 0.

        #depth_loss = self.GNNL_Loss(results['depth_coarse'][depth_mask], targets['depth'][depth_mask], results['depth_stdev_coarse'][depth_mask])

        if 'rgb_fine' in results:
            rgb_loss = rgb_loss + self.MSEloss(results['rgb_fine'], targets['rgb'])            
            orientation_loss = orientation_loss + self.orientation_loss(results, targets, 'fine')
            depth_loss = depth_loss + self.MSEloss(results['depth_fine'], targets['depth'])
            #depth_loss = results['newdepth'].mean() + depth_loss
            
            #depth_loss = self.GNNL_Loss(results['depth_fine'], targets['depth'], results['depth_stdev_fine'], targets['depth_uncertainty'])
            #                            

        #orientation_loss = 0
        loss_dict = {'rgb': rgb_loss, 'depth': depth_loss, 'orientation': orientation_loss, 'distortion': distortion_loss}
        return loss_dict
    
    def orientation_loss(self, results, targets, mode='coarse'):
        loss = 0.
        w = results[f'weights_{mode}']
        n = results[f'normals_{mode}']
        n = self.l2_normalize(n)
        # Negate viewdirs to represent normalized vectors from point to camera.
        v = -1. * targets['dirs']
        n_dot_v = (n * v[..., None, :])
        n_dot_v = n_dot_v.sum(axis=-1)
        loss = torch.mean((w * torch.minimum(torch.tensor(0.0), n_dot_v)**2).sum(axis=-1))
        return loss
               
    def l2_normalize(self, x, eps=1e-6):        
        """Normalize x to unit length along last axis."""
        return x / torch.sqrt(torch.maximum(torch.sum(x**2, axis=-1, keepdims=True), torch.tensor(eps)))

    def GNNL_Loss(self, depth_pred, depth_gt, var, depth_gt_var):
        mask = torch.logical_or(var > depth_gt_var, torch.abs(depth_gt - depth_pred) > depth_gt_var)
        #mask = torch.abs(depth_gt.squeeze() - depth_pred.squeeze()) > depth_gt_var.squeeze()
        
        loss = self.GNNLoss(depth_pred[mask], depth_gt[mask], var[mask])
        #loss = ((depth_pred[mask] - depth_gt[mask])**2 ).mean() #/ var**2
        #loss = nn.MSELoss()(depth_pred[mask].squeeze(), depth_gt[mask].squeeze())
        if torch.isnan(loss): loss = 0.
        return loss

    def lossfun_distortion(self, t, w):   #t=z_vals, w=weights. Loss from mip-nerf 360
        """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
        # The loss incurred between all pairs of intervals.
        ut = (t[..., 1:] + t[..., :-1]) / 2
        dut = torch.abs(ut[..., :, None] - ut[..., None, :])
        loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, axis=-1), axis=-1)

        # The loss incurred within each individual interval with itself.
        loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), axis=-1) / 3

        return loss_inter + loss_intra



loss_dict = {'mse': MSELoss}