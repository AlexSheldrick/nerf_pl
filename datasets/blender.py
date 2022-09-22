import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import pyexr
from skimage.transform import resize
from skimage import filters
from .ray_utils import *

class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800), num_images=120):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.num_images = num_images
        self.define_transforms()

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            factor = 120 // self.num_images
            self.meta['frames'] = self.meta['frames'][:self.num_images] * factor
            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                #depth = torch.zeros_like(img[0]).reshape(-1,1)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

                depth = pyexr.open(image_path.replace('.png','_depth_0001.exr')).get('R') #(h,w,1)
                #depth = pyexr.open(image_path.replace('.png','_distance.exr')).get() #(h,w,1)
                
                depth = resize(depth, self.img_wh, order=0, anti_aliasing=False)
                depth[depth > 10] = 0
                noise = filters.scharr(depth)
                noise = filters.gaussian(noise, sigma=3)
                noise = torch.from_numpy((noise-noise.min()) / (noise.max() - noise.min())).flatten().unsqueeze(1)*0.5
                
                depth = self.transform(depth).flatten().unsqueeze(1) #(h*w/4,1)
                depth_uncertainty = (torch.where(depth > 0, 0.5, 0) + noise)
                
                #normal = pyexr.open(image_path.replace('.png','_normal_0001.exr')).get()
                #normal = (normal @ pose[:, :3] + 1)/2 #transform from world to view and scale from [-1,1] to [0, 1] for visualization

                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
               
                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1]), 
                                             depth, depth_uncertainty],
                                             1)] # (h*w, 8) --> 10
                """self.all_rays += [torch.cat([rays_o, rays_d, 
                                             depth,
                                             depth],
                                             1)] # (h*w, 8)"""

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)            
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            depth = pyexr.open(img_path.replace('.png','_depth_0001.exr')).get('R') #(h,w,1)
            #depth = pyexr.open(img_path.replace('.png','_distance.exr')).get() #(h,w,1)

            depth = resize(depth, self.img_wh, order=0, anti_aliasing=False)
            depth[depth > 10] = 0
            noise = filters.scharr(depth)
            noise = filters.gaussian(noise, sigma=3)
            noise = torch.from_numpy((noise-noise.min()) / (noise.max() - noise.min())).flatten().unsqueeze(1)
            
            depth = self.transform(depth).flatten().unsqueeze(1) #(h*w/4,1)
            depth_uncertainty = (torch.where(depth > 0, 0.5, 0) + noise)*0.5     

            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            #depth = torch.zeros_like(img[0]).reshape(-1,1)

            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB   

            #normal = pyexr.open(img_path.replace('.png','_normal_0001.exr')).get()
            #normal = (normal @ c2w[:, :3] + 1)/2 #transform from world to view and scale from [-1,1] to [0, 1] for visualization
       

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1]),
                              depth, depth_uncertainty],
                              1) # (H*W, 8)

            """rays = [torch.cat([rays_o, rays_d, 
                            depth - depth_var,
                            depth + depth_var],
                            1)] # (h*w, 8)"""

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample


### TRANSFORMS NORMAL MAP IN WORLD COORDINATES TO VIEW COORDINATES
def normal_world_to_view(normal_map, pose):
    i = 39
    with open('data/lego/transforms_train.json', 'r') as f:
            meta = json.load(f)
    if normal_map is None:
        normal_path = f'data/lego/train/r_{i}_normal_0001.exr'
        no = pyexr.open(normal_path).get()
    if pose is None:
        c2w = np.array(meta['frames'][i]['transform_matrix'])[:3, :4]

    no = (no@ c2w[:, :3] + 1)/2 #transform from world to view and scale from [-1,1] to [0, 1] for visualization