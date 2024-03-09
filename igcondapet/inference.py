#%%
import argparse
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import torch
import sys
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism
from tqdm import tqdm
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler
torch.multiprocessing.set_sharing_strategy("file_system")
import time 
from joblib import Parallel, delayed
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(WORKING_DIR)
from utils.utils import str2bool, convert_to_N_digits, save_image
from get_datasets import get_test_unhealthy_datalist, get_transforms, get_test_datalist_trial
# %%
set_determinism(42)
#%%
def process_image(batch, model, embed, scheduler, total_timesteps, device, args, test_preds_dir, test_visuals_dir):
    guidance_scale_str = str(args.guidance_scale).replace('.', 'p')
    image = batch['PT'] 
    gt = batch['GT']
    fname = os.path.basename(batch['GT_meta_dict']['filename_or_obj'][0])[:-10]
    # filepath for 2D PET image
    ptpath = os.path.join(test_preds_dir, f'{fname}_gs{guidance_scale_str}_d{convert_to_N_digits(args.noise_level,3)}_pt.nii.gz')
    # filepath for 2D GT image
    gtpath = os.path.join(test_preds_dir, f'{fname}_gs{guidance_scale_str}_d{convert_to_N_digits(args.noise_level,3)}_gt.nii.gz')
    # filepath for 2D latent image
    ltpath = os.path.join(test_preds_dir, f'{fname}_gs{guidance_scale_str}_d{convert_to_N_digits(args.noise_level,3)}_lt.nii.gz')
    # filepath for 2D healthy counterfactual image
    hlpath = os.path.join(test_preds_dir, f'{fname}_gs{guidance_scale_str}_d{convert_to_N_digits(args.noise_level,3)}_hl.nii.gz')
    # filepath for 2D anomaly map
    anpath = os.path.join(test_preds_dir, f'{fname}_gs{guidance_scale_str}_d{convert_to_N_digits(args.noise_level,3)}_an.nii.gz')
    # filepath for image showing the above 5 images in a single plot 
    impath = os.path.join(test_visuals_dir, f'{fname}_gs{guidance_scale_str}_d{convert_to_N_digits(args.noise_level,3)}_im.png')
    pt = image[0, 0].cpu().numpy()
    gt = gt[0, 0].cpu().numpy()
    save_image(pt, ptpath)
    save_image(gt, gtpath)

    model.eval()

    current_img = image.to(device)
    scheduler.set_timesteps(num_inference_steps=total_timesteps)

    ## Enconding step
    # Encoding via class conditioning using an unconditional model 
    # (Notice the conditioning variable c=0)

    scheduler.clip_sample = False
    conditioning = torch.zeros(1).long().to(device)
    class_embedding = embed(conditioning).unsqueeze(1)
    progress_bar = tqdm(range(args.noise_level))
    for i in progress_bar:  # noising process for args.noise_level steps
        t = i
        with torch.no_grad():
            model_output = model(current_img, timesteps=torch.Tensor((t,)).to(current_img.device), context=class_embedding)
        current_img, _ = scheduler.reversed_step(model_output, t, current_img)
        progress_bar.set_postfix({"timestep input": t})

    latent_image = current_img
    lt = latent_image[0, 0].cpu().detach().numpy() # latent image after args.noise_level steps of noise encoding
    save_image(lt, ltpath)

    ## Deconding step
    # Decoding via class conditioning using both conditional (c=2 where 2:unhealthy class) and unconditional models (c=0)
    # After this we employ, implicit guidance for healthy counterfactual generation  
    conditioning = torch.cat([torch.zeros(1).long(), torch.ones(1).long()], dim=0).to(device)
    class_embedding = embed(conditioning).unsqueeze(1)
    progress_bar = tqdm(range(args.noise_level))
    for i in progress_bar:  # denoising process for args.noise_level steps
        t = args.noise_level - i
        current_img_double = torch.cat([current_img] * 2)
        with torch.no_grad():
            model_output = model(current_img_double, timesteps=torch.Tensor([t, t]).to(current_img.device), context=class_embedding)
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        # the equation below is called implicit or classifier-free guidance
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        current_img, _ = scheduler.step(noise_pred, t, current_img)
        progress_bar.set_postfix({"timestep input": t})
        torch.cuda.empty_cache()
    
    # healthy counterfactual generated after noise encoding and decoding via implicit guidance
    hl = current_img[0, 0].cpu().detach().numpy() 
    save_image(hl, hlpath)

    # anomaly map generation
    an = abs(pt - hl)
    save_image(an, anpath)
    
    # image plotting
    fig, ax = plt.subplots(1, 5, figsize=(10, 30))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)
    ax[0].imshow(np.rot90(pt), vmin=0, vmax=1, cmap="nipy_spectral")
    ax[0].set_title("Unhealthy\nPET slice")
    ax[0].axis('off')
    ax[1].imshow(np.rot90(lt), vmin=0, vmax=1, cmap="gray")
    ax[1].set_title("Latent\nimage")
    ax[1].axis('off')
    ax[2].imshow(np.rot90(hl), vmin=0, vmax=1, cmap="nipy_spectral")
    ax[2].set_title("Healthy\nreconstruction")
    ax[2].axis('off')
    ax[3].imshow(np.rot90(an), cmap="inferno")
    ax[3].set_title("Anomaly\nmap")
    ax[3].axis('off')
    ax[4].imshow(np.rot90(gt), cmap="gray")
    ax[4].set_title("Ground\ntruth")
    ax[4].axis('off')

    fig.savefig(impath, dpi=250, bbox_inches='tight')
    plt.close('all')
    print(f'Finished inference on image: {fname}')

def main(args):
    start = time.time()
    print(f'Running inference using guidance scale: {args.guidance_scale} and noise level: {args.noise_level}')
    main_dir = './results'    
    os.makedirs(main_dir, exist_ok=True)
    RESULTS_DIR = os.path.join(main_dir, args.experiment)
    
    models_dir = os.path.join(RESULTS_DIR,'models')    
    logs_dir = os.path.join(RESULTS_DIR,'logs')
    validlog_fpath = os.path.join(logs_dir, f'validlog_gpu{0}.csv')
    valid_df = pd.read_csv(validlog_fpath)
    best_epoch = args.val_interval*(1 + np.argmin(valid_df['Loss'])) 
    print(f'For {args.experiment}, the best training epoch (lowest validation loss) was: {best_epoch}')
    best_checkpoint_fpath = os.path.join(models_dir, f'checkpoint_ep{convert_to_N_digits(best_epoch, 5)}.pth')

    test_preds_dir = os.path.join(RESULTS_DIR, 'test_preds')
    test_visuals_dir = os.path.join(RESULTS_DIR, 'test_visuals')
    os.makedirs(test_preds_dir, exist_ok=True)
    os.makedirs(test_visuals_dir, exist_ok=True)

    device_id = 0
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    embedding_dimension = 64
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 64, 64),
        attention_levels=(args.attn_layer1, args.attn_layer2, args.attn_layer3),
        num_res_blocks=1,
        num_head_channels=16,
        with_conditioning=True,
        cross_attention_dim=embedding_dimension,
    ).to(device)
    embed = torch.nn.Embedding(num_embeddings=3, embedding_dim=embedding_dimension, padding_idx=0).to(device)
    scheduler = DDIMScheduler(num_train_timesteps=1000).to(device)
    
    total_timesteps = 1000

    best_checkpoint = torch.load(best_checkpoint_fpath, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    embed.load_state_dict(best_checkpoint['embed_state_dict'])
    scheduler.load_state_dict(best_checkpoint['scheduler_state_dict'])

    datalist, data_transforms = get_test_datalist_trial(), get_transforms()

    dataset = CacheDataset(datalist, transform=data_transforms, cache_rate=args.cache_rate)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False, persistent_workers=True)

    def process_batch(batch):
        process_image(batch, model, embed, scheduler, total_timesteps, device, args, test_preds_dir, test_visuals_dir)

    # Parallel processing: running args.num_workers jobs in parallel. You can change args.num_workers flag to higher values 
    # depending on the availability of GPU memory due to parallelization.  
    Parallel(n_jobs=args.num_workers, backend="loky")(delayed(process_batch)(batch) for batch in dataloader)

    elapsed = time.time() - start 
    print(f'Time taken: {elapsed/(60*60)} hrs')



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='IgCONDA-PET: Counterfactual Diffusion with Implicit Guidance for PET Anomaly Detection - Test phase')
    parser.add_argument('--experiment', type=str, default='exp0', metavar='exp',
                        help='experiment identifier')
    parser.add_argument('--attn-layer1', type=str2bool, default=False, metavar='attn1',
                        help='whether to put attention mechanism in layer 1 (default=False)')
    parser.add_argument('--attn-layer2', type=str2bool, default=True, metavar='attn2',
                        help='whether to put attention mechanism in layer 2 (default=True)')
    parser.add_argument('--attn-layer3', type=str2bool, default=True, metavar='attn3',
                        help='whether to put attention mechanism in layer 3 (default=True)')
    parser.add_argument('--guidance-scale', type=float, default=3.0, metavar='w',
                        help='Guidance scale for performing implicit guidance (default=3.0)')
    parser.add_argument('--noise-level', type=int, default=400, metavar='D',
                        help='number of noising and denoising steps for inference (default=400)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='nw',
                        help='num_workers for train and validation dataloaders (default=4)')
    parser.add_argument('--cache-rate', type=float, default=1, metavar='cr',
                        help='cache_rate for CacheDataset from MONAI (default=1)')
    parser.add_argument('--val-interval', type=int, default=10, metavar='val-interval',
                        help='epochs interval for which validation will be performed (default=10)')
    args = parser.parse_args()
    
    main(args)