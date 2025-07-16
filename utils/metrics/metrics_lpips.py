import argparse
import os
import sys
sys.path.append(os.getcwd())
import torch

from PIL import Image, ImageFile
from tqdm import tqdm
from torchvision.transforms import transforms

from utils.metrics import dnnlib
import lpips

ImageFile.LOAD_TRUNCATED_IMAGES = True

def compute_clpips(instance_dir,output_dir,resolution = 256,):
    """
    This function computes the IC-LPIPS score for given generated images.
    """
    dataset_kwargs_instance=dict(
        class_name="utils.metrics.dataset.ImageFolderDataset",
        path=instance_dir,
        use_labels=False,
        max_size=None,
        xflip=False,
        resolution=resolution)
    device = "cuda:0"

    with torch.no_grad():
        loss_fn_alex = lpips.LPIPS(net='vgg', verbose = False).to(device) # best forward scores

        data_list = []
        dataset_instance = dnnlib.util.construct_class_by_name(**dataset_kwargs_instance)                                                    
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

        # calculate the features of all the real images, which is used to calculate the genrated images.
        for img, _labels in torch.utils.data.DataLoader(dataset=dataset_instance, batch_size=64, **data_loader_kwargs):
            if img.shape[1] == 1:
                img = img.repeat([1, 3, 1, 1])
            if img.shape[1] == 4:
                img = img[:, :3, :, :]
            data_list.append(img.to(device))
        data_list = torch.cat(data_list, dim = 0)
        cluster = [[] for _ in range(data_list.shape[0])]
        label = torch.zeros([1, 0], device=device)

        # Calculate the LPIPS between 1000 anomaly images and all real images one by one,
        # and form the cluster using the realimages as the center.
        for i in range(1000):
            file_name = f"{i:03}.png"
            file_path = os.path.join(output_dir, file_name)
            img = Image.open(file_path)
            img = img.resize((256,256), Image.BILINEAR)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            img = transform(img).unsqueeze(0)
            score_list = loss_fn_alex(img.repeat(data_list.shape[0], 1, 1, 1).to(device), data_list)

            closest_index = score_list.argmin().item()
            if len(cluster[closest_index]) < 140:
                cluster[closest_index].append(img)
           
        print("done!") 
        cluster_lpips = []
        i= 0

        # Calculate the LPIPS between all the images in the same cluster,
        # and get the IC-LPIPS
        iterator = tqdm(cluster, desc = 'Computing clustered LPIPS')
        for c in iterator:
            print("Cluster {} contains {} images".format(i, len(c)))
            if len(c) <= 1:
                cluster_lpips.append(0.0)
                i+=1
                continue
            c_lpips = 0.0
            img = torch.cat(c, dim = 0).to(device)
            ref_img = img.clone()
            for _ in range(img.shape[0] - 1):
                img = torch.cat([img[1:], img[0:1]], dim = 0)
                c_lpips += loss_fn_alex(img, ref_img).sum().item()
            cluster_lpips.append(c_lpips / (img.shape[0] * (img.shape[0] - 1)))
            i+=1

    print(cluster_lpips)
    clpips = sum(cluster_lpips) / len(cluster_lpips)
    rz_sum = 0.0
    n = 0
    for score in cluster_lpips:
        if score != 0.0:
            rz_sum += score
            n += 1
    clpips_rz = rz_sum / n
    return clpips, clpips_rz

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="A folder containing the generated data.",
    )
    parser.add_argument(
        "--instance_dir",
        type=str,
        required=False,
        help="A folder containing the real data of anomaly images.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        required=False,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def main(args):
    clpips1k, clpips1k_rz = compute_clpips(
                                instance_dir=args.instance_dir,
                                output_dir=args.output_dir,
                                resolution=args.resolution
                            )
    print("clpips1k : {}  , clpips1k_rz: {} ".format(clpips1k,clpips1k_rz))

if __name__ == "__main__":
    args = parse_args()
    main(args,)
    

    




    

