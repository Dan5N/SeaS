import os
def create_infer_sh():
    """
    This function is used to generate infer.sh for each anomaly in each product,
    according to anomalylist.txt and the template scripts/infer.sh.
    We replace the product name, anomaly name and prompt in the template scripts/infer.sh.
    """
    anomaly_dict = {product: [] for product in products}
    for product in products:
        # Read anomalylist.txt and save product name and anomaly name.
        with open(anomalylist_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith(f'{product}+'):
                    anomaly = line.split('+')[1]
                    anomaly_dict[product].append(anomaly)

        # For each anomly in each product, we generate a new infer.sh.
        for product, anomalies in anomaly_dict.items(): 
            for idx, anomaly in enumerate(anomalies):
                with open(reference_infer_sh, 'r') as file:
                    content = file.read()
                    new_content = content.replace('bottle', product)  # Replace product name
                    new_content = new_content.replace('mvtec_anomaly_detection', dataset_name)  # Replace dataset name
                    new_content = new_content.replace('broken_large', anomaly)  # Replace anomaly name
                    new_content = new_content.replace('infer1.log', f'infer{idx}.log')  # Replace log file name
                    new_content = new_content.replace('sks1 sks2 sks3 sks4', f'sks{idx*4+1} sks{idx*4+2} sks{idx*4+3} sks{idx*4+4}')  # replace the prompt
                    
                    # Save the new sh file
                    infer_path_dir = f'./scripts/infer_shs/{product}'
                    new_file_path = os.path.join(infer_path_dir, f'infer{idx}.sh')
                    os.makedirs(infer_path_dir, exist_ok=True)
                    with open(new_file_path, 'w') as new_file:
                        new_file.write(new_content)

if __name__ == "__main__":
    # MVTec
    dataset_name = 'mvtec_anomaly_detection'
    anomalylist_path = './configs/product-anomaly-mvtec.txt'
    products = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    # MVTec 3D
    dataset_name = 'mvtec_3d_anomaly_detection_reorg'
    anomalylist_path = './configs/product-anomaly-mvtec3d.txt'
    products = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'foam', 'peach', 'potato', 'rope', 'tire']

    # VisA
    dataset_name = 'visa'
    anomalylist_path = './configs/product-anomaly-visa.txt'
    products = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

    reference_infer_sh = './scripts/infer.sh' # template infer.sh.
    create_infer_sh()