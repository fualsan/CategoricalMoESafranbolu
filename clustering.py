import torch
import matplotlib.pyplot as plt
import numpy as np


color_dict_str = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'cyan',
    4: 'magenta',
    5: 'yellow',
    6: 'orange',
    7: 'purple',
    8: 'gray',
    9: 'silver',
}

color_dict_rgb = {
    0: (1, 0, 0),         # red
    1: (0, 1, 0),         # green
    2: (0, 0, 1),         # blue
    3: (0, 1, 1),         # cyan
    4: (1, 0, 1),         # magenta
    5: (1, 1, 0),         # yellow
    6: (1, 0.5, 0),       # orange
    7: (0.5, 0, 1),       # purple
    8: (0.5, 0.5, 0.5),   # gray
    9: (0.75, 0.75, 0.75) # silver
}


selected_encoding_index = {
    'Floor_Enc':   0,
    'Sofa_Enc':  1,
    'Room_Enc':   2,
    'Hallway_Enc': 3,
    'Type_Enc':   4,
}


@torch.no_grad()
def generate_clusters(model, dataset, latent_dim, all_encodings, selected_encoding, device, expert_model_id=0, pre_latent=False, cached_images=None, save_figure=True):
    model.eval()

    selected_feature_index = selected_encoding_index[selected_encoding]
    
    #     0       1       2        3        4
    # Kat_Enc Sofa_Enc Oda_Enc Eyvan_Enc Tip_Enc
    selected_ids = [f[selected_feature_index].item() for i, f in dataset]
    
    if cached_images is not None:
        all_images_tensor = cached_images
    else:
        all_images_tensor = torch.stack([i for i, _ in dataset]).to(device)
    print('all_images_tensor.shape: ', all_images_tensor.shape)

    # latent, category_logits, pre_latent
    all_latent, all_category_logits, all_pre_latent = model.latent_representation(all_images_tensor, expert_model_id)
    print('all_latent.shape:', all_latent.shape)

    # use pre latents if required
    if pre_latent:
        all_latent = all_pre_latent.cpu().numpy()
    else:
        all_latent = all_latent.cpu().numpy()
    
    if latent_dim != 2:
        #pca = PCA(n_components=2)
        #all_latent_reduced = pca.fit_transform(all_latent)
        print('PCA ignored!')
    else:
        all_latent_reduced = all_latent

    print('all_latent_reduced.shape:', all_latent_reduced.shape)
    
    
    plt.figure(figsize=(10, 6))
    
    for selected_color in list(color_dict_rgb.keys())[:len(all_encodings[f'{selected_encoding}_Rev'])]:

        class_idx = np.where(selected_ids == np.array(selected_color))

        # make an exception for Sofa (keep Interior/Exterior as string)
        if selected_encoding == 'Sofa_Enc':
            _label_str = f"{selected_encoding.split('_')[0]}: {all_encodings[f'{selected_encoding}_Rev'][selected_color]}"
        else:
            _label_str = f"{selected_encoding.split('_')[0]}: {int(all_encodings[f'{selected_encoding}_Rev'][selected_color])}"
        
        plt.scatter(
            all_latent_reduced[class_idx, 0], 
            all_latent_reduced[class_idx, 1], 
            c=color_dict_str[selected_color], 
            label=_label_str,
        )
    
    plt.grid('on')
    plt.legend()
    
    if save_figure:
        if pre_latent:
            _fig_save_path = f"./saved_results/{selected_encoding.split('_')[0]}_clusters_pre_latent"
        else:
            _fig_save_path = f"./saved_results/{selected_encoding.split('_')[0]}_clusters_latent"
        plt.savefig(_fig_save_path+'.eps', format='eps', bbox_inches='tight')
        plt.savefig(_fig_save_path+'.png', format='png', bbox_inches='tight')
        plt.savefig(_fig_save_path+'.pdf', format='pdf', bbox_inches='tight')

    return all_latent_reduced, all_images_tensor.cpu()