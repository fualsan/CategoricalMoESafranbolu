import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def encode_features(df_annotations, encoding_columns, all_encodings):
    for enc_col in encoding_columns:
        print(f'Column name: {enc_col}')
        num_unique_types = len(df_annotations[enc_col].unique())
        print(f'{enc_col} number of unique: {num_unique_types}')
         
        encoding = {type_id:idx for idx, type_id in enumerate(df_annotations[enc_col].unique())}
        reverse_encoding = {idx:type_id for idx, type_id in enumerate(df_annotations[enc_col].unique())}
        
        print('Encoding:', encoding)
        print('Reverse Encoding:', reverse_encoding)
        
        all_encodings[f'{enc_col}_Enc'] = encoding
        all_encodings[f'{enc_col}_Enc_Rev'] = reverse_encoding
    
        print()
    
    for enc_col in encoding_columns:
        df_annotations[f'{enc_col}_Enc'] = df_annotations[enc_col].apply(lambda x: all_encodings[f'{enc_col}_Enc'][x])


@torch.no_grad()
def generate_images(model, latent_dim, num_images=10, initial=None, expert_model_id=0):
    model.eval()
    
    _device = next(model.parameters()).device
    
    pre_latent = torch.randn(num_images, latent_dim).to(_device)

    # INITIAL OVERWRITES!!
    if initial is not None:
        pre_latent = initial.to(_device)
    
    # Generate images from latent vectors
    generated_images = model.decoder(pre_latent, expert_model_id).sigmoid().cpu()

    # Plot the generated images
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.transpose(generated_images[i].numpy(), (1, 2, 0)))
        plt.axis('off')
    plt.show()


@torch.no_grad()
def generate_single_image(model, latent_dim, initial=None):
    model.eval()

    _device = next(model.parameters()).device
    
    if initial is not None:
        pre_latent = initial.to(_device)
    else:
        pre_latent = torch.randn(1, latent_dim).to(_device)
    
    # Generate images from latent vectors
    generated_images, probs = model.decoder_all_with_predictions(pre_latent)
    generated_images = generated_images.cpu()[0]

    # Plot the generated images
    plt.figure(figsize=(8, 2))
    plt.imshow(np.transpose(generated_images.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.show()


@torch.no_grad()
def test_category(model, dataset, all_encodings, sample_idx=None):
    model.eval()

    _device = next(model.parameters()).device

    _type_enc_rev = all_encodings[f'Tip_Enc_Rev']

    if sample_idx is not None:
        sample_img, sample_feature_ids = dataset[sample_idx]
    else:
        sample_img, sample_feature_ids = dataset[random.choice(range(len(dataset)))]

    # Kat_Enc Sofa_Enc Oda_Enc Eyvan_Enc Tip_Enc
    sample_type_id = sample_feature_ids[4]

    #x_gen, category_logits, x_category_latent, sampled_type_ids
    recon_images, pred_logits, pred_latent, _ = model(sample_img.unsqueeze(0).to(_device))

    generated_images = recon_images.cpu()[0]
    pred_type = pred_logits.argmax(dim=1).cpu()[0]

    all_probs = pred_logits.cpu().softmax(dim=1).numpy()[0]
    all_probs = ' '.join([f'Tip_Enc {_type_enc_rev[idx]}: {p:0.6f} ' for idx, p in enumerate(all_probs)])
    print(f'{all_probs}')
    
    plt.figure(figsize=(8, 2))
    
    sample_type_id = _type_enc_rev[sample_type_id.item()]
    pred_type = _type_enc_rev[pred_type.item()]
    
    plt.title(f'Tip_Enc Actual: {sample_type_id}, Pred: {pred_type}')
    plt.imshow(np.transpose(generated_images.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.show()


def calculcate_silhouette_scores(num_clusters, latent_data, seed):
    best_kmeans_num_clusters = -1
    best_gmm_num_clusters = -1

    best_kmeans_score = float('-inf')
    best_gmm_score = float('-inf')
    
    for n_clusters in num_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        kmeans_labels = kmeans.fit_predict(latent_data)

        gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
        gmm_labels = gmm.fit_predict(latent_data)
        
        silhouette_kmeans = silhouette_score(latent_data, kmeans_labels)
        silhouette_gmm = silhouette_score(latent_data, gmm_labels)
        
        if silhouette_kmeans > best_kmeans_score:
            best_kmeans_score = silhouette_kmeans
            best_kmeans_num_clusters = n_clusters

        if silhouette_gmm > best_gmm_score:
            best_gmm_score = silhouette_gmm
            best_gmm_num_clusters = n_clusters
        
        print(f'Num Clusters: {n_clusters:<2} Silhouette Score K-Means: {silhouette_kmeans:.3f} GMM: {silhouette_gmm:.3f}')
    print(f'---> Best K-Means score: {best_kmeans_score:.3f}, Number of clusters: {best_kmeans_num_clusters}')
    print(f'---> Best GMM score: {best_gmm_score:.3f}, Number of clusters: {best_gmm_num_clusters}')