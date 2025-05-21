import math
import time
import matplotlib.pyplot as plt
import gzip
import seaborn as sns
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from urllib.parse import unquote
import numpy as np
from keras import Sequential, layers, Model
from collections import Counter

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras import layers
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical
from keras import mixed_precision
import os

import torch
from torch import nn
from layer_HGNN import HGNN_conv
import torch.nn.functional as F



# === Paths ===
def read_data():
    import pandas as pd
    import numpy as np
    import os
    import gzip
    from urllib.parse import unquote
    from Bio import SeqIO
    from Bio.Seq import Seq
    from sklearn.preprocessing import StandardScaler

    dna_dir = "dataset/dataset1/dna_chromosomes/"
    gff3_dir = "dataset/dataset1/gff3_files/"

    # === Collect all FASTA and GFF3 files ===
    fasta_files = sorted([
        os.path.join(dna_dir, f) for f in os.listdir(dna_dir)
        if f.lower().endswith(".fa.gz")
    ])
    gff3_files = sorted([
        os.path.join(gff3_dir, f) for f in os.listdir(gff3_dir)
        if f.lower().endswith(".gff3")
    ])

    # === Parse GFF3 attributes ===
    def parse_attributes(attr_str):
        attr_dict = {}
        for pair in attr_str.strip().split(";"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                attr_dict[key.strip()] = unquote(value.strip())
        return attr_dict

    # === Function to parse GFF3 and extract CDS entries for a given chromosome ===
    def parse_gff3_cds(gff3_file, chrom_id):
        cds_dict = {}
        with open(gff3_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue
                if parts[2] != "CDS":
                    continue
                if parts[0] != chrom_id:
                    continue

                start = int(parts[3]) - 1  # GFF3 is 1-based; convert to 0-based
                end = int(parts[4])  # end is exclusive
                strand = parts[6]
                attrs = parse_attributes(parts[8])
                parent_id = attrs.get("Parent", "NA")

                if parent_id not in cds_dict:
                    cds_dict[parent_id] = {
                        "strand": strand,
                        "ranges": []
                    }
                cds_dict[parent_id]["ranges"].append((start, end))
        return cds_dict

    # === Extract CDS sequences with start/end positions ===
    cds_sequences = []

    for fasta_path, gff_path in zip(fasta_files, gff3_files):
        print(f"Processing: {os.path.basename(fasta_path)} with {os.path.basename(gff_path)}")

        # Read chromosome sequence
        with gzip.open(fasta_path, "rt", encoding="utf-8") if fasta_path.endswith(".gz") else open(fasta_path, "r",
                                                                                                   encoding="utf-8") as f:
            record = next(SeqIO.parse(f, "fasta"))

        chrom_seq = record.seq
        chrom_id = record.id

        # Parse CDS entries
        cds_dict = parse_gff3_cds(gff_path, chrom_id)
        print(f"Found {len(cds_dict)} CDS transcripts in {os.path.basename(gff_path)}")

        for parent_id, info in cds_dict.items():
            strand = info["strand"]
            regions = sorted(info["ranges"], key=lambda x: x[0])
            full_seq = "".join(str(chrom_seq[start:end]) for start, end in regions)
            if strand == "-":
                full_seq = str(Seq(full_seq).reverse_complement())

            transcript_start = min(start for start, end in regions)
            transcript_end = max(end for start, end in regions)

            cds_sequences.append({
                "transcript_id": parent_id,
                "chrom": chrom_id,
                "strand": strand,
                "start": transcript_start,
                "end": transcript_end,
                "sequence": full_seq
            })

    df_cds = pd.DataFrame(cds_sequences)
    df_cds.to_csv("dataset/dataset1/cds_sequences.csv", index=False)

    # === Load mapping and expression data ===
    alias_df = pd.read_csv("dataset/dataset1/alias/genes_to_alias_ids.tsv", sep="\t", header=None)
    tpm_df = pd.read_csv("dataset/dataset1/geo_file/abundance.tsv", sep="\t")

    alias_df.columns = ["e_id", "source", "d_id", "agpv4_id"]
    tpm_df["gene_id"] = tpm_df["target_id"].apply(lambda x: x.split("_T")[0])
    df_cds["clean_gene_id"] = df_cds["transcript_id"].apply(lambda x: x.replace("transcript:", "").split("_T")[0])

    df_merged = df_cds.merge(alias_df[["e_id", "d_id"]], left_on="clean_gene_id", right_on="e_id", how="left")

    avg_tpm = tpm_df.groupby("gene_id", as_index=False)["tpm"].mean().rename(columns={"tpm": "avg_tpm"})
    final_df = df_merged.merge(avg_tpm, left_on="d_id", right_on="gene_id", how="left")

    final_df = final_df.drop(columns=["clean_gene_id", "e_id", "gene_id"])
    final_df.dropna(subset=['avg_tpm'], inplace=True)

    # === Normalize start and end columns using z-score
    scaler = StandardScaler()
    final_df[["start_z", "end_z"]] = scaler.fit_transform(final_df[["start", "end"]])

    # === Encode transcript_id, strand, sequence
    final_df["transcript_index"] = pd.factorize(final_df["transcript_id"])[0]
    final_df["strand_numeric"] = final_df["strand"].map({'+': 1, '-': 0})

    def encode_sequence(seq):
        mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        return [mapping.get(base.upper(), 4) for base in seq]

    final_df["sequence_encoded"] = final_df["sequence"].apply(encode_sequence)

    # === Convert avg_tpm to expression labels
    def compute_labels(tpm_array):
        mean_tpm = np.mean(tpm_array)
        low = mean_tpm / 2
        high = mean_tpm * 1.5
        return np.array([0 if t < low else 1 if t < high else 2 for t in tpm_array], dtype=np.int32)

    final_df["expression_label"] = compute_labels(final_df["avg_tpm"].values)

    # === Pad or truncate sequences
    FIXED_LEN = 6000
    PAD_VALUE = 4

    def pad_or_truncate(seq):
        return seq[:FIXED_LEN] if len(seq) > FIXED_LEN else seq + [PAD_VALUE] * (FIXED_LEN - len(seq))

    final_df['sequence_encoded'] = final_df['sequence_encoded'].apply(pad_or_truncate)

    # === Final feature selection and saving
    # Save X and y
    np.save('dataset/dataset1/sequence_encoded.npy', np.array(final_df['sequence_encoded'].tolist(), dtype=np.int32))
    np.save('dataset/dataset1/expression_label.npy', final_df['expression_label'].values.astype(np.int32))

    # Save other features (excluding sequence, label, and transcript index)
    other_features = final_df.drop(columns=['sequence_encoded', 'expression_label', 'transcript_index','transcript_id',"sequence","avg_tpm","start","end","strand","d_id"])
    np.save('dataset/dataset1/other_features.npy', other_features.values)

    # === Logging info ===
    print("Unique label values:", final_df["expression_label"].unique())
    print("Transcript ID sample:", final_df["transcript_id"].head())
    print("other_features shape:", other_features.shape)
    print("sequence_encoded shape:", np.load('dataset/dataset1/sequence_encoded.npy').shape)
    print("expression_label shape:", np.load('dataset/dataset1/expression_label.npy').shape)


# read_data()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# # === Load Only First 50 Samples Directly ===
# sequence_data = np.load('dataset/dataset1/sequence_encoded.npy')[:100, :6000]
# expression_labels = np.load("dataset/dataset1/expression_label.npy")[:100].astype(int)
# node_features = np.load('dataset/dataset1/other_features.npy',allow_pickle=True)[:2000]  # shape (2000, 4)





# Optional: Convert to tensors
import torch
balanced_seq=np.load("dataset/dataset1/balanced_data/sequence_balanced.npy")[:100,:6000]
balanced_label=np.load("dataset/dataset1/balanced_data/labels_balanced.npy")[:100]
balanced_features=np.load("dataset/dataset1/balanced_data/other_features_balanced.npy",allow_pickle=True)[:1000]
sequence_tensor = torch.tensor(balanced_seq, dtype=torch.long)
labels_tensor = torch.tensor(balanced_label, dtype=torch.long)
other_features = np.array(balanced_features.tolist(), dtype=np.float32)



unique_labels, counts = np.unique(balanced_label, return_counts=True)
print("Labels:", unique_labels)
print("Counts:", counts)


print("Node features shape:",other_features.shape)
# # === Create Shared Hypergraph Matrix ===
# def create_random_hypergraph(num_nodes, num_hyperedges, connection_prob=0.1):
#     return np.random.rand(num_nodes, num_hyperedges) < connection_prob
#
# shared_G = create_random_hypergraph(50, 50).astype(np.float32)  # (50, 50)
# shared_G_batch = tf.convert_to_tensor(shared_G[np.newaxis, ...])  # (1, 50, 50)
#
# # === Train/Test Split (same for transformer and HGNN) ===
# indices = np.arange(50)
# train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=expression_labels, random_state=42)
#
# x_train_dna = sequence_data[train_idx]
# x_test_dna = sequence_data[test_idx]
# x_train_node = node_features[train_idx]
# x_test_node = node_features[test_idx]
# y_train = expression_labels[train_idx]
# y_test = expression_labels[test_idx]
#
# from sklearn.model_selection import train_test_split
#
# # Split 20% of the training data for validation
# x_train_dna, x_val_dna, x_train_node, x_val_node, y_train, y_val = train_test_split(
#     x_train_dna,
#     x_train_node,
#     y_train,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_train  # Keep class distribution balanced
# )
x=sequence_tensor #shape(100,6000)
y=labels_tensor #shape(100)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# import numpy as np
# from collections import Counter
# from sklearn.metrics import accuracy_score
#
# class KNN:
#     def __init__(self, k):
#         self.k = k
#         print(f"KNN initialized with k = {self.k}")
#
#     def fit(self, X_train, y_train):
#         if self.k > len(X_train):
#             raise ValueError("k cannot be greater than the number of training samples")
#         self.x_train = np.array(X_train)
#         self.y_train = np.array(y_train).flatten()
#
#     def calculate_euclidean(self, sample1, sample2):
#         return np.linalg.norm(sample1.astype(np.float32) - sample2.astype(np.float32))
#
#     def nearest_neighbors(self, test_sample):
#         distances = [
#             (self.y_train[i], self.calculate_euclidean(self.x_train[i], test_sample))
#             for i in range(len(self.x_train))
#         ]
#         distances.sort(key=lambda x: x[1])  # Sort by distance
#         neighbors = [distances[i][0] for i in range(self.k)]
#         return neighbors
#
#     def majority_vote(self, neighbors):
#         count = Counter(neighbors)
#         return sorted(count.items(), key=lambda x: (-x[1], x[0]))[0][0]
#
#     def predict(self, test_set):
#         predictions = []
#         for test_sample in test_set:
#             neighbors = self.nearest_neighbors(test_sample)
#             prediction = self.majority_vote(neighbors)
#             predictions.append(prediction)
#         return predictions
# #KNN Model Building
# # === Apply KNN to your dataset ===
# # Make sure x_train, x_test, y_train, y_test are already defined
# #
# # model3 = KNN(k=5)
# # model3.fit(x_train, y_train)
# # predictions = model3.predict(x_test)
# #
# # accuracy = accuracy_score(y_test, predictions)
# # print(f"Accuracy for KNN: {accuracy:.4f}")
#
#
#
#
#
# def create_BiLSTM(input_shape, num_classes):
#     model = Sequential()
#     model.add(Bidirectional(LSTM(units=64,
#                                  return_sequences=False,
#                                  activation='tanh'),
#                             input_shape=input_shape))
#     model.add(Dense(units=num_classes, activation='softmax'))
#
#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                   metrics=["accuracy"])
#     return model
#
# #BiLSTM Model Buildimg
#
# # # Number of classes in your classification
# # num_classes = 3  # e.g., low = 0, medium = 1, high = 2
#
#
# # model2 = create_BiLSTM(x_train_bilstm.shape[1:], num_classes)
# # model2.fit(x_train_bilstm, y_train, epochs=10, batch_size=32, validation_split=0.1)
# #
# #
# # loss_bilstm, acc_bilstm = model2.evaluate(x_test_bilstm, y_test)
# # print("The loss of BiLSTM:", loss_bilstm)
# # print("The accuracy of BiLSTM:", acc_bilstm)
#
#
#
# def compute_metrics(y_true, y_pred, average='macro'):
#     cm = confusion_matrix(y_true, y_pred)
#     tp = np.diag(cm)
#     fn = cm.sum(axis=1) - tp
#     fp = cm.sum(axis=0) - tp
#     tn = cm.sum() - (tp + fn + fp)
#     specificity = np.mean(tn / (tn + fp)) if np.all(tn + fp) else 0.0
#
#     return {
#         "confusion_matrix": cm,
#         "accuracy": accuracy_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
#         "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
#         "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
#         "specificity": specificity,
#         "mae": mean_absolute_error(y_true, y_pred),
#         "mse": mean_squared_error(y_true, y_pred)
#     }
#
# # === Main Evaluation Loop ===
# results = {"ProposedModel": [], "KNN": [], "BiLSTM": []}
# metrics = {"ProposedModel": [], "KNN": [], "BiLSTM": []}
# training_percentage = [40, 50, 60, 70, 80, 90]
#
# #
# x_all = np.concatenate([x_train, x_test], axis=0)
# y_all = np.concatenate([y_train, y_test], axis=0)
#
# for percent in training_percentage:
#     print(f"\n=== Training with {percent}% of total data ===")
#     indices = np.arange(len(x_all))
#     np.random.shuffle(indices)
#     num_train = int(len(x_all) * percent / 100)
#
#
#     train_idx = indices[:num_train]
#     test_idx = indices[num_train:]
#
#
#     def get_generators(x_all, y_all, node_features_reduced, hg_adj, train_idx, test_idx, batch_size=32):
#         x_train = x_all[train_idx]
#         y_train = y_all[train_idx]
#         x_test = x_all[test_idx]
#         y_test = y_all[test_idx]
#
#         node_features_train = node_features_reduced[train_idx]
#         node_features_test = node_features_reduced[test_idx]
#         hg_adj_train = hg_adj[train_idx]
#         hg_adj_test = hg_adj[test_idx]
#
#         # Create data generators
#         train_gen = HybridDataGenerator(x_train, node_features_train, hg_adj_train, y_train,
#                                         batch_size=batch_size, shuffle=True)
#         test_gen = HybridDataGenerator(x_test, node_features_test, hg_adj_test, y_test,
#                                        batch_size=batch_size, shuffle=False)
#
#         return train_gen, test_gen, y_test
#
#
#     print(
#         f"x_train: {x_train.shape}, node_features_train: {node_features_train.shape}, hg_adj_train: {hg_adj_train.shape}")
#     print(f"x_test: {x_test.shape}, node_features_test: {node_features_test.shape}, hg_adj_test: {hg_adj_test.shape}")
#
#     # --- Proposed Model ---
#     train_gen, test_gen, y_test_split = get_generators(x_all, y_all, node_features_reduced, hg_adj, train_idx, test_idx,
#                                                        batch_size=32)
#
#     # --- Proposed Model ---
#     combined_model = CombinedModel(diffusion_transformer, hgnn_embedding, fusion_dim=64, num_classes=3)
#     combined_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                            loss='sparse_categorical_crossentropy',
#                            metrics=['accuracy'])
#
#     combined_model.fit(train_gen, validation_data=test_gen, epochs=10, verbose=1)
#
#     # Predict on test set using generator
#     y_pred_probs = combined_model.predict(test_gen)
#     y_pred = np.argmax(y_pred_probs, axis=1)
#     metric_vals = compute_metrics(y_test, y_pred)
#     results["ProposedModel"].append(metric_vals["accuracy"])
#     metrics["ProposedModel"].append(metric_vals)
#     print(f"ProposedModel Accuracy: {metric_vals['accuracy']:.4f}")
#
#     # #--- BiLSTM Model---
#     # x_train_bilstm = np.expand_dims(x_train, axis=-1).astype(np.float32)
#     # x_test_bilstm = np.expand_dims(x_test, axis=-1).astype(np.float32)
#     # model2 = create_BiLSTM(x_train_bilstm.shape[1:], num_classes=3)
#     # model2.fit(x_train_bilstm, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
#     # y_pred = np.argmax(model2.predict(x_test_bilstm), axis=-1)
#     # metric_vals = compute_metrics(y_test, y_pred)
#     # results["BiLSTM"].append(metric_vals["accuracy"])
#     # metrics["BiLSTM"].append(metric_vals)
#     # print(f"BiLSTM Accuracy: {metric_vals['accuracy']:.4f}")
#     #
#     # # --- KNN ---
#     # knn_model = KNN(k=5)
#     # knn_model.fit(x_train, y_train)
#     # y_pred = knn_model.predict(x_test)
#     # metric_vals = compute_metrics(y_test, y_pred)
#     # results["KNN"].append(metric_vals["accuracy"])
#     # metrics["KNN"].append(metric_vals)
#     # print(f"KNN Accuracy: {metric_vals['accuracy']:.4f}")
#
# # === Save results ===
# np.save("model_accuracy_results.npy", results)
# np.save("model_detailed_metrics.npy", metrics)
#
# # === Accuracy Plot ===
# bar_width = 0.2
# x_range = np.arange(len(training_percentage))
# model_names = list(results.keys())
# plt.figure(figsize=(12, 6))
#
# for i, model_name in enumerate(model_names):
#     plt.bar(x_range + i * bar_width, results[model_name], width=bar_width, label=model_name)
#
# plt.xlabel("Training Percentage")
# plt.ylabel("Accuracy")
# plt.title("Model Accuracy vs Training Data Percentage")
# plt.xticks(x_range + bar_width, training_percentage)
# plt.legend()
# plt.tight_layout()
# plt.savefig("training_percentage_comparison_bar.png")
# plt.show()
#
# # === Confusion Matrices ===
# for model_name in model_names:
#     for i, percent in enumerate(training_percentage):
#         cm = metrics[model_name][i]["confusion_matrix"]
#         plt.figure(figsize=(6, 5))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.title(f"{model_name} Confusion Matrix ({percent}%)")
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.tight_layout()
#         plt.savefig(f"conf_matrix_{model_name}_{percent}.png")
#         plt.close()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage to avoid GPU overload

# Simplified version for small dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# HGNN convolution layer
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    def forward(self, X, H):
        # X: [N, in_ft], H: [N, M] incidence matrix
        H = H.float()
        # Degrees
        v_degree = torch.sum(H, dim=1)  # [N]
        e_degree = torch.sum(H, dim=0)  # [M]
        # Avoid division by zero
        v_degree = torch.where(v_degree > 0, v_degree, torch.ones_like(v_degree))
        e_degree = torch.where(e_degree > 0, e_degree, torch.ones_like(e_degree))
        Dv_inv_sqrt = torch.diag(1.0 / torch.sqrt(v_degree))
        De_inv = torch.diag(1.0 / e_degree)
        # Hypergraph adjacency approximation: Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
        HT = H.transpose(0,1)  # [M, N]
        out = Dv_inv_sqrt @ H @ De_inv @ HT @ Dv_inv_sqrt @ X @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

# HGNN model with two layers
class HGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(HGNN, self).__init__()
        self.conv1 = HGNN_conv(in_feats, hidden_feats)
        self.conv2 = HGNN_conv(hidden_feats, out_feats)
        self.relu = nn.ReLU()
    def forward(self, X, H):
        x = self.conv1(X, H)
        x = self.relu(x)
        x = self.conv2(x, H)
        return x  # [N, out_feats]

# Linformer-based self-attention (single head)
class LinformerSelfAttention(nn.Module):
    def __init__(self, embed_dim, seq_len, k_proj):
        super(LinformerSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.k = k_proj
        # Projection matrices for key and value along sequence dimension
        self.H_k = nn.Parameter(torch.randn(seq_len, k_proj))
        self.H_v = nn.Parameter(torch.randn(seq_len, k_proj))
        # Linear to produce Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        B, L, D = x.size()
        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each [B, L, D]
        # Project keys and values along sequence
        # K_proj: [B, k_proj, D], V_proj: [B, k_proj, D]
        K_proj = torch.einsum('lk,bld->bkd', self.H_k, k)
        V_proj = torch.einsum('lk,bld->bkd', self.H_v, v)
        # Compute scaled dot-product attention
        scores = torch.einsum('bld,bkd->blk', q, K_proj) / (D ** 0.5)  # [B, L, k_proj]
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum('blk,bkd->bld', weights, V_proj)  # [B, L, D]
        return out

# Transformer encoder layer with Linformer attention
class LinformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, seq_len, k_proj):
        super(LinformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = LinformerSelfAttention(embed_dim, seq_len, k_proj)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    def forward(self, x):
        # x: [B, seq_len, embed_dim]
        # Self-attention block
        x2 = self.norm1(x)
        attn_out = self.attn(x2)
        x = x + attn_out
        # Feed-forward block
        x2 = self.norm2(x)
        ff_out = self.ff(x2)
        x = x + ff_out
        return x  # [B, seq_len, embed_dim]

# DiT1D model: Linformer-based Transformer for 1D sequence
class DiT1D(nn.Module):
    def __init__(self, input_dim, embed_dim, seq_len, k_proj, num_layers):
        super(DiT1D, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, embed_dim))
        self.layers = nn.ModuleList([
            LinformerEncoderLayer(embed_dim, seq_len, k_proj)
            for _ in range(num_layers)
        ])
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        B, L, _ = x.size()
        # Embed and add positional encoding
        x = self.embedding(x) + self.pos_embed.unsqueeze(0).expand(B, -1, -1)
        # Apply Transformer layers
        for layer in self.layers:
            x = layer(x)
        # Pooling: mean over sequence
        out = x.mean(dim=1)  # [B, embed_dim]
        return out

# Fusion module: projects two branches and applies attention-based fusion and MLP
class FusionModule(nn.Module):
    def __init__(self, dim_h, dim_d, fuse_dim, num_classes):
        super(FusionModule, self).__init__()
        self.proj_h = nn.Linear(dim_h, fuse_dim)
        self.proj_d = nn.Linear(dim_d, fuse_dim)
        # Attention scoring for two branches
        self.score = nn.Linear(fuse_dim, 1)
        # MLP for classification
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.ReLU(),
            nn.Linear(fuse_dim // 2, num_classes)
        )
    def forward(self, h, d):
        # h: [B, dim_h], d: [B, dim_d]
        h_proj = self.proj_h(h)  # [B, fuse_dim]
        d_proj = self.proj_d(d)  # [B, fuse_dim]
        # Compute attention weights for each branch
        score_h = self.score(h_proj)  # [B, 1]
        score_d = self.score(d_proj)  # [B, 1]
        scores = torch.cat([score_h, score_d], dim=1)  # [B, 2]
        weights = F.softmax(scores, dim=1)  # [B, 2]
        # Weighted fusion
        fused = weights[:, 0:1] * h_proj + weights[:, 1:2] * d_proj  # [B, fuse_dim]
        out = self.classifier(fused)  # [B, num_classes]
        return out

# Load dataset
graph_features = np.load('dataset/dataset1/other_features.npy',allow_pickle=True)[:100]  # shape (100, feat_dim)
dna_data = np.load('dataset/dataset1/sequence_encoded.npy')[:100,:6000]  # shape (100, 6000)
labels = np.load('dataset/dataset1/expression_label.npy')[:100]  # shape (100,)
graph_features = np.array(graph_features.tolist(), dtype=np.float32)
graph_features = torch.tensor(graph_features)
dna_data = torch.tensor(dna_data, dtype=torch.float)
labels = torch.tensor(labels, dtype=torch.long)
print("the shape of dna_data:",dna_data.shape)
# Prepare DNA sequence tensor: assume one-hot has 4 channels, flatten 6000 -> (1500,4)
# Adjust if needed according to actual encoding
seq_len = 6000
channels = dna_data.size(1) // seq_len
dna_data = dna_data.view(-1, seq_len, channels)

# Generate hypergraph incidence matrix from labels (10-node hyperedges per class)
num_nodes = labels.size(0)
unique_labels = torch.unique(labels)
num_classes = int(unique_labels.max().item()) + 1
# We will create one hyperedge per class, each with up to 10 nodes
H = torch.zeros(num_nodes, num_classes)
for cls in unique_labels.tolist():
    cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
    if cls_indices.numel() > 10:
        chosen = cls_indices[:10]
    else:
        chosen = cls_indices
    H[chosen, cls] = 1

# Split into train/test indices (80/20 split)
torch.manual_seed(0)
indices = torch.randperm(num_nodes)
split = int(0.8 * num_nodes)
train_idx = indices[:split]
test_idx = indices[split:]

# Define model dimensions
in_feats = graph_features.size(1)
hgnn_hidden = 16
hgnn_out = 8
dit_input = channels  # e.g. 4
dit_embed = 32
dit_k = 128
dit_layers = 2
fusion_dim = 16
num_classes = 3

# Instantiate models
hgnn = HGNN(in_feats, hgnn_hidden, hgnn_out)
dit = DiT1D(input_dim=dit_input, embed_dim=dit_embed, seq_len=seq_len, k_proj=dit_k, num_layers=dit_layers)
fusion = FusionModule(hgnn_out, dit_embed, fusion_dim, num_classes)

# Optimizer and loss
params = list(hgnn.parameters()) + list(dit.parameters()) + list(fusion.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(1, 11):
    hgnn.train()
    dit.train()
    fusion.train()
    optimizer.zero_grad()
    # Forward pass (full batch)
    g_emb = hgnn(graph_features, H)  # [N, hgnn_out]
    d_emb = dit(dna_data)  # [N, dit_embed]
    outputs = fusion(g_emb, d_emb)  # [N, num_classes]
    # Compute loss on training subset
    loss = criterion(outputs[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    # Compute accuracies
    hgnn.eval()
    dit.eval()
    fusion.eval()
    with torch.no_grad():
        g_emb = hgnn(graph_features, H)
        d_emb = dit(dna_data)
        outputs = fusion(g_emb, d_emb)
        preds = torch.argmax(outputs, dim=1)
        train_acc = (preds[train_idx] == labels[train_idx]).float().mean().item()
        test_acc = (preds[test_idx] == labels[test_idx]).float().mean().item()
    print(f"Epoch {epoch}: Loss={loss.item():.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
from sklearn.metrics import classification_report


