# =============================
# Imports and Global Constants
# =============================
import os
import gc
import io
import pickle
import datetime
#import psutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Tuple, Dict
from google.cloud import storage
from mamba_ssm import Mamba  # your Mamba
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader
from pytorch_forecasting.utils import move_to_device
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
import glob

# GCS constants
_BUCKET_NAME = "cgmproject2025"
_BASE_PREFIX = "models/predictions"
_gcs_client = storage.Client()

# =============================
# Utility Functions
# =============================
def log_memory(message=""):
    """Log CPU and GPU memory usage."""
    pid = os.getpid()
    #process = psutil.Process(pid)
    #mem_cpu = process.memory_info().rss / 1e9  # in GB
    if torch.cuda.is_available():
        mem_gpu_allocated = torch.cuda.memory_allocated() / 1e9
        mem_gpu_reserved = torch.cuda.memory_reserved() / 1e9
    else:
        mem_gpu_allocated = mem_gpu_reserved = 0.0
    print(f"[{datetime.datetime.now()}] {message}")
    #print(f"ðŸ§  CPU Mem used: {mem_cpu:.2f} GB")
    print(f"GPU Mem allocated: {mem_gpu_allocated:.2f} GB | reserved: {mem_gpu_reserved:.2f} GB")

def _last_step_summary(seq: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (h, c) by taking the last real step for each sequence."""
    idx = lengths.to(seq.device) - 1
    h = seq[torch.arange(seq.size(0), device=seq.device), idx]
    h = h.unsqueeze(0)
    return h, h.clone()

def upload_latest_checkpoint_to_gcs(local_dir, bucket_name, gcs_prefix):
    """Upload all .ckpt files in local_dir to GCS under gcs_prefix."""
    bucket = _gcs_client.bucket(bucket_name)
    for file_path in glob.glob(f"{local_dir}/*.ckpt"):
        blob = bucket.blob(f"{gcs_prefix}/{os.path.basename(file_path)}")
        blob.upload_from_filename(file_path)
        print(f"Uploaded {file_path} to gs://{bucket_name}/{gcs_prefix}/")

def download_latest_checkpoint_from_gcs(local_dir, bucket_name, gcs_prefix):
    """Download the latest .ckpt from GCS to local_dir. Returns local path or None."""
    bucket = _gcs_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    ckpt_blobs = [b for b in blobs if b.name.endswith('.ckpt')]
    if not ckpt_blobs:
        return None
    latest_blob = max(ckpt_blobs, key=lambda b: b.updated)
    local_path = os.path.join(local_dir, os.path.basename(latest_blob.name))
    latest_blob.download_to_filename(local_path)
    print(f"Downloaded {latest_blob.name} to {local_path}")
    return local_path

# =============================
# Mamba Blocks and TFT Variant
# =============================
class ResidualMambaBlock(nn.Module):
    """Norm -> Mamba -> Dropout -> Residual."""
    def __init__(self, d_model: int, dropout: float = 0.1, checkpoint: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)
        self.drop = nn.Dropout(dropout)
        self.checkpoint = checkpoint
    def _forward_inner(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mamba(self.norm(x))
        return x + self.drop(y)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_inner, x, use_reentrant=False)
        return self._forward_inner(x)

class StackedMamba(nn.Module):
    """Stack multiple residual Mamba blocks."""
    def __init__(self, d_model: int, depth: int = 4, dropout: float = 0.1, checkpoint: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualMambaBlock(d_model, dropout, checkpoint) for _ in range(depth)
        ])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

class DummyMambaAttn(nn.Module):
    """Replace MultiHeadAttention with a lightweight Mamba mixer on q only."""
    def __init__(self, d_model: int, n_head: int = 1, depth: int = 1, dropout: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.mixer = StackedMamba(d_model=d_model, depth=depth, dropout=dropout)
    def forward(self, q, k=None, v=None, mask=None):
        out = self.mixer(q)
        B, L_q, _ = q.shape
        L_k = k.size(1) if k is not None else L_q
        attn = q.new_zeros(B, self.n_head, L_q, L_k)
        return out, attn

class MambaTFT(TemporalFusionTransformer):
    """Temporal Fusion Transformer with LSTM & MHA replaced by Mamba blocks."""
    def __init__(self, *args, mamba_depth: int = 4, mamba_dropout: float = 0.1, mamba_checkpoint: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        del self.lstm_encoder
        del self.lstm_decoder
        del self.multihead_attn
        del self.static_context_initial_cell_lstm
        del self.static_context_initial_hidden_lstm
        d_model = self.hparams.hidden_size
        self.lstm_encoder = StackedMamba(
            d_model=d_model,
            depth=mamba_depth,
            dropout=mamba_dropout,
            checkpoint=mamba_checkpoint,
        )
        self.lstm_decoder = nn.Identity()
        self.multihead_attn = DummyMambaAttn(
            d_model=d_model,
            n_head=self.hparams.attention_head_size,
            depth=1,
            dropout=mamba_dropout,
        )
    def forward(self, x: Dict[str, torch.Tensor]):
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        total_lengths = encoder_lengths + decoder_lengths
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        timesteps = x_cont.size(1)
        max_encoder_length = int(encoder_lengths.max())
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update({
            name: x_cont[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.hparams.x_reals)
            if name in self.reals
        })
        if len(self.static_variables) > 0:
            static_embedding_src = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding_src)
        else:
            static_embedding = torch.zeros((x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device)
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)
        static_context_varsel = self.expand_static_context(self.static_context_variable_selection(static_embedding), timesteps)
        embeddings_varying_encoder = {n: input_vectors[n][:, :max_encoder_length] for n in self.encoder_variables}
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder, static_context_varsel[:, :max_encoder_length]
        )
        embeddings_varying_decoder = {n: input_vectors[n][:, max_encoder_length:] for n in self.decoder_variables}
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder, static_context_varsel[:, max_encoder_length:]
        )
        embeddings_full = torch.cat([embeddings_varying_encoder, embeddings_varying_decoder], dim=1)
        full_seq = self.lstm_encoder(embeddings_full)
        encoder_output = full_seq[:, :max_encoder_length]
        decoder_output = full_seq[:, max_encoder_length:]
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)
        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)
        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)
        static_context_enrich = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(lstm_output, self.expand_static_context(static_context_enrich, timesteps))
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths),
        )
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])
        output = self.pos_wise_ff(attn_output)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        if self.n_targets > 1:
            output = [layer(output) for layer in self.output_layer]
        else:
            output = self.output_layer(output)
        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )

# =============================
# Data Preparation
# =============================
def create_tft_dataloaders(train_df, horizon=12, context_length=72, batchsize=32):
    """Create PyTorch Forecasting dataloaders for TFT."""
    log_memory("Start of Dataloader Creation")
    static_categoricals = ["participant_id", "clinical_site", "study_group"]
    static_reals = ["age"]
    time_varying_known_categoricals = ["sleep_stage"]
    time_varying_known_reals = [
        "ds", "minute_of_day", "tod_sin", "tod_cos", "activity_steps", "calories_value",
        "heartrate", "oxygen_saturation", "respiration_rate", "stress_level", 'predmeal_flag',
    ]
    time_varying_unknown_reals = [
        "cgm_glucose", "cgm_lag_1", "cgm_lag_3", "cgm_lag_6", "cgm_diff_lag_1", "cgm_diff_lag_3",
        "cgm_diff_lag_6", "cgm_lagdiff_1_3", "cgm_lagdiff_3_6", "cgm_rolling_mean", "cgm_rolling_std",
    ]
    cut_off_date = train_df["ds"].max() - horizon
    training = TimeSeriesDataSet(
        train_df[train_df["ds"] < cut_off_date],
        time_idx="ds",
        target="cgm_glucose",
        group_ids=["participant_id"],
        max_encoder_length=context_length,
        max_prediction_length=horizon,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["participant_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = training.from_dataset(training, train_df, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batchsize, persistent_workers=True, num_workers=1, pin_memory=True
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batchsize, num_workers=0, persistent_workers=False, pin_memory=False
    )
    return training, val_dataloader, train_dataloader, validation

# =============================
# Model Training
# =============================

class FirstEpochTimer(pl.Callback):
    def __init__(self, local_path, gcs_bucket, gcs_prefix):
        super().__init__()
        self.local_path = local_path
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.start_time = None
        self.logged = False

    def on_train_epoch_start(self, trainer, pl_module):
        # always record, not just epoch 0
        self.start_time = datetime.datetime.now()

    def on_train_epoch_end(self, trainer, pl_module):
        # only log+upload once, still guarding with `self.logged`
        if not self.logged:
            end_time = datetime.datetime.now()
            elapsed = (end_time - self.start_time).total_seconds()
            with open(self.local_path, 'w') as f:
                f.write(f"first_epoch_runtime_seconds: {elapsed}\n")
            # Upload to GCS
            bucket = _gcs_client.bucket(self.gcs_bucket)
            blob = bucket.blob(f"{self.gcs_prefix}/first_epoch_runtime.txt")
            blob.upload_from_filename(self.local_path)
            print(f"Uploaded first epoch runtime to gs://{self.gcs_bucket}/{self.gcs_prefix}/first_epoch_runtime.txt")
            self.logged = True

class GCSCheckpointUploader(pl.Callback):
    def __init__(self, local_dir, bucket_name, gcs_prefix):
        super().__init__()
        self.local_dir = local_dir
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # This hook is called after a checkpoint is saved
        upload_latest_checkpoint_to_gcs(self.local_dir, self.bucket_name, self.gcs_prefix)

class GCSLossLogger(pl.Callback):
    def __init__(self, log_path, bucket_name, gcs_prefix, log_every_n_batches=5000):
        super().__init__()
        self.log_path = log_path
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.log_every_n_batches = log_every_n_batches
        self.batch_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_count += 1
        # Try to extract loss from outputs
        loss = None
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss'].item()
        elif hasattr(outputs, 'item'):
            loss = outputs.item()
        if self.batch_count % self.log_every_n_batches == 0 and loss is not None:
            with open(self.log_path, 'a') as f:
                f.write(f"Batch {self.batch_count}: loss={loss}\n")
            # Upload to GCS
            bucket = _gcs_client.bucket(self.bucket_name)
            blob = bucket.blob(f"{self.gcs_prefix}/loss_log.txt")
            blob.upload_from_filename(self.log_path)
            print(f"Uploaded loss log to gs://{self.bucket_name}/{self.gcs_prefix}/loss_log.txt")

def TFT_train(train_df):
    """Train the MambaTFT model and return all relevant objects, with GCS checkpointing."""
    log_memory("Start loading dataloaders")
    horizon = 12
    context_length = 576 #2days
    batchsize = 32
    training, val_dataloader, train_dataloader, validation = create_tft_dataloaders(
        train_df, horizon=horizon, context_length=context_length, batchsize=batchsize
    )
    del train_df
    gc.collect()
    log_memory("First creation of dataloaders")
    loss = QuantileLoss(quantiles=[0.2, 0.5, 0.8])
    tft = MambaTFT.from_dataset(
        training,
        learning_rate=0.003,
        hidden_size=64,
        attention_head_size=2,
        dropout=0.2,
        loss=loss,
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=4,
    )
    log_memory("Model creation before training")
    # --- ModelCheckpoint callback ---
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="mamba-tft-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_on_train_epoch_end=True
    )
    # --- First epoch timer callback ---
    first_epoch_timer = FirstEpochTimer(
        local_path="first_epoch_runtime.txt",
        gcs_bucket=_BUCKET_NAME,
        gcs_prefix="checkpoints_mamba_576"
    )
    # --- GCS checkpoint uploader callback ---
    gcs_ckpt_prefix = "checkpoints_mamba_576"
    gcs_checkpoint_uploader = GCSCheckpointUploader(
        local_dir="checkpoints",
        bucket_name=_BUCKET_NAME,
        gcs_prefix=gcs_ckpt_prefix
    )
    # --- GCS loss logger callback ---
    gcs_loss_logger = GCSLossLogger(
        log_path="loss_log.txt",
        bucket_name=_BUCKET_NAME,
        gcs_prefix=gcs_ckpt_prefix,
        log_every_n_batches=5000
    )
    # --- Download latest checkpoint from GCS if exists ---
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = download_latest_checkpoint_from_gcs("checkpoints", _BUCKET_NAME, gcs_ckpt_prefix)
    # log what happened
    if ckpt_path:
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
    else:
        print("[INFO] No checkpoint found â†’ starting fresh")
    trainer = Trainer(
        max_epochs=30,
        gradient_clip_val=0.1,
        val_check_interval=0.2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
            checkpoint_callback,
            first_epoch_timer,
            gcs_checkpoint_uploader,
            gcs_loss_logger,
        ],
        enable_progress_bar=False,
        accelerator="gpu",
        devices=4, # Use 4 GPUs
        strategy="ddp", # Distributed Data Parallel
    )
    log_memory("Before training")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,ckpt_path=ckpt_path)
    log_memory("End training")
    # --- Remove post-training upload, now handled per-epoch ---
    return tft, trainer, val_dataloader, train_dataloader, validation, training

# =============================
# Model Save/Load Utilities
# =============================
def save_tft_to_gcs(
    tft: TemporalFusionTransformer,
    model_name: str = "default_model",
    bucket_name: str = _BUCKET_NAME,
    prefix: str = _BASE_PREFIX,
):
    """Save TFT state_dict and hyperparams to GCS."""
    bucket = _gcs_client.bucket(bucket_name)
    model_prefix = f"{prefix}/{model_name}"
    buf = io.BytesIO()
    torch.save(tft.state_dict(), buf)
    buf.seek(0)
    blob = bucket.blob(f"{model_prefix}/model.pth")
    blob.upload_from_file(buf)
    buf.close()
    buf = io.BytesIO()
    pickle.dump(tft.hparams, buf)
    buf.seek(0)
    blob = bucket.blob(f"{model_prefix}/model_kwargs.pkl")
    blob.upload_from_file(buf)
    buf.close()
    print(f"Model artifacts uploaded to gs://{bucket_name}/{model_prefix}/")

def load_tft_from_gcs(
    training_dataset,
    model_name: str = "default_model",
    map_location=None,
    bucket_name: str = _BUCKET_NAME,
    prefix: str = _BASE_PREFIX,
) -> MambaTFT:
    """Fetch model_kwargs.pkl and model.pth from GCS, rebuild the TFT, load weights, and return it."""
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = _gcs_client.bucket(bucket_name)
    model_prefix = f"{prefix}/{model_name}"
    blob = bucket.blob(f"{model_prefix}/model_kwargs.pkl")
    hparams_bytes = blob.download_as_bytes()
    model_kwargs = pickle.loads(hparams_bytes)
    tft = MambaTFT.from_dataset(training_dataset, **model_kwargs)
    blob = bucket.blob(f"{model_prefix}/model.pth")
    state_bytes = blob.download_as_bytes()
    buf = io.BytesIO(state_bytes)
    state_dict = torch.load(buf, map_location=map_location)
    tft.load_state_dict(state_dict)
    buf.close()
    print(f"Loaded TFT from gs://{bucket_name}/{model_prefix}/")
    return tft

# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    # Print rank and GPU info
    rank = os.environ.get("LOCAL_RANK", "NotSet")
    gpu_count = torch.cuda.device_count()
    print(f"[INFO] RANK={rank} | CUDA devices={gpu_count}")
    # Download dataset from GCS
    client = storage.Client()
    bucket = client.bucket(_BUCKET_NAME)
    blob = bucket.blob('ai-ready/data/train_timeseries_meal.feather')
    data_bytes = blob.download_as_bytes()
    train = pd.read_feather(io.BytesIO(data_bytes))
    # Train model
    tft, trainer, val_dataloader, train_dataloader, validation, training = TFT_train(train)
    # Save model to GCS
    save_tft_to_gcs(tft, model_name="Mamba_12h_576c")