from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import joblib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eeg_emotion.models.torch.multimodal_cvae_cnn import MultiModalCVAECNN, MultiModalCVAECNNConfig
from eeg_emotion.models.torch.cvae_model import CVAE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path
    model_type: str = "multimodal_cnn"
    device: str = "auto"
    num_classes: int = 3
    modalities: List[str] = ("filtered", "powerspec", "att", "med")
    time_steps: int = 10
    feat_dim: int = 4
    use_cvae: bool = True
    cvae_latent_dim: int = 64
    cvae_input_dim: int = 160
    cvae_checkpoint: Optional[str] = None
    dropout: float = 0.5
    scalers_dir: Optional[Path] = None
    skip_scaling: bool = False  # 跳过归一化（用于已经归一化的训练数据）


class EmotionInferenceModel:
    """情绪推理模型加载与推理模块"""

    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.device = self._init_device()
        self.scalers = self._load_scalers()
        self.model = self._load_model()
        self.class_names = ["happy", "sad", "normal"]

    def _init_device(self) -> torch.device:
        if self.cfg.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.cfg.device)

    def _load_scalers(self) -> Dict[str, any]:
        """加载训练时的归一化器"""
        scalers = {}
        
        if self.cfg.scalers_dir is None:
            logger.warning("No scalers directory provided, skipping normalization")
            return scalers
        
        scalers_dir = Path(self.cfg.scalers_dir)
        
        for mod in self.cfg.modalities:
            scaler_path = scalers_dir / f"scaler_{mod}.joblib"
            if scaler_path.exists():
                scalers[mod] = joblib.load(scaler_path)
                logger.info(f"Loaded scaler for {mod} from {scaler_path}")
            else:
                logger.warning(f"Scaler not found for {mod}: {scaler_path}")
        
        return scalers

    def _apply_scalers(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """应用归一化器 - 按照原始项目的方式"""
        result = {}
        
        # 如果配置了跳过归一化，直接返回原始数据
        if self.cfg.skip_scaling:
            logger.info("[SCALER] Skipping scaling (using pre-normalized training data)")
            for mod in self.cfg.modalities:
                result[mod] = data.get(mod, np.zeros((self.cfg.time_steps, self.cfg.feat_dim), dtype=np.float32))
            return result
        
        for mod in self.cfg.modalities:
            arr = data.get(mod, np.zeros((self.cfg.time_steps, self.cfg.feat_dim), dtype=np.float32))
            
            logger.debug(f"[SCALER] {mod} - Before scaling: mean={arr.mean():.4f}, std={arr.std():.4f}")
            
            if mod in self.scalers:
                scaler = self.scalers[mod]
                # 归一化期望 (N*T, F) - N=1个样本，T=10，F=4 → (10, 4)
                original_shape = arr.shape
                arr_reshaped = arr.reshape(-1, original_shape[1])  # (10, 4)
                arr_scaled = scaler.transform(arr_reshaped)
                result[mod] = arr_scaled.reshape(original_shape)
                logger.debug(f"[SCALER] {mod} - After scaling: mean={result[mod].mean():.4f}, std={result[mod].std():.4f}")
            else:
                result[mod] = arr
                logger.warning(f"[SCALER] {mod} - No scaler found, using raw data")
        
        return result

    def _load_model(self) -> nn.Module:
        logger.info(f"Loading model from: {self.cfg.model_path}")
        
        try:
            cvae_model = None
            if self.cfg.use_cvae:
                cvae_model = CVAE(
                    input_dim=self.cfg.cvae_input_dim,
                    num_classes=self.cfg.num_classes,
                    latent_dim=self.cfg.cvae_latent_dim,
                    hidden_dim=256,
                )
                cvae_model.eval()
                for p in cvae_model.parameters():
                    p.requires_grad = False
            
            mcfg = MultiModalCVAECNNConfig(
                modalities=list(self.cfg.modalities),
                dropout=self.cfg.dropout,
                cvae_latent_dim=self.cfg.cvae_latent_dim,
                use_cvae=self.cfg.use_cvae,
            )
            
            model = MultiModalCVAECNN(
                num_classes=self.cfg.num_classes,
                cfg=mcfg,
                cvae_model=cvae_model,
            )
            
            checkpoint = torch.load(self.cfg.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            
            model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    @torch.no_grad()
    def predict(self, data: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
        """
        执行情绪分类推理
        
        Args:
            data: 多模态数据字典，每个模态形状为 (time_steps, feat_dim)
            
        Returns:
            (emotion_label, probabilities)
        """
        logger.debug("[MODEL] Starting inference...")
        
        data = self._apply_scalers(data)
        
        x_dict = {}
        for mod in self.cfg.modalities:
            if mod not in data:
                data[mod] = np.zeros((self.cfg.time_steps, self.cfg.feat_dim), dtype=np.float32)
                logger.warning(f"[MODEL] Missing modality {mod}, using zero padding")
            
            arr = data[mod]
            if arr.shape[0] < self.cfg.time_steps:
                pad = np.zeros((self.cfg.time_steps - arr.shape[0], arr.shape[1]), dtype=np.float32)
                arr = np.vstack([arr, pad])
                logger.debug(f"[MODEL] Padded modality {mod} from {arr.shape[0]-pad.shape[0]} to {self.cfg.time_steps} steps")
            elif arr.shape[0] > self.cfg.time_steps:
                arr = arr[:self.cfg.time_steps]
                logger.debug(f"[MODEL] Truncated modality {mod} from {arr.shape[0]+self.cfg.time_steps} to {self.cfg.time_steps} steps")
            
            x_dict[mod] = torch.from_numpy(arr).unsqueeze(0).to(self.device, dtype=torch.float32)
        
        outputs = self.model(x_dict)
        logger.debug(f"[MODEL] Raw model outputs: {outputs.cpu().numpy()[0]}")
        
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        emotion = self.class_names[pred_idx]
        
        logger.debug(f"[MODEL] Probabilities - happy={probs[0]:.4f}, sad={probs[1]:.4f}, normal={probs[2]:.4f}")
        logger.info(f"[MODEL] Predicted emotion: {emotion} (confidence={probs[pred_idx]:.4f})")
        
        return emotion, probs

    @torch.no_grad()
    def predict_batch(self, batch_data: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
        """批量推理"""
        batch_data = self._apply_scalers(batch_data)
        
        batch_size = batch_data[self.cfg.modalities[0]].shape[0]
        
        x_dict = {}
        for mod in self.cfg.modalities:
            arr = batch_data[mod]
            x_dict[mod] = torch.from_numpy(arr).to(self.device, dtype=torch.float32)
        
        outputs = self.model(x_dict)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        pred_indices = np.argmax(probs, axis=1)
        emotions = [self.class_names[idx] for idx in pred_indices]
        
        return emotions, probs
