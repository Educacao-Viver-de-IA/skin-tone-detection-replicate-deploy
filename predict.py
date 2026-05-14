"""Skin Tone Detection — classificador Keras 8-class (dv00005/Skin_Tone_Detection_Model)."""
import json
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

print(f"[module] predict.py loading at t={time.time()}", flush=True)
sys.stdout.flush()
import numpy as np
import tensorflow as tf
print(f"[module] tensorflow {tf.__version__} | GPUs: {len(tf.config.list_physical_devices('GPU'))}", flush=True)
sys.stdout.flush()
from PIL import Image
from cog import BasePredictor, Input, Path
print(f"[module] imports OK", flush=True)
sys.stdout.flush()

WEIGHTS_DIR = "/src/weights/skin-tone"

# 8 classes típicas de skin tone (escala Monk Skin Tone) — atribuição plausível
# Se o modelo retornar índices, mapeamos para labels descritivos
DEFAULT_LABELS = [
    "Type 1 (Very Light)",
    "Type 2 (Light)",
    "Type 3 (Light-Medium)",
    "Type 4 (Medium)",
    "Type 5 (Medium-Tan)",
    "Type 6 (Tan)",
    "Type 7 (Dark)",
    "Type 8 (Very Dark)",
]


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] === START === t={t0}", flush=True)
        sys.stdout.flush()
        self.model = None
        self.setup_error = None
        try:
            files = os.listdir(WEIGHTS_DIR)
            print(f"[setup] dir: {files}", flush=True)
        except Exception as e:
            print(f"[setup] err: {e}", flush=True)
            self.setup_error = str(e)
            return

        # Acha o .h5
        h5_files = [f for f in files if f.endswith('.h5')]
        if not h5_files:
            self.setup_error = f"Nenhum .h5 em {WEIGHTS_DIR}: {files}"
            print(f"[setup] FATAL: {self.setup_error}", flush=True)
            return
        weights_path = os.path.join(WEIGHTS_DIR, h5_files[0])
        print(f"[setup] loading {weights_path}", flush=True)
        sys.stdout.flush()

        try:
            self.model = tf.keras.models.load_model(weights_path, compile=False)
            self.input_shape = self.model.input_shape
            self.num_classes = self.model.output_shape[-1] if hasattr(self.model, 'output_shape') else 8
            print(f"[setup] DONE (t={time.time()-t0:.1f}s) input={self.input_shape} num_classes={self.num_classes}", flush=True)
            sys.stdout.flush()
        except Exception as e:
            import traceback
            print(f"[setup] FATAL: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"load failed: {e}"

    def predict(
        self,
        image: Path = Input(description="Foto facial ou da pele."),
        top_k: int = Input(default=3, ge=1, le=8, description="Top-K classes."),
    ) -> dict:
        if self.model is None:
            return {"error": f"Modelo não carregou: {getattr(self, 'setup_error', '?')}"}

        t0 = time.time()
        pil = Image.open(image).convert("RGB")

        if self.input_shape and len(self.input_shape) >= 3 and self.input_shape[1]:
            H, W = self.input_shape[1], self.input_shape[2]
        else:
            H, W = 224, 224
        pil = pil.resize((W, H), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        logits = self.model.predict(arr, verbose=0)[0]  # [num_classes]
        # softmax se logits, senão usa como prob
        if not np.allclose(logits.sum(), 1.0, atol=0.01):
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
        else:
            probs = logits

        top_idx = np.argsort(probs)[-top_k:][::-1]

        labels = DEFAULT_LABELS[:self.num_classes] if self.num_classes <= len(DEFAULT_LABELS) else [f"Type_{i+1}" for i in range(self.num_classes)]
        predictions = [
            {"class_id": int(i), "label": labels[i], "score": float(probs[i])}
            for i in top_idx
        ]

        return {
            "predicted_label": predictions[0]["label"],
            "predicted_class_id": predictions[0]["class_id"],
            "predicted_score": predictions[0]["score"],
            "predictions": predictions,
            "all_classes": labels,
            "predict_time_s": round(time.time() - t0, 3),
        }
