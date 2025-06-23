import sounddevice as sd
import numpy as np
import queue
import os
import json
import onnxruntime as ort
import torch
import torchaudio.transforms as T
from librosa import resample
import sentencepiece as spm

class StandaloneASR:
    def __init__(self, model_dir: str = "model_components"):
        self.model_dir = model_dir
        self.device = "cuda" if ort.get_device() == "GPU" else "cpu"

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")

        onnx_path = os.path.join(model_dir, "hrsvrn-Hindi-speech-to-text.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file {onnx_path} not found")
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"]
        )

        decoder_config_path = os.path.join(model_dir, "decoder_config.json")
        if not os.path.exists(decoder_config_path):
            raise FileNotFoundError(f"Decoder config {decoder_config_path} not found")
        with open(decoder_config_path, "r") as f:
            self.decoder_config = json.load(f)

    def preprocess_audio_chunk(self, audio_chunk: np.ndarray, fs: int) -> tuple:
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)
        audio_chunk = audio_chunk.astype(np.float32)

        if fs != 16000:
            audio_chunk = resample(audio_chunk, orig_sr=fs, target_sr=16000)

        audio = torch.from_numpy(audio_chunk).float()

        mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=0.0,
            f_max=8000.0,
            n_mels=80,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False
        )

        audio = audio.unsqueeze(0)
        mel_spec = mel_transform(audio)
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)

        audio_features = mel_spec.numpy().astype(np.float16)
        audio_length = np.array([mel_spec.shape[2]], dtype=np.int64)

        expected_shape = (1, 80, audio_features.shape[2])
        if audio_features.shape != expected_shape:
            raise ValueError(f"Expected audio features shape {expected_shape}, got {audio_features.shape}")

        return audio_features, audio_length

    def run_inference(self, audio_features: np.ndarray, audio_length: np.ndarray) -> np.ndarray:
        input_names = [inp.name for inp in self.session.get_inputs()]
        inputs = {
            input_names[0]: audio_features,
            input_names[1]: audio_length
        }
        outputs = self.session.run(None, inputs)
        return outputs[0]

    def decode_output(self, logits: np.ndarray) -> str:
        if isinstance(logits, torch.Tensor):
            logits = logits.numpy()

        predictions = np.argmax(logits, axis=-1).squeeze(0)

        blank_id = self.decoder_config.get("blank_idx", logits.shape[-1] - 1)
        decoded = []
        previous = blank_id
        for p in predictions:
            if p != blank_id and p != previous:
                decoded.append(int(p))
            previous = p

        vocab_path = os.path.join(self.model_dir, "tokenizer_hi.vocab")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file {vocab_path} not found")

        vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip().split('\t')[0]
                id_hi = int(line.strip().split('\t')[1].replace("-", "")) + 1537
                vocab[id_hi] = token

        text = ''.join([vocab[id] if id in vocab else '<UNK>' for id in decoded])
        text = text.replace('▁', ' ').strip()
        return text

def transcribe_from_mic(model_dir="model_components", duration=5):
    fs = 16000
    asr = StandaloneASR(model_dir=model_dir)
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, callback=callback, blocksize=int(fs * duration)):
        print("Recording... Speak now.")
        audio_chunk = q.get()
        try:
            features, length = asr.preprocess_audio_chunk(audio_chunk, fs)
            logits = asr.run_inference(features, length)
            text = asr.decode_output(logits)
            return text
        except Exception as e:
            return f"[ERROR] {str(e)}"
