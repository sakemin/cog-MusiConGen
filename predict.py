# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import random

# We need to set `TRANSFORMERS_CACHE` before any imports, which is why this is up here.
MODEL_PATH = "/src/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH

from audiocraft.data.audio import audio_write
import audiocraft.models

from typing import Optional
from cog import BasePredictor, Input, Path

# Model specific imports
import numpy as np

import torch

import subprocess

class Predictor(BasePredictor):
  def setup(self, weights: Optional[Path] = None):
    """Load the model into memory to make running multiple predictions efficient"""
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = audiocraft.models.MusicGen.get_pretrained('./ckpt/musicongen')

  def predict(
    self,
    prompt: str = Input(
      description="A description of the music you want to generate.", default="A laid-back blues shuffle with a relaxed tempo, warm guitar tones, and a comfortable groove, perfect for a slow dance or a night in. Instruments: electric guitar, bass, drums."
    ),
    text_chords: str = Input(
      description="A text based chord progression condition. Single uppercase alphabet character(eg. `C`) is considered as a major chord. Chord attributes like(`maj`, `min`, `dim`, `aug`, `min6`, `maj6`, `min7`, `minmaj7`, `maj7`, `7`, `dim7`, `hdim7`, `sus2` and `sus4`) can be added to the root alphabet character after `:`.(eg. `A:min7`) Each chord token splitted by `SPACE` is allocated to a single bar. If more than one chord must be allocated to a single bar, cluster the chords adding with `,` without any `SPACE`.(eg. `C,C:7 G, E:min A:min`)", default='C G A:min F'
    ),
    bpm: float = Input(
      description="BPM condition for the generated output. Chord and rhythm conditions are generated upon this value. This will be appended at the end of `prompt`.", default=120
    ),
    time_sig: str = Input(
      description="Meter value for the generate output. Chord and rhythm conditions are generated upon this value. This will be appended at the end of `prompt`.", default="4/4"
    ),
    duration: int = Input(
      description="Duration of the generated audio in seconds.", default=30, le=30,
    ),
    top_k: int = Input(
      description="Reduces sampling to the k most likely tokens.", default=250
    ),
    top_p: float = Input(
      description="Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used.",
      default=0.0,
    ),
    temperature: float = Input(
      description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.",
      default=1.0,
    ),
    classifier_free_guidance: int = Input(
      description="Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.",
      default=3,
    ),
    output_format: str = Input(
      description="Output format for generated audio.",
      default="wav",
      choices=["wav", "mp3"],
    ),
    seed: int = Input(
      description="Seed for random number generator. If `None` or `-1`, a random seed will be used.",
      default=None,
    ),
  ) -> Path:

    if text_chords and not bpm:
      raise ValueError("There must be `bpm` value set when text based chord conditioning.")
    if text_chords and (not time_sig or time_sig==""):
      raise ValueError("There must be `time_sig` value set when text based chord conditioning.")
    
    if prompt is None:
      prompt = ''

    if time_sig is not None and not time_sig == '':
      if prompt == '':
        prompt = time_sig
      else:
        prompt = prompt + ', ' + time_sig
    if bpm is not None:
      if prompt == '':
        prompt = str(bpm)
      else:
        prompt = prompt + f', bpm : {bpm}'

    model = self.model
    
    if duration < 30:
      extend_stride = duration // 2
    else:
      extend_stride = 15
      
    set_generation_params = lambda duration: model.set_generation_params(
      duration=duration,
      extend_stride=extend_stride,
      top_k=top_k,
      top_p=top_p,
      temperature=temperature,
      cfg_coef=classifier_free_guidance,
    )

    if not seed or seed == -1:
      seed = torch.seed() % 2 ** 32 - 1
      set_all_seeds(seed)
    set_all_seeds(seed)
    print(f"Using seed {seed}")
    
    set_generation_params(duration)

    if text_chords is None or text_chords == '':
      wav = model.generate([prompt], progress=True, return_tokens=True)
    else:
      wav = model.generate_with_chords_and_beats(descriptions = [prompt], 
                                                melody_chords = [text_chords],
                                                bpms = [bpm], 
                                                meters = [int(time_sig.split('/')[0])]
                                                )
    
    audio_write(
      "out",
      wav[0].cpu(),
      model.sample_rate,
      strategy='loudness',
      loudness_compressor=True
    )
    wav_path = "out.wav"

    if output_format == "mp3":
      mp3_path = "out.mp3"
      subprocess.call(["ffmpeg", "-i", wav_path, mp3_path])
      os.remove(wav_path)
      path = mp3_path
    else:
      path = wav_path

    return Path(path)

# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
