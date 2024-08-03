# Cog Implementation of MusiConGen
[![Replicate](https://replicate.com/sakemin/musicongen/badge)](https://replicate.com/sakemin/musicongen) 
This is the cog implementation of paper: "MusiConGen: Rhythm and chord control for Transformer-based text-to-music generation" in Proc. Int. Society for Music Information Retrieval Conf. (ISMIR), 2024.

MusiConGen is based on pretrained [Musicgen](https://github.com/facebookresearch/audiocraft) with additional controls: Rhythm and Chords. The project contains inference, training code and training data (youtube list). 

[Arxiv Paper](https://arxiv.org/abs/2407.15060) | [Demo](https://musicongen.github.io/musicongen_demo/) 

You can demo this model or learn how to use it with Replicate's [API](https://replicate.com/sakemin/musicongen) here.

# Run with Cog

[Cog](https://github.com/replicate/cog) is an open-source tool that packages machine learning models in a standard, production-ready container. 
You can deploy your packaged model to your own infrastructure, or to [Replicate](https://replicate.com/), where users can interact with it via web interface or API.

## Prerequisites 

**Cog.** Follow these [instructions](https://github.com/replicate/cog#install) to install Cog, or just run: 

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

Note, to use Cog, you'll also need an installation of [Docker](https://docs.docker.com/get-docker/).

* **GPU machine.** You'll need a Linux machine with an NVIDIA GPU attached and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. If you don't already have access to a machine with a GPU, check out our [guide to getting a 
GPU machine](https://replicate.com/docs/guides/get-a-gpu-machine).

## Step 1. Clone this repository

```sh
git clone https://github.com/sakemin/cog-musicongen
```
The model weight is at [link](https://huggingface.co/Cyan0731/MusiConGen/tree/main).
Move the model weight `compression_state_dict.bin` and `state_dict.bin` to directory `ckpt/musicongen`.

## Step 2. Run the model

To run the model, you need a local copy of the model's Docker image. You can satisfy this requirement by specifying the image ID in your call to `predict` like:

```
cog predict r8.im/sakemin/musicgencongen@sha256:a05ec8bdf5cc902cd849077d985029ce9b05e3dfb98a2d74accc9c94fdf15747 -i prompt="k pop, cool synthwave, drum and bass with jersey club beats" -i duration=30 -i text_chords="C G A:min F" -i bpm=140 -i time_sig="4/4"
```

For more information, see the Cog section [here](https://replicate.com/sakemin/musicongen/api#run)

Alternatively, you can build the image yourself, either by running `cog build` or by letting `cog predict` trigger the build process implicitly. For example, the following will trigger the build process and then execute prediction: 

```
cog predict -i prompt="k pop, cool synthwave, drum and bass with jersey club beats" -i duration=30 -i text_chords="C G A:min F" -i bpm=140 -i time_sig="4/4"
```

Note, the first time you run `cog predict`, model weights and other requisite assets will be downloaded if they're not available locally. This download only needs to be executed once.

# Run on replicate

## Step 1. Ensure that all assets are available locally

If you haven't already, you should ensure that your model runs locally with `cog predict`. This will guarantee that all assets are accessible. E.g., run: 

```
cog predict -i prompt="k pop, cool synthwave, drum and bass with jersey club beats" -i duration=30 -i text_chords="C G A:min F" -i bpm=140 -i time_sig="4/4"
```

## Step 2. Create a model on Replicate.

Go to [replicate.com/create](https://replicate.com/create) to create a Replicate model. If you want to keep the model private, make sure to specify "private".

## Step 3. Configure the model's hardware

Replicate supports running models on variety of CPU and GPU configurations. For the best performance, you'll want to run this model on an A100 instance.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

## Step 4: Push the model to Replicate


Log in to Replicate:

```
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 1:

```
cog push r8.im/username/modelname
```
[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)

---
# Prediction
## Prediction Parameters
- `prompt` (`string`) : A description of the music you want to generate.
- `text_chords` (`string`) : A text based chord progression condition. Single uppercase alphabet character(eg. `C`) is considered as a major chord. Chord attributes like(`maj`, `min`, `dim`, `aug`, `min6`, `maj6`, `min7`, `minmaj7`, `maj7`, `7`, `dim7`, `hdim7`, `sus2` and `sus4`) can be added to the root alphabet character after `:`.(eg. `A:min7`) Each chord token splitted by `SPACE` is allocated to a single bar. If more than one chord must be allocated to a single bar, cluster the chords adding with `,` without any `SPACE`.(eg. `C,C:7 G, E:min A:min`)
- `bpm` (`number`) : BPM condition for the generated output. Chord and rhythm conditions will be processed based on this value. This will be appended at the end of `prompt`.
- `time_sig` (`string`) : Time signature value for the generate output. Chord and rhythm conditions will be processed based on this value. This will be appended at the end of `prompt`.
- `duration` (`integer`) : Duration of the generated audio in seconds.(Default = 30, Maximum = 30)
- `top_k` (`integer`) : Reduces sampling to the k most likely tokens.(Default = 250)
- `top_p` (`number`) : Reduces sampling to tokens with cumulative probability of p. When set to `0` (default), top_k sampling is used.(Default = 0)
- `temperature` (`number`) : Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.(Default = 1)
- `classifier_free_guidance` (`integer`) : Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.(Default = 3)
- `output_format` (`string`) : Output format for generated audio.(Allowed values : `wav`, `mp3` / Default = `wav`)
- `seed` (`integer`) : Seed for random number generator. If `None` or `-1`, a random seed will be used.
  
## Text Based Chord Conditioning
### Text Chord Condition Format

```
<progression> ::= <bar> " " <bar>
<bar> ::= <chord> "," <chord>
<chord> ::= <note> ":" <shorthand>
<note> ::= <natural> | <note> <modifier>
<natural> ::= "A" | "B" | "C" | "D" | "E" | "F" | "G"
<modifier> ::= "b" | "#"
<shorthand> ::= "maj" | "min" | "dim" | "aug" | "maj7" | "min7" | "7" | "dim7" | "hdim7" | "minmaj7" | "maj6" | "min6" | "9" | "maj9" | "min9" | "sus4"
```

- `SPACE` is used as split token. Each splitted chunk is assigned to a single bar.
	-	`C G E:min A:min`
- When multiple chords must be assigned in a single bar, then append more chords with `,`.
	-	`C G,G:7 E:min,E:min7 A:min`
- Chord type can be specified after `:`.
	- 	Just using a single uppercase alphabet(eg. `C`, `E`) is considered as a major chord.
	-	 `maj`, `min`, `dim`, `aug`, `min6`, `maj6`, `min7`, `minmaj7`, `maj7`, `7`, `dim7`, `hdim7`, `sus2` and `sus4` can be appended with `:`.
		- 	eg. `E:dim`, `B:sus2`
- 'sharp' and 'flat' can be specified with `#` and `b`.
	- 	eg. `E#:min` `Db`
### BPM and Time Signature
- To create chord chroma, `bpm` and `time_sig` values must be specified.
	- `bpm` can be a float value. (eg. `132`, `60`)
	- The format of `time_sig` is `(int)/(int)`. (eg. `4/4`, `3/4`, `6/8`, `7/8`, `5/4`)
- `bpm` and `time_sig` values will be automatically concatenated after `prompt` description value, so you don't need to specify bpm or time signature information in the description for `prompt`.


## Citation
```bibtex
@misc{lan2024musicongenrhythmchordcontrol,
      title={MusiConGen: Rhythm and Chord Control for Transformer-Based Text-to-Music Generation}, 
      author={Yun-Han Lan and Wen-Yi Hsiao and Hao-Chung Cheng and Yi-Hsuan Yang},
      year={2024},
      eprint={2407.15060},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
}
```
