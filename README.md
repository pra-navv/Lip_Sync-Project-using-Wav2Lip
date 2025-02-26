# Lip Sync Project using Wav2Lip

This project utilizes **Wav2Lip** to generate a lip-synced video from a static image and an audio file. The implementation includes steps for setting up the environment, processing input files, and running inference to produce the final output.

## Table of Contents
- [Installation and Setup](#installation-and-setup)
- [Generating Video from an Image](#generating-video-from-an-image)
- [Generating Speech Audio](#generating-speech-audio)
- [Running the Wav2Lip Model](#running-the-wav2lip-model)
- [Fixing Common Errors](#fixing-common-errors)
- [Output](#output)

## Installation and Setup

Clone the Wav2Lip repository and install the required dependencies:

```bash
!git clone https://github.com/Rudrabha/Wav2Lip.git
%cd Wav2Lip
!pip install -r requirements.txt
!pip install gdown
```

Install specific versions of dependencies to ensure compatibility:

```bash
!pip install numpy==1.17.1 librosa==0.9.2
!pip install opencv-python==4.5.5.64 opencv-contrib-python==4.5.5.64
!pip install torch torchvision torchaudio
```

## Generating Video from an Image

Use OpenCV to convert an image into a video:

```python
import cv2
import numpy as np

# Load image
image_path = "/content/img.png"  # Ensure the image exists in the given path
image = cv2.imread(image_path)

# Define video properties
height, width, _ = image.shape
fps = 25
duration = 5
num_frames = fps * duration

# Create video writer
video_path = "/content/img_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# Write frames
for _ in range(num_frames):
    video_writer.write(image)

video_writer.release()
print(f"Video saved at {video_path}")
```

## Generating Speech Audio

Generate speech from text using **gTTS**:

```bash
pip install gTTS
```

```python
from gtts import gTTS

# Define the text
text = """Namaste Mathangi! My name is Anika, and I’m here to guide you through managing your credit card dues.
Mathangi, as of today, your credit card bill shows an amount due of INR 5,053 which needs to be paid by 31st December 2024.

Missing this payment could lead to two significant consequences:
First, a late fee will be added to your outstanding balance.
Second, your credit score will be negatively impacted, which may affect your future borrowing ability.

Make your payment by clicking the link here... Pay through UPI or bank transfer. Thank you!"""

# Generate speech
tts = gTTS(text=text, lang='en', tld='co.in')

# Save the audio file
audio_path = "/content/audio.mp3"
tts.save(audio_path)
```

## Running the Wav2Lip Model

Execute the following command to generate the lip-synced video:

```bash
!python /content/Wav2Lip/inference.py \
    --checkpoint_path /content/Wav2Lip/checkpoints/wav2lip_gan.pth \
    --face /content/img_video.mp4 \
    --audio /content/audio.mp3 \
    --outfile /content/output_video.mp4
```

This will generate the final lip-synced video in the `/content/` directory.

## Fixing Common Errors

### **Fix: librosa.filters.mel() Function Error**

Modify the following line in `audio.py` to explicitly specify the keyword arguments:

#### **Existing Code (Causing Error)**
```python
return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
                           fmin=hp.fmin, fmax=hp.fmax)
```

#### **Fixed Code (Explicit Keyword Arguments)**
```python
return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,
                           fmin=hp.fmin, fmax=hp.fmax)
```

#### **Why This Fix?**
- **librosa.filters.mel()** now requires explicit parameter names (`sr=`, `n_fft=`, etc.).
- By specifying them explicitly, we align with the new function definition.

## Output

After running the inference script, the final lip-synced video will be available at:

```plaintext
/content/output_video.mp4
```

---

### **Notes**
- The **Wav2Lip + GAN** model was used for improved visual quality.
- Ensure that all dependencies are installed and properly configured.
- If any additional errors occur, check the **librosa** and **gTTS** versions to match the expected compatibility.

### **References**
- [Wav2Lip GitHub Repository](https://github.com/Rudrabha/Wav2Lip)
- [Librosa Documentation](https://librosa.org/doc/latest/)
- [gTTS Documentation](https://pypi.org/project/gTTS/)

---

This README.md provides all necessary steps and explanations to successfully execute the project. 🚀


