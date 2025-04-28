# face-osc-demo

Proof-of-concept: capture webcam, detect landmarks via MediaPipe, send OSC to Wekinator.

## Quickstart

```bash
conda activate face-osc-demo
python src/face_osc.py
```

## Installation

1. Clone this repository
2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

## Usage

The application captures your webcam feed, detects facial landmarks, and sends facial feature data via OSC:
- A single message `/wek/inputs` containing raw, unprocessed landmark distances:
  - Index 0: eyebrow raise (distance between eyebrow and eye points)
  - Index 1: mouth openness (vertical distance between lip points)

### Command line options

```bash
python src/face_osc.py --device 0
```

- `--device`: Camera device index (default is 0, try different numbers if your webcam isn't detected)

## OSC Output

The application sends OSC messages to `127.0.0.1:6448` (default Wekinator port) with the following path:
- `/wek/inputs`: A list containing [brow_raise, mouth_open] values as raw, unprocessed data

The values represent:
- `brow_raise`: The raw distance between eyebrow and eye points (typically small values around 0.02-0.05)
- `mouth_open`: The raw vertical distance between top and bottom lip landmarks (typically small values around 0.01-0.1)

These values are sent as full precision 32-bit floats without any scaling, normalization, or clamping. This provides the rawest possible data directly from the facial landmark detection.

## Controls

- Press `ESC` to quit the application

## Requirements

- Python 3.10
- OpenCV
- MediaPipe
- python-osc
- matplotlib
- numpy

## License

This project is provided as-is for educational and experimental purposes.
