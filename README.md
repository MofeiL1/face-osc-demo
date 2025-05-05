# face-osc-demo

Real-time facial expression tracking via webcam with MediaPipe and OSC output for interactive applications.

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

## Features

This application provides comprehensive facial expression tracking through 13 normalized metrics:

1. **Mouth Open**: Vertical distance between lips
2. **Mouth Width**: Horizontal mouth extension (smile detection)
3. **Left Brow Raise**: Left eyebrow height relative to eye
4. **Right Brow Raise**: Right eyebrow height relative to eye
5. **Left Eye Open**: Left eye openness
6. **Right Eye Open**: Right eye openness
7. **Brow Contraction**: Inner eyebrows distance (concentration/frown)
8. **Cheek Raise**: Cheek position (genuine smile indicator)
9. **Mouth Corner**: Mouth corners vertical position (smile/frown)
10. **Head Tilt**: Basic head rotation estimation
11. **Left Corner H**: Left mouth corner horizontal position
12. **Right Corner H**: Right mouth corner horizontal position
13. **Mouth Asymmetry**: Difference between mouth corners (smirk detection)

All measurements are normalized relative to face dimensions, making them robust against head movement and distance from camera.

## Usage

The application provides:
- Real-time webcam feed with facial landmark overlay
- On-screen metrics display showing all tracked values
- Separate visualization window showing graphs of all metrics over time
- OSC output for integration with applications like Wekinator, Max, or Processing

### Command line options

```bash
python src/face_osc.py --device 0
```

- `--device`: Camera device index (default is 0, try different numbers if your webcam isn't detected)

## OSC Output

The application sends OSC messages to `127.0.0.1:6448` (default Wekinator port) with the following path:
- `/wek/inputs`: An array of 13 normalized facial feature values

These values are normalized relative to the face dimensions, making them consistent regardless of distance from camera or face size.

## Controls

- Press `ESC` to quit the application

## Visualization

The application provides:
- Facial mesh overlay on webcam feed
- Real-time metrics display on webcam feed
- Separate graph window showing all metrics over time

## Requirements

- Python 3.10
- OpenCV
- MediaPipe
- python-osc
- matplotlib
- numpy

## License

This project is provided as-is for educational and experimental purposes.