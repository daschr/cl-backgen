# Background Generator

* generates a static background image from videos
* uses a opencl kernel which calculates pixel changes by comparing a frame with an history of frames 
* optional video deflickering (`--deflicker`) before backsub processing 
 
# Requirements

## Windows

* install [python](https://www.python.org/downloads/windows/)
* `pip install numpy opencv`
* install pyopencl
	* `pip install pyopencl` or 
	* download the right wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl) and f.e. `pip install pyopencl‑2021.1+cl21‑cp39‑cp39‑win_amd64.whl`

## Linux

* `pip install -r requirements.txt`

# Usage

* `python backgen.py --input <video file or url or camera>` -- generates a background image with default weight and threshold
```
usage: backgen.py [-h] [--input INPUT] [--output OUTPUT] [--vidoutput VIDOUTPUT] [--weight WEIGHT] [--threshold THRESHOLD] [--silent [SILENT]] [--sidebyside [SIDEBYSIDE]] [--deflicker [DEFLICKER]]

This program generates a static background image from videos using an opencl kernel which calculates pixel values based on averages.

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Path to a video or a sequence of image.
  --output OUTPUT       Save the last frame to this path
  --vidoutput VIDOUTPUT
                        Write video to path
  --weight WEIGHT       the weight which a new image gets merged into (0;inf]
  --threshold THRESHOLD
                        threshold which an pixel is seen as changed [1;255]
  --silent [SILENT]     silent mode, do not show images
  --sidebyside [SIDEBYSIDE]
                        show the generated background and the newest frame side by side
  --deflicker [DEFLICKER]
                        deflickers frames before processing them with backsub
```
