#!/bin/sh

set -xe

./build.sh
./perceptron
ffmpeg -y -i weights/weights-%02d.ppm demo/demo.mp4
ffmpeg -y -i demo/demo.mp4 -vf palettegen demo/palette.png
ffmpeg -y -i demo/demo.mp4 -i demo/palette.png -filter_complex "[0:v][1:v]paletteuse" demo.gif
