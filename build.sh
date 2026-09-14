#!/bin/sh

set -xe

cc -Wall -Wextra -Werror -ggdb -o perceptron main.c -lm
