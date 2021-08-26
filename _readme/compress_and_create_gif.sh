#!/bin/bash
for i in $(ls *png); do   pngquant -f $i -o $i; done
convert -delay 10 -loop 0 $1*png $1.gif
