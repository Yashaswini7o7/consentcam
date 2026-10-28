#!/usr/bin/env bash
# Group photos from the CMU/MIT frontal face test set (shipped in opencv_extra).
# Research-use images; we don't redistribute them, this just pulls them locally.
set -e
mkdir -p data && cd data
if [ ! -d opencv_extra ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/opencv/opencv_extra
  (cd opencv_extra && git sparse-checkout set testdata/cv/cascadeandhog/images)
fi
ln -sfn opencv_extra/testdata/cv/cascadeandhog/images group_photos
echo "test images in data/group_photos/"
