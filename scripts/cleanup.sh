#!/bin/bash

find . -iname "__pycache__" -type d -exec rm -r "{}" \;
find . -iname ".DS_Store" -type f -exec rm -r "{}" \;
