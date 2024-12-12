#!/bin/bash
if [ ! -f target/release/rua ]; then
    cargo build --release
fi
./target/release/rua
python draw.py
