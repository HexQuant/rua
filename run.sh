#!/bin/bash
export HTTPS_PROXY="socks5://127.0.0.1:1080"
if [ ! -f target/release/rua ]; then
    cargo build --release
fi
./target/release/rua
source ~/venv/myds313/bin/activate
python draw.py
