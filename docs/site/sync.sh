#!/bin/sh
# mirror data/*.json to data/*.js so the page also works from file:// (no server needed)
cd "$(dirname "$0")" && { printf 'window.RESULTS = '; cat data/results.json; printf ';\n'; } > data/results.js
