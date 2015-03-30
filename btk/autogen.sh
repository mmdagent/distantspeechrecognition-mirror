#!/bin/sh

export ACLOCAL="aclocal -I m4"
autoreconf -v -i "$@"
