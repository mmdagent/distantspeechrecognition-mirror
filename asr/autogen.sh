#!/bin/sh

rm -f config.guess config.sub ltmain.sh
export ACLOCAL="aclocal -I . -I m4 -I /usr/share/aclocal"
autoreconf --verbose -i "$@"
