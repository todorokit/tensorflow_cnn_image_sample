#!/bin/sh

if [ -e $1 ] ; then
    
    cat $1 | while read x; do echo -e "$RANDOM\t$x"; done | sort -k1,1n | cut -f 2- > /tmp/shuffle.txt
    cat /tmp/shuffle.txt > $1
else
    usage $0 file
fi
