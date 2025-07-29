#!/bin/bash

p="tsp"
d=" "
echo "{\"cmd\":\"$@\""
echo ",\"log\":["
java -cp .. "$p.$@" | grep "^\{" | while read l; do echo "  $d$l"; d=","; done
echo "]}"
