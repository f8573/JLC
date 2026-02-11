#!/bin/bash
cd "$(dirname "$0")"
./gradlew clean compileJava --no-daemon > compile_output.txt 2>&1
echo "Exit code: $?" >> compile_output.txt
