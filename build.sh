#!/usr/bin/env bash
set -e

JNIPATH="./src/main/java/com/github/jbaiter/kenlm/jni"
SWIGPATH="./src/main/swig/KenLM.swg"

mkdir -p $JNIPATH

swig -c++ -java -Wall -package com.github.jbaiter.kenlm.jni \
     -outdir $JNIPATH -o kenlm_wrap.cc $SWIGPATH

CXXFLAGS="-I. -O3 -DNDEBUG -DHAVE_BZLIB -DKENLM_MAX_ORDER=6 -fPIC $CXXFLAGS"

if [[ "$OSTYPE" == "darwin"* ]]; then
    g++ $CXXFLAGS ./kenlm_wrap.cc \
        -dynamiclib \
        $objects -ldl -shared -lkenlm -lkenlm_util \
        -Wno-deprecated -pthread -o libkenlm-jni.dylib
elif [[ "$OSTYPE" == "msys"* ]]; then
    g++ $CXXFLAGS ./kenlm_wrap.cc \
        $objects -static -static-libgcc -lkenlm -lkenlm_util -lz -lbz2 -llzma \
        -Wno-deprecated -pthread -shared -o kenlm-jni.dll
else
    g++ $CXXFLAGS ./kenlm_wrap.cc \
        $objects -ldl -shared -lkenlm -lkenlm_util \
        -Wno-deprecated -pthread -o libkenlm-jni.so
fi
