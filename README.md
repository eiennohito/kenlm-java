# KenLM-Java Uzushio flavor

This is a fork of the kenlm-java project, which contains modifications for [Uzushio](https://github.com/WorksApplications/uzushio) tool.
Namely, we add `BufferEvaluator` class which provides a different API for querying the language model.
Basically, the implementation creates a space-delimited utf-8 encoded direct byte buffer and passes it to the JNI implementation
which performs the model querying purely in the C++.
By doing this, we try to minimize number of JNI calls.

We additionaly have two variants of the API:
1. Return the sum of the all log-probabilities
2. Return the sum of log-probabilities except bottom several percent.

The second implementation can be used to ignore several outliers in the input text while estimating the log-probabilities of other tokens.


# How to build from source

To build this project you need the following prerequisites:
* jdk
* swig v3+
* g++
* kenlm

The order to build this project is:
1. Run ./build.sh. This will generate the required Java JNI classes and the C++ interface using swig.
2. put the output libkenlm-jni into your java library path for running. For tests Maven automatically adds the root directory to the `java.library.path`,
3. run mvn package

The following system environment variables need to be set when running build:
* LIBRARY_PATH to have libkenlm and libkenlm_util
* CPATH to point to kenlm source, JAVA_HOME/include and JAVA_HOME/include/<arch>

The library can be renamed. If that is the case com-github-jbaiter-kenlm.properties needs to be provided pointing directly to the file. An example can be found in test resources. If the properties are not set the library is loaded from java.library.path.

## Windows

On Windows we recommend using [MSYS2](https://www.msys2.org/) toolchain, namely its `UCRT` environemt.
Build kenlm using it and then build the JNI wrapper using the `build.sh` script.
It will build a statically-linked library which can be deployed on other machines.