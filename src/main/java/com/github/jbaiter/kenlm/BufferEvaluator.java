package com.github.jbaiter.kenlm;

import com.github.jbaiter.kenlm.jni.Evaluator;
import com.github.jbaiter.kenlm.jni.Model;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CoderResult;
import java.nio.charset.StandardCharsets;

/**
 * Form an utf-8 buffer and evaluate the whole buffer at once
 */
public class BufferEvaluator {
    // this field is not accessed, but evaluator needs to keep model alive
    // so there is a reference to the model object to keep it from getting garbage collected
    private final Model model;

    private final Evaluator evaluator;

    private final ByteBuffer buffer;

    private final CharsetEncoder encoder = StandardCharsets.UTF_8.newEncoder();

    private int numTokens = 0;

    private static final byte SPACE = (byte) 0x20;

    BufferEvaluator(Model model, int bufferSize, int beamBuffer) {
        this.model = model;
        this.evaluator = new Evaluator(model, beamBuffer);
        buffer = ByteBuffer.allocateDirect(bufferSize);
    }

    /**
     * Appends the specified sequence of characters to this object.
     *
     * @param data the sequence of characters to be appended
     * @return the number of bytes remaining in the buffer
     */
    public int append(CharSequence data) {
        return append(CharBuffer.wrap(data));
    }

    public int append(CharBuffer data) {
        ByteBuffer buf = buffer;
        CoderResult coderResult = encoder.encode(data, buf, true);
        if (coderResult.isError()) {
            return -1;
        }
        if (coderResult.isOverflow()) {
            return -1;
        }
        int remaining = buf.remaining();
        if (remaining > 0) {
            buf.put(SPACE);
            remaining -= 1;
        }
        numTokens += 1;
        return remaining;
    }

    /**
     * Evaluates the model using the data in the buffer.
     *
     * @return log-probability of the input
     */
    public double evaluate() {
        ByteBuffer buf = buffer;
        int size = buf.position();
        return evaluator.evaluateSum(buf, size) / numTokens;
    }

    public double evaluateNoOutliers(float ratio) {
        ByteBuffer buf = buffer;
        int size = buf.position();
        return evaluator.evaluateNoOutliers(buf, size, ratio) / evaluator.numNonOutlierTokens();
    }

    public void clear() {
        buffer.clear();
        numTokens = 0;
    }
}
