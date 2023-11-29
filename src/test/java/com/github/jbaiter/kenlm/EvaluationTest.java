package com.github.jbaiter.kenlm;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.function.Consumer;

public class EvaluationTest {
    private Model model;

    @Before
    public void setUp() throws ModelException {
        model = new Model(Resources.resource("/test.arpa"));
    }

    @Test
    public void evaluateSomething() {
        BufferEvaluator e = model.bufferEvaluator(2048, 10);
        e.append("a");
        e.append("little");
        e.append("more");
        e.append(".");
        double score = e.evaluate();
        Assert.assertEquals(score, -0.65889, 1e-3);
    }

    @Test
    public void evaluateWithoutOutliers() {
        BufferEvaluator e = model.bufferEvaluator(2048, 3);
        e.append("looking");
        e.append("on");
        e.append("a");
        e.append("little");
        e.append("little");
        e.append("little");
        e.append("little");
        e.append("more");
        e.append("loin");
        double score1 = e.evaluate();
        double score2 = e.evaluateNoOutliers(0.2f);
        double score3 = e.evaluateNoOutliers(0.4f);
        Assert.assertTrue(score2 > score1);
        Assert.assertTrue(score3 > score2);
    }

    @Test
    public void evaluateWithoutOutliersAndLargeBuffer() {
        BufferEvaluator e = model.bufferEvaluator(2048, 16);
        e.append("looking");
        e.append("on");
        e.append("a");
        e.append("little");
        e.append("little");
        e.append("little");
        e.append("little");
        e.append("more");
        e.append("loin");
        double score1 = e.evaluate();
        double score2 = e.evaluateNoOutliers(0.2f);
        double score3 = e.evaluateNoOutliers(0.4f);
        Assert.assertTrue(score2 > score1);
        Assert.assertTrue(score3 > score2);
    }

    @Test
    public void smallBufferAndLargeBufferProduceSameResults() {
        BufferEvaluator e1 = model.bufferEvaluator(2048, 3);
        BufferEvaluator e2 = model.bufferEvaluator(2048, 12);
        Consumer<BufferEvaluator> fnc = (e) -> {
            e.append("looking");
            e.append("on");
            e.append("a");
            e.append("little");
            e.append("little");
            e.append("little");
            e.append("little");
            e.append("more");
            e.append("loin");
        };
        fnc.accept(e1);
        fnc.accept(e2);

        Assert.assertEquals(e1.evaluateNoOutliers(0.2f), e2.evaluateNoOutliers(0.2f), 1e-3);
        Assert.assertEquals(e1.evaluateNoOutliers(0.4f), e2.evaluateNoOutliers(0.4f), 1e-3);
    }
}
