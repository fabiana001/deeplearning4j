package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by Andrea on 11/05/16.
 */
public class SkipGramNoB<T extends SequenceElement> extends SkipGram<T> {

    @Override
    public String getCodeName() {
        return "SkipGramNoB";
    }

    @Override
    public double learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom, @NonNull double learningRate) {
        Sequence<T> tempSequence = sequence;
        if (sampling > 0) tempSequence = super.applySubsampling(sequence, nextRandom);

        double score = 0;

        for(int i = 0; i < tempSequence.getElements().size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            score = skipGram(i, tempSequence.getElements(), nextRandom, learningRate);
        }

        return score;
    }

    private double skipGram(int i, List<T> sentence, AtomicLong nextRandom, double alpha) {
        final T word = sentence.get(i);
        if(word == null || sentence.isEmpty()) {
            return 0;
        }

        double score = 0;

        int start = i - window;
        int end = i;
        for(int a = start; a < end; a++) {
            if(a >= 0 && a < sentence.size()) {
                T lastWord = sentence.get(a);
                score = super.iterateSample(word, lastWord, nextRandom, alpha);
            }
        }

        return score;
    }
}
