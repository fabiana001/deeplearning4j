package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by Andrea on 11/05/16.
 */
public class SkipGramWithB<T extends SequenceElement> extends SkipGram<T> {

    @Override
    public String getCodeName() {
        return "SkipGramWithB";
    }

    @Override
    public double learnSequence(@NonNull Sequence<T> sequence, @NonNull AtomicLong nextRandom,
                                @NonNull double learningRate) {
        Sequence<T> tempSequence = sequence;
        if (sampling > 0) tempSequence = super.applySubsampling(sequence, nextRandom);

        double score = 0;

        for(int i = 0; i < tempSequence.getElements().size(); i++) {
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            score = skipGram(i, tempSequence.getElements(), (int) nextRandom.get() % window, nextRandom, learningRate);
        }

        return score;
    }

    private double skipGram(int i, List<T> sentence, int b, AtomicLong nextRandom, double alpha) {
        final T word = sentence.get(i);
        if(word == null || sentence.isEmpty())
            return 0;

        double score = 0;

        int end = window - b;
        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = a - window + i;
                if(c >= 0 && c < sentence.size()) {
                    T lastWord = sentence.get(c);
                    score = super.iterateSample(word,lastWord,nextRandom,alpha);
                }
            }
        }

        return score;
    }
}
