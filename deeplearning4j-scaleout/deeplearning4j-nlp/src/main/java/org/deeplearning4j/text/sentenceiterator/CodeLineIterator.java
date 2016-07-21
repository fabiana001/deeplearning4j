package org.deeplearning4j.text.sentenceiterator;

import lombok.NonNull;
import org.deeplearning4j.text.mappingTools.Mapper;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;

/**
 * Created by Andrea on 19/05/16.
 */
public class CodeLineIterator extends BasicLineIterator {

    private Mapper<String, String> mapper;

    public CodeLineIterator(@NonNull File file, Mapper<String, String> mapper) throws FileNotFoundException {
        super(file);

        this.mapper = mapper;
    }

    public CodeLineIterator(@NonNull InputStream stream, Mapper<String, String> mapper) {
        super(stream);

        this.mapper = mapper;
    }

    public CodeLineIterator(@NonNull String filePath, Mapper<String, String> mapper) throws FileNotFoundException {
        super(filePath);

        this.mapper = mapper;
    }

    @Override
    public synchronized String nextSentence() {
        try {
            String sentence = reader.readLine();
            String mappedSentence = mapper.mapSentence(sentence);
            return (preProcessor != null) ? preProcessor.preProcess(mappedSentence) : mappedSentence;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
