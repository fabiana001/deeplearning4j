package org.deeplearning4j.models.embeddings.loader;

import com.sun.istack.internal.NotNull;
import lombok.NonNull;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.paragraphvectors.tools.MeansBuilder;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.mappingTools.Mapper;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by Andrea on 13/07/16.
 */
public class MySerializer {

    private static final Logger log = LoggerFactory.getLogger(MySerializer.class);

    public static void writeWordVectors(@NonNull Word2Vec vec, @NonNull String path,
                                        @NonNull Mapper<String, String> mapper) throws IOException {
        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path), false));

        writeWordVectors(vec, write, mapper);

        write.flush();
        write.close();
    }

    public static void writeParagraphVectors(@NonNull ParagraphVectors vec,
                                             @NonNull LabelAwareIterator iterator,
                                             @NonNull TokenizerFactory tokenizerFactory,
                                             @NonNull String path) throws IOException {
        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path), false));

        writeParagraphVectors(vec, iterator, tokenizerFactory, write);

        write.flush();
        write.close();
    }

    private static void writeWordVectors(@NonNull Word2Vec vec, @NonNull BufferedWriter writer,
                                         @NonNull Mapper<String, String> mapper) throws IOException {
        int words = 0;
        String id;

        for (String word : vec.vocab().words()) {
            if (word == null) {
                continue;
            }
            StringBuilder sb = new StringBuilder();
            id = mapper.mapKey(word);
            sb.append(id);
            sb.append(" ");
            INDArray wordVector = vec.getWordVectorMatrix(word);
            for (int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if (j < wordVector.length() - 1) {
                    sb.append(" ");
                }
            }
            sb.append("\n");
            writer.write(sb.toString());
            words++;
        }

        log.info("Wrote " + words + " with size of " + vec.lookupTable().layerSize());
    }

    private static void writeParagraphVectors(@NonNull ParagraphVectors vec,
                                              @NonNull LabelAwareIterator iterator,
                                              @NonNull TokenizerFactory tokenizerFactory,
                                              @NonNull BufferedWriter writer) throws IOException {
        int words = 0;
        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>) vec.getLookupTable(),
                tokenizerFactory);

        iterator.reset();

        while (iterator.hasNextDocument()) {
            LabelledDocument document = iterator.nextDocument();
            StringBuilder builder = new StringBuilder();
            builder.append(document.getLabel());
            builder.append(" ");

            INDArray vector = meansBuilder.documentAsVector(document);
            for (int j = 0; j < vector.length(); j++) {
                builder.append(vector.getDouble(j));
                if (j < vector.length() - 1) {
                    builder.append(" ");
                }
            }
            builder.append("\n");
            writer.write(builder.toString());
            words++;
        }

        log.info("Wrote " + words + " with size of " + vec.lookupTable().layerSize());
    }
}
