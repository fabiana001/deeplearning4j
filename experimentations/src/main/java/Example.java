import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;

/**
 * Created by Andrea on 30/04/16.
 */
public class Example {

    private static Logger log = LoggerFactory.getLogger(Example.class);

    public static void main(String[] args) throws Exception {

        File file = new File("experimentations/raw_sentences.txt");

        log.info("Load & Vectorize Sentences....");
        SentenceIterator iter = new BasicLineIterator(file);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .workers(1)
                .batchSize(1)
                .windowSize(2)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");

        Collection<String> str = vec.wordsNearest("now", 5);
        System.out.println(str);
    }
}
