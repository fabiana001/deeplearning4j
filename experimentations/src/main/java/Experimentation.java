import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGramNoB;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGramWithB;
import org.deeplearning4j.models.embeddings.loader.MySerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.mappingTools.Mapper;
import org.deeplearning4j.text.mappingTools.UrlMapper;
import org.deeplearning4j.text.sentenceiterator.CodeLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

import java.io.IOException;

/**
 * Created by Andrea on 13/07/16.
 */
public class Experimentation {

    public static void main(String[] args) throws IOException {
        if (args.length != 6 && args.length != 7) {
            System.err.println("ABORTING OPERATION...Wrong args number!\n" +
                    "To run the jar do: java -cp name.jar Experimentation <urlMap> <sentences> <outputEmbedding> " +
                    "<learningAlgorithm> <windowSize> <iteractions> [<separator_urlMap>]\n" +
                    "                       | where:\n" +
                    "                       | <urlMap> : file containing codeUrl and url\n" +
                    "                       | <sentences> : sentences to learn\n" +
                    "                       | <outputEmbedding> : file containing embeddings\n" +
                    "                       | <learningAlgorithm> : use skipgramWithB for skipgram only left context " +
                    "with b value" +
                    " or use skipgramNoB for skipgram only left context without b value\n" +
                    "                       | <windowSize> : set an integer for window size\n" +
                    "                       | <iteractions> : set an integer for iteractions\n" +
                    "                       | <separator_urlMap> : OPTIONAL, separator for read url map");
            System.exit(1);
        }

        //set parameters
        String urlMapFile = args[0];
        String sentencesFile = args[1];
        String outputEmbeddingsFile = args[2];
        String learningAlgorithm = args[3];
        int windowSize = Integer.parseInt(args[4]);
        int iteractions = Integer.parseInt(args[5]);
        String separator;
        try {
            separator = args[6];
        } catch(IndexOutOfBoundsException e) {
            separator = null;
        }

        System.out.println("Mapping urls and ids...");
        Mapper<String, String> mappedUrl = separator != null ? new UrlMapper(urlMapFile, separator)
                : new UrlMapper(urlMapFile);
        System.out.println("Built a mapper with " + mappedUrl.size() + " words");

        System.out.println("Load & Vectorize Sentences...");
        SentenceIterator iterator = new CodeLineIterator(sentencesFile, mappedUrl);

        ElementsLearningAlgorithm<VocabWord> learningAlgorithmObj = getLearningAlgorithm(learningAlgorithm);

        System.out.println("Building model...");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .iterations(iteractions)
                .layerSize(100)
                .negativeSample(10)
                .windowSize(windowSize)
                //.seed(42)
                .elementsLearningAlgorithm(learningAlgorithmObj)
                .iterate(iterator)
                //.tokenizerFactory(t)
                .build();

        System.out.println("Fitting Word2Vec model....");
        vec.fit();

        System.out.println("Write embeddings...");
        //WordVectorSerializer.writeWordVectors(vec, args[2]);
        MySerializer.writeWordVectors(vec, outputEmbeddingsFile, mappedUrl);

        System.out.println("Done.");
    }

    private static ElementsLearningAlgorithm<VocabWord> getLearningAlgorithm(String arg) {
        ElementsLearningAlgorithm<VocabWord> result = null;

        if (arg.equalsIgnoreCase("skipgramWithB")) {
            result = new SkipGramWithB<>();
            System.out.println("Using SkipGram only left context WITH B VALUE");
        } else if (arg.equalsIgnoreCase("skipgramNoB")) {
            result = new SkipGramNoB<>();
            System.out.println("Using SkipGram only left context WITHOUT B VALUE");
        } else {
            result = new SkipGram<>();
            System.err.println("Using normal skipgram for Wrong argument...");
        }

        return result;
    }
}
