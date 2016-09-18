import org.deeplearning4j.models.embeddings.loader.MySerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.documentiterator.VertexLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by Andrea on 17/09/16.
 */
public class Doc2VecExperimentation {

    private static final Logger logger = LoggerFactory.getLogger(Doc2VecExperimentation.class);

    public static void main(String[] args) throws Exception {
        if (args.length != 1 && args.length != 2) {
            System.err.println("ABORTING OPERATION...Wrong args number!\n" +
                    "To run the jar do: java -cp name.jar Doc2VecExperimentation <fileToLearn> [<output>]\n" +
                    "                       | where:\n" +
                    "                       | <fileToLearn> : file to learn\n" +
                    "                       | <output> [OPTIONAL] : output path\n");

            System.exit(1);
        }

        String fileToLearn = args[0];
        String output = args.length == 1 ? getOutputFilePath(fileToLearn, "embeddings_doc2vec.txt") : args[1];

        logger.info("Creating iterator");
        VertexLineIterator iterator = new VertexLineIterator.Builder()
                    .setFilePath(fileToLearn)
                    .build();
        logger.info("Iterator created. " + iterator.size() + " vertices fetched.");

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());


        logger.info("Build Paragraph2Vec model...");
        ParagraphVectors vec = new ParagraphVectors.Builder()
                .tokenizerFactory(t)
                .minWordFrequency(1)
                .layerSize(100)
                .windowSize(5)
                .negativeSample(10)
                .trainSequencesRepresentation(false)
                .iterate(iterator)
                .build();

        logger.info("Fitting Paragraph2Vec model...");
        vec.fit();

        logger.info("Write Paragraph2Vec embeddings...");
        MySerializer.writeParagraphVectors(vec, iterator, t, output);

        logger.info("Done.");
    }

    private static String getOutputFilePath(String path, String outputFile) {
        String[] elements = path.split("/");
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < elements.length-1; i++) {
            sb.append(elements[i]).append("/");
        }

        return sb.append(outputFile).toString().trim();
    }
}
