package org.deeplearning4j.text.documentiterator;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Andrea on 17/09/16.
 */
public class VertexLineIterator implements LabelAwareIterator {

    private LabelsSource ids;
    private List<LabelledDocument> labelledDocuments;
    private int i;

    public VertexLineIterator(List<LabelledDocument> labelledDocuments, LabelsSource ids) {
        this.labelledDocuments = labelledDocuments;
        this.ids = ids;
        this.i = 0;
    }

    @Override
    public boolean hasNextDocument() {
        return i < labelledDocuments.size();
    }

    @Override
    public LabelledDocument nextDocument() {
        LabelledDocument document = labelledDocuments.get(i);
        i++;

        return document;
    }

    @Override
    public void reset() {
        i = 0;
    }

    @Override
    public LabelsSource getLabelsSource() {
        return ids;
    }

    public int size() {
        return labelledDocuments.size();
    }

    public static class Builder {

        private String filePath;
        private String splitKey = "\\t";

        public Builder() {

        }

        public Builder setFilePath(String val) {
            filePath = val;
            return this;
        }

        public Builder setSplitKey(String val) {
            splitKey = val;
            return this;
        }

        public VertexLineIterator build() throws IOException {
            if (filePath == null) {
                throw new NullPointerException("filePath is null");
            }

            List<String> ids = new ArrayList<>();
            List<LabelledDocument> labelledDocuments = new ArrayList<>();

            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            String line;
            String[] parts;

            while ((line = reader.readLine()) != null) {
                parts = line.split(splitKey);

                String id = parts[0];
                String text = parts[1];

                LabelledDocument doc = new LabelledDocument();

                ids.add(id);
                doc.setContent(text);
                doc.setLabel(id);
                labelledDocuments.add(doc);
            }

            LabelsSource source = new LabelsSource(ids);
            return new VertexLineIterator(labelledDocuments, source);
        }
    }
}
