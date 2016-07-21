package org.deeplearning4j.text.mappingTools;

import lombok.NonNull;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by Andrea on 19/05/16.
 */
public class UrlMapper implements Mapper<String, String> {

    private static final String SPLIT_WORD_READ_SOURCE = ",";
    private static final String SPLIT_WORD_MAP_SENTENCE = " ";

    private Map<String, String> map;
    private Map<String, String> reverseMap;

    public UrlMapper(@NonNull String pathFile) throws IOException {
        this(new File(pathFile), SPLIT_WORD_READ_SOURCE);
    }

    public UrlMapper(@NonNull FileReader fileReader) throws IOException {
        this(fileReader, SPLIT_WORD_READ_SOURCE);
    }

    public UrlMapper(@NonNull File file) throws IOException {
        this(new FileReader(file), SPLIT_WORD_READ_SOURCE);
    }

    public UrlMapper(@NonNull String pathFile, @NonNull String splitWord) throws IOException {
        this(new File(pathFile), splitWord);
    }

    public UrlMapper(@NonNull FileReader fileReader, @NonNull String splitWord) throws IOException {
        map = new HashMap<>();
        reverseMap = new HashMap<>();

        readSource(fileReader, splitWord);
    }

    public UrlMapper(@NonNull File file, @NonNull String splitWord) throws IOException {
        this(new FileReader(file), splitWord);
    }

    @Override
    public String mapElement(@NonNull String key) {
        return map.get(key);
    }

    @Override
    public String mapKey(@NonNull String value) {
        return reverseMap.get(value);
    }

    @Override
    public String mapSentence(@NonNull String sentence) {
        return mapSentence(sentence, SPLIT_WORD_MAP_SENTENCE);
    }

    @Override
    public String mapSentence(@NonNull String sentence, @NonNull String splitKey) {
        String result = "";
        String[] keys = sentence.split(splitKey);

        for(String aKey : keys) {
            if(isMappedElement(aKey)) {
                result += String.format("%s ", map.get(aKey));
            }
        }

        return result.trim();
    }

    @Override
    public boolean isMappedElement(@NonNull String key) {
        return map.containsKey(key);
    }

    @Override
    public boolean isMappedKey(@NonNull String value) {
        return reverseMap.containsKey(value);
    }

    @Override
    public int size() {
        return map.size();
    }

    @Override
    public String toString() {
        return map.toString();
    }

    private void readSource(FileReader fileReader, String splitWord) throws IOException {
        BufferedReader buffer = new BufferedReader(fileReader);

        String[] words;

        for(String currentLine = buffer.readLine(); currentLine != null; currentLine = buffer.readLine()) {
            words = currentLine.split(splitWord);

            String key = words[1];
            String value = words[0];

            map.put(key, value);
            reverseMap.put(value, key);
        }

        buffer.close();
    }
}
