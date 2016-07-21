package org.deeplearning4j.text.mappingTools;

import lombok.NonNull;

/**
 * Created by Andrea on 19/05/16.
 */
public interface Mapper<Key, Value> {

    /**
     * Get the related Value by its Key
     *
     * @param key the key
     * @return the value if keys is contained in mapper, null else
     */
    Value mapElement(@NonNull Key key);

    /**
     * Get the related Key by its Value
     *
     * @param value the value
     * @return the key if value is contained in mapper, null else
     */
    Key mapKey(@NonNull Value value);

    /**
     * Map a sentence containing keys.
     * This method uses a default split key to get the keys.
     *
     * @param sentence a string containing keys
     * @return a string containing only values found, or an empty string if no values are found
     */
    String mapSentence(@NonNull String sentence);

    /**
     * Map a sentence containing keys.
     * This method uses a specified split key.
     *
     * @param sentence a string containing keys
     * @param splitKey the split key used to get the keys
     * @return a string containing only values found, or an empty string if no values are found
     */
    String mapSentence(@NonNull String sentence, @NonNull String splitKey);

    /**
     * Check if mapper contains this key
     *
     * @param key the key
     * @return true iff the key is contained into mapper, false else
     */
    boolean isMappedElement(@NonNull Key key);

    /**
     * Check if mapper contains this value
     *
     * @param value the value
     * @return true iff the value is contained into mapper, false else
     */
    boolean isMappedKey(@NonNull Value value);

    /**
     * Get the map size
     *
     * @return the map size
     */
    int size();
}
