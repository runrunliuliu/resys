/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.impl.model;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;

/**
 * <p>
 * This implementation maintains two parallel arrays, of user IDs and values.
 * The idea is to save allocating {@link org.apache.mahout.cf.taste.model.Preference} objects themselves. This
 * saves the overhead of {@link org.apache.mahout.cf.taste.model.Preference} objects but also duplicating the
 * user ID value.
 * </p>
 *
 * @see BooleanUserPreferenceArray
 * @see org.apache.mahout.cf.taste.impl.model.GenericItemPreferenceArray
 * @see GenericPreference
 */
public final class GenericUserPreferenceArray implements PreferenceArray {

    private static final int ITEM = 1;
    private static final int VALUE = 2;
    private static final int VALUE_REVERSED = 3;

    private final long[] ids;// item ids
    private long id; // user id
    private final float[] values; // rate values

    public GenericUserPreferenceArray(int size) {
        this.ids = new long[size];
        values = new float[size];
        this.id = Long.MIN_VALUE; // as a sort of 'unspecified' value
    }

    public GenericUserPreferenceArray(List<Preference> prefs) {
        this(prefs.size());
        int size = prefs.size();
        for (int i = 0; i < size; i++) {
            Preference pref = prefs.get(i);
            ids[i] = pref.getItemID();
            values[i] = pref.getValue();
        }
        if (size > 0) {
            id = prefs.get(0).getUserID();
        }
    }

    /**
     * This is a copy construct
     */
    public GenericUserPreferenceArray(long id, int size) {
        ids = null;
        id = id;
        values = new float[size];
    }

    /**
     * This is a private copy constructor for clone().
     */
    private GenericUserPreferenceArray(long[] ids, long id, float[] values) {
        this.ids = ids;
        this.id = id;
        this.values = values;
    }

    @Override
    public int length() {
        return ids.length;
    }

    @Override
    public Preference get(int i) {
        return new PreferenceView(i);
    }

    @Override
    public void set(int i, Preference pref) {
        id = pref.getUserID();
        ids[i] = pref.getItemID();
        values[i] = pref.getValue();
    }

    @Override
    public long getUserID(int i) {
        return id;
    }

    /**
     * {@inheritDoc}
     * <p/>
     * Note that this method will actually set the user ID for <em>all</em>
     * preferences.
     */
    @Override
    public void setUserID(int i, long userID) {
        id = userID;
    }

    @Override
    public long getItemID(int i) {
        return ids[i];
    }

    @Override
    public void setItemID(int i, long itemID) {
        ids[i] = itemID;
    }

    /**
     * @return all item IDs
     */
    @Override
    public long[] getIDs() {
        return ids;
    }

    @Override
    public float getValue(int i) {
        return values[i];
    }

    @Override
    public void setValue(int i, float value) {
        values[i] = value;
    }

    @Override
    public void sortByUser() {
    }

    @Override
    public void sortByItem() {
        selectionSort(ITEM);
    }

    @Override
    public void sortByValue() {
        selectionSort(VALUE);
    }

    @Override
    public void sortByValueReversed() {
        selectionSort(VALUE_REVERSED);
    }

    @Override
    public boolean hasPrefWithUserID(long userID) {
        return id == userID;
    }

    @Override
    public boolean hasPrefWithItemID(long itemID) {
        for (long id : ids) {
            if (itemID == id) {
                return true;
            }
        }
        return false;
    }

    private void selectionSort(int type) {
        // I think this sort will prove to be too dumb, but, it's in place and
        // OK for tiny, mostly sorted data
        int max = length();
        boolean sorted = true;
        for (int i = 1; i < max; i++) {
            if (isLess(i, i - 1, type)) {
                sorted = false;
                break;
            }
        }
        if (sorted) {
            return;
        }
        for (int i = 0; i < max; i++) {
            int min = i;
            for (int j = i + 1; j < max; j++) {
                if (isLess(j, min, type)) {
                    min = j;
                }
            }
            if (i != min) {
                swap(i, min);
            }
        }
    }

    private boolean isLess(int i, int j, int type) {
        switch (type) {
            case ITEM:
                return ids[i] < ids[j];
            case VALUE:
                return values[i] < values[j];
            case VALUE_REVERSED:
                return values[i] >= values[j];
            default:
                throw new IllegalStateException();
        }
    }

    private void swap(int i, int j) {
        long temp1 = ids[i];
        float temp2 = values[i];
        ids[i] = ids[j];
        values[i] = values[j];
        ids[j] = temp1;
        values[j] = temp2;
    }

    @Override
    public GenericUserPreferenceArray clone() {
        return new GenericUserPreferenceArray(ids.clone(), id, values.clone());
    }

    @Override
    public int hashCode() {
        return (int) (id >> 32) ^ (int) id ^ Arrays.hashCode(ids)
                ^ Arrays.hashCode(values);
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof GenericUserPreferenceArray)) {
            return false;
        }
        GenericUserPreferenceArray otherArray = (GenericUserPreferenceArray) other;
        return id == otherArray.id && Arrays.equals(ids, otherArray.ids)
                && Arrays.equals(values, otherArray.values);
    }

    @Override
    public Iterator<Preference> iterator() {
        return new PreferenceArrayIterator();
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder(20 * ids.length);
        result.append("GenericUserPreferenceArray[userID:");
        result.append(id);
        result.append(",{");
        for (int i = 0; i < ids.length; i++) {
            if (i > 0) {
                result.append(',');
            }
            result.append(ids[i]);
            result.append('=');
            result.append(values[i]);
        }
        result.append("}]");
        return result.toString();
    }

    private final class PreferenceArrayIterator implements Iterator<Preference> {
        private int i;

        @Override
        public boolean hasNext() {
            return i < length();
        }

        @Override
        public Preference next() {
            if (i >= length()) {
                throw new NoSuchElementException();
            }
            return new PreferenceView(i++);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    private final class PreferenceView implements Preference {

        private final int i;

        private PreferenceView(int i) {
            this.i = i;
        }

        @Override
        public long getUserID() {
            return GenericUserPreferenceArray.this.getUserID(i);
        }

        @Override
        public long getItemID() {
            return GenericUserPreferenceArray.this.getItemID(i);
        }

        @Override
        public float getValue() {
            return values[i];
        }

        @Override
        public void setValue(float value) {
            values[i] = value;
        }

    }

}