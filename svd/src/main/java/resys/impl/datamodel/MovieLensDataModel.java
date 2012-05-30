package resys.impl.datamodel;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.list.AbstractList;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntIntHashMap;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.apache.mahout.math.set.OpenHashSet;
import org.apache.mahout.math.set.OpenIntHashSet;

import com.javamex.classmexer.MemoryUtil;
import com.javamex.classmexer.MemoryUtil.VisibilityFilter;

public class MovieLensDataModel implements DataModel {

    protected OpenIntObjectHashMap<Vector> userItemMatrix = new OpenIntObjectHashMap<Vector>();
    protected OpenIntObjectHashMap<Vector> itemUserMatrix = new OpenIntObjectHashMap<Vector>();

    protected IntArrayList users = new IntArrayList();
    protected IntArrayList items = new IntArrayList();

    protected double meanRating = 0.0;

    /**
     * Read file into memory model
     *
     * @param filepath
     */
    public MovieLensDataModel(String filepath) {

        BufferedReader reader;

        OpenIntHashSet tusers = new OpenIntHashSet();
        OpenIntHashSet titems = new OpenIntHashSet();

        String line;
        try {
            reader = new BufferedReader(new FileReader(filepath));
            line = reader.readLine();

            int n = 0;
            double score = 0.0;
            while (line != null) {

                n++;
                if (n % 1000 == 0)
                    System.out.println("now iteration: " + n);

                String[] tmp = line.split("\t");
                int userid = Integer.parseInt(tmp[0]);
                int itemid = Integer.parseInt(tmp[1]);
                double rating = Double.parseDouble(tmp[2]);

                tusers.add(userid);
                titems.add(itemid);

                score += rating;

                // Initialize user-item rating matrix
                Vector tmp_vector = null;
                if (userItemMatrix.containsKey(userid)) {
                    tmp_vector = userItemMatrix.get(userid);
                    tmp_vector.setQuick(itemid, rating);
                } else {
                    tmp_vector = new SequentialAccessSparseVector(400);
                    tmp_vector.setQuick(itemid, rating);
                    userItemMatrix.put(userid, tmp_vector);
                }

                // Initialize item-user rating matrix
                tmp_vector = null;
                if (itemUserMatrix.containsKey(itemid)) {
                    tmp_vector = itemUserMatrix.get(itemid);
                    tmp_vector.setQuick(userid, rating);
                } else {
                    tmp_vector = new SequentialAccessSparseVector(400);
                    tmp_vector.setQuick(userid, rating);
                    itemUserMatrix.put(itemid, tmp_vector);
                }

                // read the next line
                line = reader.readLine();
            }
            reader.close();

            meanRating = score / n;

        } catch (IOException e) {
            e.printStackTrace();
        }

        users = tusers.keys();
        items = titems.keys();

        tusers = null;
        titems = null;
    }

    /**
     * Initialized uRatingNum and iRatedNum Initialized mean
     */
    public void StatMatrix() {

        int num = 0;
        double score = 0.0;
        for (int i = 0; i < users.size(); i++) {

            int userid = users.get(i);

            Vector tmp = userItemMatrix.get(userid);
            Iterator<Element> iter = tmp.iterateNonZero();

            while (iter.hasNext()) {
                Element el = iter.next();
                int itemid = el.index();
            }
        }
    }

    @Override
    public int getNumUsers() {
        return users.size();
    }

    @Override
    public int getNumItems() {
        return items.size();
    }

    @Override
    public AbstractList getItemIDs() {
        return items;
    }

    @Override
    public AbstractList getUserIDs() {
        return users;
    }

    @Override
    public double getPreferenceValue(int userid, int itemid) {
        return this.userItemMatrix.get(userid).getQuick(itemid);
    }

    @Override
    public int getNumOfItemsRateByUser(int userid) {
        return this.userItemMatrix.get(userid).getNumNondefaultElements();
    }

    @Override
    public Vector getVectorOfUsers(int itemid) {
        return this.itemUserMatrix.get(itemid);
    }

    @Override
    public Vector getVectorOfItems(int userid) {
        return this.userItemMatrix.get(userid);
    }

    @Override
    public double getMean() {
        return meanRating;
    }

    public static void main(String[] args) throws IOException {

        int mb = 1024 * 1024;

        MovieLensDataModel datamodel = new MovieLensDataModel("training.txt");

        System.out.println(datamodel.getNumItems());
        System.out.println(datamodel.getNumUsers());
        // Print used memory
        System.out.println("Used Memory:"
                + (Runtime.getRuntime().totalMemory() - Runtime.getRuntime()
                .freeMemory()) / mb);
    }

}
