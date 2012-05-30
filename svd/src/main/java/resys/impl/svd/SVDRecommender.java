package resys.impl.svd;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import javax.security.auth.Refreshable;

import org.apache.mahout.cf.taste.impl.model.GenericItemPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.list.AbstractList;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import resys.impl.datamodel.DataModel;
import resys.impl.recommender.Recommender;

public final class SVDRecommender extends Recommender {

    protected OpenIntDoubleHashMap bitems = new OpenIntDoubleHashMap();
    protected OpenIntDoubleHashMap busers = new OpenIntDoubleHashMap();

    protected OpenIntObjectHashMap<PreferenceArray> pusers;

    protected OpenIntObjectHashMap<PreferenceArray> qitems;

    protected static int parameter_k = 50;
    protected static int iteration = 44;
    // used of bias computation
    protected int lamda2 = 25;
    protected int lamda3 = 10;
    protected double gamma1 = 0.01;
    protected double gamma2 = 0.01;
    protected double lamda6 = 0.05;
    protected double lamda7 = 0.05;
    protected double slowrate = 0.9;

    public SVDRecommender(DataModel dataModel) {

        super(dataModel);

        // get user ids and initialize pu
        pusers = new OpenIntObjectHashMap<PreferenceArray>(
                dataModel.getNumUsers());
        IntArrayList list = (IntArrayList) dataModel.getUserIDs();
        for (int i = 0; i < list.size(); i++) {
            int key = list.get(i);
            PreferenceArray value = new GenericUserPreferenceArray(key,
                    parameter_k);
            pusers.put(key, value);
        }

        // get item ids and initialize qi;
        qitems = new OpenIntObjectHashMap<PreferenceArray>(
                dataModel.getNumItems());
        list = (IntArrayList) dataModel.getItemIDs();
        for (int i = 0; i < list.size(); i++) {
            int key = list.get(i);
            PreferenceArray value = new GenericItemPreferenceArray(key,
                    parameter_k);
            qitems.put(key, value);
        }
    }

    /**
     * initialize the bias bu and bi, the method in the page 2 of koren's
     * TKDD'09 paper
     */
    public void initBias() {

        DataModel datamodel = getDataModel();

        double mean = datamodel.getMean();
        // 1. first item bias
        IntArrayList items = (IntArrayList) datamodel.getItemIDs();
        for (int i = 0; i < items.size(); i++) {
            int itemid = items.get(i);
            Vector tmp = datamodel.getVectorOfUsers(itemid);
            Iterator<Element> iter = tmp.iterateNonZero();
            double rate = 0.0;
            while (iter.hasNext()) {
                rate += (iter.next().get() - mean);
            }
            bitems.put(itemid, rate / (tmp.getNumNondefaultElements() + lamda2));
            // bitems.put(itemid, 0);
        }

        // 2. second user bias
        IntArrayList users = (IntArrayList) datamodel.getUserIDs();
        for (int i = 0; i < users.size(); i++) {
            int userid = users.get(i);
            Vector tmp = datamodel.getVectorOfItems(userid);
            Iterator<Element> iter = tmp.iterateNonZero();
            double rate = 0.0;
            while (iter.hasNext()) {
                Element e = iter.next();
                rate += (e.get() - mean - bitems.get(e.index()));
            }
            busers.put(userid, rate / (tmp.getNumNondefaultElements() + lamda3));
            // busers.put(userid, 0);
        }
    }

    /**
     * Initialize the matrix of user character(P), the matrix of item
     * character(Q) and implicit matrix Y
     */
    public void initPQ() {

        // Initialize user-factor matrix P
        DataModel datamodel = getDataModel();
        IntArrayList users = (IntArrayList) datamodel.getUserIDs();
        for (int i = 0; i < users.size(); i++) {
            int userid = users.get(i);
            PreferenceArray pa = pusers.get(userid);
            for (int j = 0; j < parameter_k; j++) {
                float randvalue = (float) (0.1 * Math.random() / Math
                        .sqrt(parameter_k));
                pa.setValue(j, randvalue);
            }
        }

        // Initialize item-factor matrix q and y
        IntArrayList items = (IntArrayList) datamodel.getItemIDs();
        for (int i = 0; i < items.size(); i++) {
            int itemid = items.get(i);
            PreferenceArray paItem = qitems.get(itemid);
            for (int j = 0; j < parameter_k; j++) {
                float randvalue = (float) (0.1 * Math.random() / Math
                        .sqrt(parameter_k));
                paItem.setValue(j, randvalue);
            }
        }
    }

    /**
     * Main training procedure
     */
    @Override
    public void train() {

        DataModel datamodel = getDataModel();

        initBias();
        initPQ();

        for (int iter = 0; iter < iteration; iter++) {

            double rmse = 0.0;
            int n = 0;
            IntArrayList users = (IntArrayList) datamodel.getUserIDs();

            for (int u = 0; u < users.size(); u++) {
                // get userid
                int userid = users.get(u);

                Vector tmpitems = datamodel.getVectorOfItems(userid);

                PreferenceArray UserFactor = pusers.get(userid);

                // iterate to items
                Iterator<Element> itor = tmpitems.iterateNonZero();
                while (itor.hasNext()) {

                    Element e = itor.next();
                    int itemid = e.index();

                    // actual rating and estimated rating
                    double rui = datamodel.getPreferenceValue(userid, itemid);
                    double pui = predictPreference(userid, itemid);
                    double eui = rui - pui;

                    rmse += eui * eui;
                    n++;

                    // update bias values
                    double tmp = busers.get(userid) + gamma1
                            * (eui - lamda6 * busers.get(userid));
                    busers.put(userid, tmp);

                    tmp = bitems.get(itemid) + gamma1
                            * (eui - lamda6 * bitems.get(itemid));
                    bitems.put(itemid, tmp);

                    // update user factor and movie factor vectors
                    PreferenceArray ItemFactor = qitems.get(itemid);
                    for (int k = 0; k < parameter_k; k++) {

                        // user factor,pu
                        float preval = UserFactor.getValue(k);
                        float temp = ItemFactor.getValue(k);
                        preval = (float) (preval + gamma2
                                * (eui * temp - lamda7 * preval));
                        UserFactor.setValue(k, preval);

                        // item factor,qi
                        preval = ItemFactor.getValue(k);
                        temp = UserFactor.getValue(k);
                        preval = (float) (preval + gamma2
                                * (eui * temp - lamda7 * preval));
                        ItemFactor.setValue(k, preval);
                    }
                }
            }

            rmse = Math.sqrt(rmse / n);

            System.out.println("iteration: " + iter + " rmse: " + rmse + " "
                    + n);
            lamda6 *= slowrate;
            lamda7 *= slowrate;
        }
    }

    public double dot(PreferenceArray a, PreferenceArray b, int dim) {

        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            sum += a.getValue(i) * b.getValue(i);
        }
        return sum;
    }

    /**
     * Compute preference based on userid and itemid
     *
     * @param userID
     * @param itemID
     * @return
     */
    public double predictPreference(int userID, int itemID) {

        DataModel datamodel = getDataModel();

        // number of items which are rated by the userid
        int uRateItemNum = datamodel.getNumOfItemsRateByUser(userID);

        double mean = datamodel.getMean();
        double result;
        if (uRateItemNum >= 1) {
            result = mean + busers.get(userID) + bitems.get(itemID)
                    + dot(pusers.get(userID), qitems.get(itemID), parameter_k);
        } else {
            result = mean + busers.get(userID) + bitems.get(itemID);
        }
        if (result < 1.0)
            result = 1;
        if (result > 5.0)
            result = 5;

        return result;
    }
}
