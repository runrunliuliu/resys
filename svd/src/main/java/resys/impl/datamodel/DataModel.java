package resys.impl.datamodel;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.list.AbstractList;

public interface DataModel {

    public AbstractList getUserIDs();

    public AbstractList getItemIDs();

    public int getNumItems();

    public int getNumUsers();

    public double getPreferenceValue(int userid, int itemid);

    public int getNumOfItemsRateByUser(int userid);

    public Vector getVectorOfUsers(int itemid);

    public Vector getVectorOfItems(int userid);

    public double getMean();
}
