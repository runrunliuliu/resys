package resys.impl.recommender;

import resys.impl.datamodel.DataModel;

public abstract class Recommender {

    private final DataModel dataModel;

    protected Recommender(DataModel dataModel) {
        this.dataModel = dataModel;
    }

    public DataModel getDataModel() {
        return dataModel;
    }

    /**
     *
     */
    public void train() {
    }

}
