package resys.impl.recommender;

import resys.impl.datamodel.MovieLensDataModel;
import resys.impl.svd.SVDRecommender;
import resys.impl.svdplusplus.SVDPlusPlusRecommender;

public final class MovieLensRecommendRunner {

    public static void main(String[] args) {

        MovieLensDataModel datamodel = new MovieLensDataModel(
                "data/training.txt");

        System.out.println("data model is load into memory....");

        Recommender recommender = new SVDPlusPlusRecommender(datamodel);
        // Recommender recommender = new SVDRecommender(datamodel);

        recommender.train();

    }

}
