package OnlineKmeans

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors, Vector => MLVector}
import org.apache.spark.BLAS
import breeze.linalg.{squaredDistance, SparseVector => BSV, Vector => BV}
import org.apache.spark.sql.Dataset

object SparkMLUtils {

  // Converts Spark MLlib Vector to Breeze Vector
  private def toBreeze(v: MLVector): BV[Double] = v match {
    case dv: DenseVector => breeze.linalg.DenseVector(dv.values)
    case sv: SparseVector => new BSV[Double](sv.indices, sv.values, sv.size)
  }



  /**
   * Computes the squared Euclidean distance between two vectors.
   * Optimized to avoid full distance computation when norms are similar.
   *
   * @param v1 First vector
   * @param norm1 L2 norm of the first vector
   * @param v2 Second vector
   * @param norm2 L2 norm of the second vector
   * @return Squared Euclidean distance between the vectors
   */
  def fastSquaredDistance(
                           v1: MLVector, norm1: Double,
                           v2: MLVector, norm2: Double): Double = {

    val precision = 1e-6
    val sumSquaredNorm = norm1 * norm1 + norm2 * norm2
    val normDiff = norm1 - norm2

    if (normDiff * normDiff < precision * sumSquaredNorm) {
      val dot = BLAS.dot(v1, v2)
      math.max(sumSquaredNorm - 2.0 * dot, 0.0)
    } else {
      squaredDistance(toBreeze(v1), toBreeze(v2))
    }
  }

  /**
   * Computes clustering cost (WCSS) for any clustering model.
   *
   * @param dataset  DataFrame containing a 'features' column with ML Vectors
   * @param centers  Cluster centers from any algorithm
   * @return Total clustering cost (sum of squared distances)
   */
  def computeClusteringCost(dataset: Dataset[_], centers: Seq[MLVector]): Double = {
    val broadcastCenters = centers.map(c => new VectorWithNorm(c, Vectors.norm(c, 2.0)))

    dataset.select("features").rdd
      .map(_.get(0).asInstanceOf[MLVector])
      .map { point =>
        val p = new VectorWithNorm(point, Vectors.norm(point, 2.0))
        broadcastCenters.map(center =>
          SparkMLUtils.fastSquaredDistance(p.vector, p.norm, center.vector, center.norm)
        ).min
      }.sum()
  }

}
