/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package OnlineKmeans

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasPredictionCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.linalg.{Vector => MLVector, Vectors => MLVectors}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types._
import scala.collection.mutable.ListBuffer

/**
 * Common Params for onlineKmeans and onlineKmeansModel.
 */
trait onlineKmeansParams extends Params with HasFeaturesCol with HasPredictionCol {

  final val k = new IntParam(this, "k", "The number of target clusters to create. " +
    "Must be > 1.", ParamValidators.gt(1))

  setDefault(k -> 2) //We set the default value to 2 for the number of clusters
}

/**
 *  An algorithm for online K-means clustering.
 * (the online kmeans algorithm by Liberty et al, 2015).
 */
class onlineKmeans(override val uid: String) extends Estimator[onlineKmeansModel]
  with onlineKmeansParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("onlineKmeans"))

  /**
   * Set the name of the features column.
   */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /**
   * Set the name of the predictions column.
   */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /**
   * Set the number of target clusters to create (k).
   *
   * @note It is possible for fewer than k clusters to
   * be returned, for example, if there are fewer than k distinct points to cluster. Default: 2.
   */
  def setK(value: Int): this.type = set(k -> value)

  /**
   * Number of clusters to create ( abs((k-15)/5) ).
   *
   * @note It is possible for fewer than k clusters to
   * be returned, for example, if there are fewer than k distinct points to cluster.
   */
  def getK: Int = (($(k)-15)/5).abs

  override def copy(extra: ParamMap): onlineKmeans = {
    defaultCopy(extra)
  }

  /**
   * Transforms the input schema
   *
   * @param schema
   * @return schema
   */
  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  /**
   * Computes the Squared Distances between a data point c and all of the data points
   * in the centers List. Returns a List with the distances calculated.
   *
   * @param c
   * @param centers
   * @return sqDists
   */
  def computeSquaredDistances(c: MLVector, centers: List[MLVector]): List[Double] = {
    val sqDists = for (center <- centers)
      yield MLVectors.sqdist(c, center)
    sqDists
  }

  /**
   * The fully online kmeans algorithm by Liberty et al., 2015.
   *
   * @param dataset
   */
  override def fit(dataset: Dataset[_]): onlineKmeansModel = {
    lazy val points = dataset.select($(featuresCol)).collectAsList()
    lazy val kSwitch = getK + 10
    lazy val facilityCostDoublingTrigger = getK
    lazy val facilityCostDoublingFactor = 10.0
    var facilityCost = 0.0
    var centersAddedInPhase = 0
    lazy val clusterCentersBufffer = new ListBuffer[MLVector]

    points.forEach { point =>
      def pointVector = point.get(0).asInstanceOf[MLVector]

      /**
       * The first (kSwitch) data points become cluster centers.
       * Then the distances between all the initial cluster centers are calculated
       * and half the sum of the min( kSwitch+2*(kSwitch-getK) ) is computed as the initial facility cost.
       *
       * For the rest of the data points:
       * a probability is calculated as described in the paper, and the data point becomes a new cluster center
       * based on that probability.
       * If the new clusters created exceeds the facilityCostTrigger ,
       * the facilityCost increases as described in the paper.
       */
      if (clusterCentersBufffer.length < kSwitch) {
        clusterCentersBufffer.append(pointVector)

        if (clusterCentersBufffer.length == kSwitch) {
          lazy val minInnerClusterDistances = kSwitch + 2 * (kSwitch - getK)
          lazy val initialClusterCenters = clusterCentersBufffer.toList

          val allSquaredDistances = (for (c <- initialClusterCenters)
            yield computeSquaredDistances(c, initialClusterCenters)).flatten

          val allSquaredDistancesSorted = allSquaredDistances.sorted
          facilityCost = (allSquaredDistancesSorted.take(minInnerClusterDistances).sum) / 2.0

        }
      }
      else {
        def minIndexSqDist(point: MLVector, centers: List[MLVector]) = computeSquaredDistances(point, centers).min

        lazy val p = (minIndexSqDist(pointVector, clusterCentersBufffer.toList) / facilityCost).min(1.0)
        lazy val rand = scala.util.Random
        if (rand.nextDouble() < p) {
          clusterCentersBufffer.append(pointVector)
          centersAddedInPhase += 1
        }

        if (centersAddedInPhase > facilityCostDoublingTrigger) {
          facilityCost = facilityCost * facilityCostDoublingFactor
          centersAddedInPhase = 0
        }
      }
    }

    val clusters = clusterCentersBufffer.toList
    new onlineKmeansModel(uid,clusters)
  }
}

object onlineKmeans extends DefaultParamsReadable[onlineKmeans] {
  override def load(path: String): onlineKmeans = super.load(path)
}