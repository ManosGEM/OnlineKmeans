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

import org.apache.spark.ml.Model
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.ml.linalg.{Vector => MLVector, Vectors}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}

import SparkMLUtils.fastSquaredDistance

import scala.collection.mutable.ListBuffer


case class onlineKmeansModel(override val uid: String, clusters: List[VectorWithNorm])
  extends Model[onlineKmeansModel] with onlineKmeansParams with DefaultParamsWritable{

  def this(clusters: List[VectorWithNorm]) = this(Identifiable.randomUID("onlineKmeansModel"),clusters)

  override def copy(extra: ParamMap): onlineKmeansModel = {
    defaultCopy(extra)
  }

  /**
   * Computes the Squared Distances between a data point c and all of the data points
   * in the centers List. Returns a List with the distances calculated.
   *
   * @param c
   * @param centers
   * @return sqDists
   */
  def computeSquaredDistances(point: VectorWithNorm, centers: List[VectorWithNorm]): List[Double] = {
    val sqDists = for (center <- centers)
      yield fastSquaredDistance(point.vector,point.norm,center.vector,center.norm)
    sqDists
  }

  /**
   * Calculates the distance between each data point and the all of the cluster centers,
   * and assigns each data point to a cluster.
   *
   * @param dataset
   * @return
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val predictions = new ListBuffer[Int]()

    def minIndexSqDist(point: VectorWithNorm, centers: List[VectorWithNorm]) = computeSquaredDistances(point, centers)
      .zipWithIndex.minBy(_._1)._2

    lazy val points = dataset.select($(featuresCol)).collectAsList()

    points.forEach { point =>
      def pointVectorwoNorm = point.get(0).asInstanceOf[MLVector]
      def pointVector = new VectorWithNorm(pointVectorwoNorm)
      predictions.append(minIndexSqDist(pointVector,clusters))
    }

    val spark = dataset.sparkSession
    /** We get the spark session from the dataset in order to turn the prediction List
    into a dataframe */

    import spark.implicits._ // Used for implicit dataframe conversion
    import org.apache.spark.sql.functions.monotonically_increasing_id

    val predictionsDf = spark
      .sparkContext
      .parallelize(predictions.toList.map(_ + 1))
      .toDF($(predictionCol))
      .withColumn("clusterId", monotonically_increasing_id)

    /**
     * Joins the prediction column with the input dataset.
     */
    dataset
      .withColumn("clusterId", monotonically_increasing_id())
      .join(predictionsDf, "clusterId") //inner join
      .drop("clusterId")
      .toDF

  }

  /**
   * Transforms the schema.
   * @param schema
   */
  override def transformSchema(schema: StructType): StructType = {
    StructType(Seq(StructField($(predictionCol), IntegerType, true)).++(schema))
  }

  /**
   * Gets the Cluster Centers
   */
  def getClusterCenters: Seq[MLVector] = clusters.map(_.vector)

}

object onlineKmeansModel extends DefaultParamsReadable[onlineKmeansModel] {
  override def load(path: String): onlineKmeansModel = super.load(path)
}
