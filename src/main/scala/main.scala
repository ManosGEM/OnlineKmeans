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

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types._
import org.apache.spark.sql.{SQLContext, SparkSession}

/**
 * A main object with an example run, using an uber dataset (see resources).
 */
object main {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("OnlineKmeans").master("local")
      .getOrCreate()

    /**
     * The schema
     */
    val schema = StructType(
        StructField("time", TimestampType, nullable = true) ::
        StructField("lat", DoubleType, nullable = true) ::
        StructField("lon", DoubleType, nullable = true) ::
        StructField("base", StringType, nullable = true) ::
        Nil
    )

    /**
     * Read to DataFrame
     */
    spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
    val uberDf = spark.read.format("csv")
      .option("header", value = true)
      .option("delimiter", ",")
      .option("mode", "DROPMALFORMED")
      .option("timestampFormat", "yyyy/MM/dd HH:mm:ss")
      .schema(schema)
      .load(getClass().getResource("/uber.csv").getPath)
      .cache()
    uberDf.printSchema()
    uberDf.show(10)

    /**
     * Transform userDf with VectorAssembler to add feature column
     */
    val cols = Array("lat", "lon")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(uberDf)
    featureDf.printSchema()
    featureDf.show(10)

    val onlineKmeans = new onlineKmeans()
      .setK(30)
      .setFeaturesCol("features")
      .setPredictionCol("predictions")

    val onlinekmeansMod = onlineKmeans.fit(featureDf)
    onlinekmeansMod.transform(featureDf).show(50)
  }
}