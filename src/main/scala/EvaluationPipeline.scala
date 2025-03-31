import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector => MLVector, Vectors}
import org.apache.spark.sql.types._
import scala.util.Random
import java.io.PrintWriter
import OnlineKmeans._

object EvaluationPipeline {
  def main(args: Array[String]): Unit = {
    // Initialize Spark session
    val spark = SparkSession.builder()
      .appName("OnlineKMeans Evaluation")
      .master("local[*]")
      .getOrCreate()

    // Ensure compatibility with older timestamp formats
    spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

    // Dataset configuration (only Uber for now)
    val datasets = Seq(
      ("Uber", "/uber.csv", Array("Lat", "Lon"), new StructType()
        .add("Date/Time", TimestampType)
        .add("Lat", DoubleType)
        .add("Lon", DoubleType)
        .add("Base", StringType))
    )

    // Target k values and number of experiment runs per setting
    val kTargets = Seq(20, 25, 30, 35, 40, 45, 50, 55, 60)
    val numRuns = 5

    // Helper function to compute median of a sequence
    def median(values: Seq[Double]): Double = {
      val sorted = values.sorted
      val size = sorted.size
      if (size % 2 == 0) (sorted(size / 2 - 1) + sorted(size / 2)) / 2.0
      else sorted(size / 2)
    }

    // Prepare CSV output file
    val output = new PrintWriter("evaluation_results.csv")
    output.println("Dataset,kTarget,Model,kAvg,WCSS_avg,WCSS_median,kPerRun,WCSSPerRun")

    // Iterate over each dataset
    for ((datasetName, path, inputCols, schema) <- datasets) {
      // Read dataset and apply schema
      val df = spark.read
        .option("header", "true")
        .option("timestampFormat", "yyyy/MM/dd HH:mm:ss")
        .schema(schema)
        .csv(getClass.getResource(path).getPath)
        .na.drop()

      // Assemble feature column
      val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features")
      val featureDf = assembler.transform(df).cache()

      // Loop through each target k value
      for (kTarget <- kTargets) {
        // Containers to hold per-run metrics
        val costsOnline = scala.collection.mutable.ArrayBuffer.empty[Double]
        val costsVar2 = scala.collection.mutable.ArrayBuffer.empty[Double]
        val costsSpark = scala.collection.mutable.ArrayBuffer.empty[Double]
        val costsRandom = scala.collection.mutable.ArrayBuffer.empty[Double]
        val kActuals = scala.collection.mutable.ArrayBuffer.empty[Int]
        val kVars = scala.collection.mutable.ArrayBuffer.empty[Int]

        // Run experiments multiple times per kTarget
        for (_ <- 1 to numRuns) {
          // OnlineKMeans (Algorithm 3 from paper)
          val onlineKMeans = new onlineKmeans().setK(kTarget).setFeaturesCol("features")
          val modelOnline = onlineKMeans.fit(featureDf)
          val kActual = modelOnline.getClusterCenters.size
          val costOnline = SparkMLUtils.computeClusteringCost(featureDf, modelOnline.getClusterCenters)
          costsOnline += costOnline
          kActuals += kActual

          // OnlineKMeans Variation 2 (no k-scaling)
          val onlineVar = new onlineKmeans().setK(kTarget).setFeaturesCol("features").setVariation(2)
          val modelVar = onlineVar.fit(featureDf)
          val kVar = modelVar.getClusterCenters.size
          val costVar = SparkMLUtils.computeClusteringCost(featureDf, modelVar.getClusterCenters)
          costsVar2 += costVar
          kVars += kVar

          // Spark MLlib KMeans++ baseline (using kActual)
          val sparkKMeans = new KMeans().setK(kActual).setSeed(System.nanoTime()).setFeaturesCol("features")
          val sparkModel = sparkKMeans.fit(featureDf)
          val costSpark = SparkMLUtils.computeClusteringCost(featureDf, sparkModel.clusterCenters)
          costsSpark += costSpark

          // Random baseline (kActual random centers)
          val allVectors = featureDf.select("features").rdd.map(_.getAs[MLVector](0)).collect().toList
          val randomCenters = Random.shuffle(allVectors).take(kActual)
          val costRandom = SparkMLUtils.computeClusteringCost(featureDf, randomCenters)
          costsRandom += costRandom
        }

        // Write results to CSV per model
        def logRow(model: String, kAvg: Double, costs: Seq[Double], kRuns: Seq[Int]): Unit = {
          val avgCost = costs.sum / numRuns
          val medCost = median(costs)
          val kPerRun = if (kRuns.nonEmpty) "\"" + kRuns.mkString("[", ",", "]") + "\"" else "-"
          val costPerRun = "\"" + costs.mkString("[", ",", "]") + "\""
          output.println(s"$datasetName,$kTarget,$model,$kAvg,$avgCost,$medCost,$kPerRun,$costPerRun")
        }

        logRow("OnlineKMeans", kActuals.sum.toDouble / numRuns, costsOnline.toSeq, kActuals.toSeq)
        logRow("OnlineKMeans_Var2", kVars.sum.toDouble / numRuns, costsVar2.toSeq, kVars.toSeq)
        logRow("KMeans++", kActuals.sum.toDouble / numRuns, costsSpark.toSeq, Seq())
        logRow("Random", kActuals.sum.toDouble / numRuns, costsRandom.toSeq, Seq())

        // Console output for tracking
        println(s"Dataset: $datasetName | kTarget: $kTarget | avg kActual: ${kActuals.sum.toDouble / numRuns} | avg kVar: ${kVars.sum.toDouble / numRuns}")
        println(s"kActuals per run: ${kActuals.mkString(", ")}")
        println(s"kVars per run: ${kVars.mkString(", ")}")
        println(f"[OnlineKMeans]        WCSS (avg): ${costsOnline.sum / numRuns}%2.5f   WCSS (med): ${median(costsOnline.toSeq)}%2.5f")
        println(f"[OnlineKMeans Var 2]  WCSS (avg): ${costsVar2.sum / numRuns}%2.5f   WCSS (med): ${median(costsVar2.toSeq)}%2.5f")
        println(f"[KMeans++]           WCSS (avg): ${costsSpark.sum / numRuns}%2.5f   WCSS (med): ${median(costsSpark.toSeq)}%2.5f")
        println(f"[Random Baseline]    WCSS (avg): ${costsRandom.sum / numRuns}%2.5f   WCSS (med): ${median(costsRandom.toSeq)}%2.5f")
        println("=" * 60)
      }
    }

    // Close output file and Spark session
    output.close()
    spark.stop()
  }
}
