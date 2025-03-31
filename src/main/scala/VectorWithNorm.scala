package OnlineKmeans

import org.apache.spark.ml.linalg.{Vectors,Vector}

class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

  def this(array: Array[Double]) = this(Vectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm =
    new VectorWithNorm(Vectors.dense(vector.toArray), norm)
}