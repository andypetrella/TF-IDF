package org.apache.spark.mllib.devoxx

import org.apache.spark.SparkContext._

import scala.collection.JavaConverters._
import scala.collection.mutable

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils

/**
 * :: Experimental ::
 * Maps a sequence of terms to their term frequencies using the hashing trick.
 *
 * @param numFeatures number of features (default: 2^20^)
 */
class HashingTF(val numFeatures: Int) extends Serializable {
  def nonNegativeMod(x: Int, mod: Int): Int = {
    val rawMod = x % mod
    rawMod + (if (rawMod < 0) mod else 0)
  }

  def this() = this(1 << 20)

  /**
   * Returns the index of the input term.
   */
  def indexOf(term: Any): Int = nonNegativeMod(term.##, numFeatures)

  /**
   * Transforms the input document into a sparse term frequency vector.
   */
  def transform(document: Iterable[_]): Vector = {
    val termFrequencies = mutable.HashMap.empty[Int, Double]
    document.foreach { term =>
      val i = indexOf(term)
      termFrequencies.put(i, termFrequencies.getOrElse(i, 0.0) + 1.0)
    }
    Vectors.sparse(numFeatures, termFrequencies.toSeq)
  }

  /**
   * Transforms the input document to term frequency vectors.
   */
  def transform[A:reflect.ClassTag, D <: Iterable[_]:reflect.ClassTag](dataset: RDD[(A, D)]): RDD[(A,Vector)] = {
    dataset.mapValues(this.transform)
  }
}