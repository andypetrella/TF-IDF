package org.apache.spark.mllib.devoxx



import breeze.linalg.{DenseVector => BDV}

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.rdd.RDDFunctions._
import org.apache.spark.rdd.RDD

class IDF(val minDocFreq: Int) {

  def this() = this(0)

  // TODO: Allow different IDF formulations.

  /**
   * Computes the inverse document frequency.
   * @param dataset an RDD of term frequency vectors
   */
  def fit(dataset: RDD[Vector]): IDFModel = {
    val idf = dataset.treeAggregate(new IDF.DocumentFrequencyAggregator(
          minDocFreq = minDocFreq))(
      seqOp = (df, v) => df.add(v),
      combOp = (df1, df2) => df1.merge(df2)
    ).idf()
    new IDFModel(idf)
  }

}

private object IDF {

  /** Document frequency aggregator. */
  class DocumentFrequencyAggregator(val minDocFreq: Int) extends Serializable {

    /** number of documents */
    private var m = 0L
    /** document frequency vector */
    private var df: BDV[Long] = _


    def this() = this(0)

    /** Adds a new document. */
    def add(doc: Vector): this.type = {
      if (isEmpty) {
        df = BDV.zeros(doc.size)
      }
      doc match {
        case sv: SparseVector =>
          val nnz = sv.indices.size
          var k = 0
          while (k < nnz) {
            if (sv.values(k) > 0) {
              df(sv.indices(k)) += 1L
            }
            k += 1
          }
        case dv: DenseVector =>
          val n = dv.size
          var j = 0
          while (j < n) {
            if (dv.values(j) > 0.0) {
              df(j) += 1L
            }
            j += 1
          }
        case other =>
          throw new UnsupportedOperationException(
            s"Only sparse and dense vectors are supported but got ${other.getClass}.")
      }
      m += 1L
      this
    }

    /** Merges another. */
    def merge(other: DocumentFrequencyAggregator): this.type = {
      if (!other.isEmpty) {
        m += other.m
        if (df == null) {
          df = other.df.copy
        } else {
          df += other.df
        }
      }
      this
    }

    private def isEmpty: Boolean = m == 0L

    /** Returns the current IDF vector. */
    def idf(): Vector = {
      if (isEmpty) {
        throw new IllegalStateException("Haven't seen any document yet.")
      }
      val n = df.length
      val inv = new Array[Double](n)
      var j = 0
      while (j < n) {
        /*
         * If the term is not present in the minimum
         * number of documents, set IDF to 0. This
         * will cause multiplication in IDFModel to
         * set TF-IDF to 0.
         *
         * Since arrays are initialized to 0 by default,
         * we just omit changing those entries.
         */
        if(df(j) >= minDocFreq) {
          inv(j) = math.log((m + 1.0) / (df(j) + 1.0))
        }
        j += 1
      }
      Vectors.dense(inv)
    }
  }
}


class IDFModel private[mllib] (val idf: Vector) extends Serializable {

  /**
   * Transforms term frequency (TF) vectors to TF-IDF vectors.
   *
   * If `minDocFreq` was set for the IDF calculation,
   * the terms which occur in fewer than `minDocFreq`
   * documents will have an entry of 0.
   *
   * @param dataset an RDD of term frequency vectors
   * @return an RDD of TF-IDF vectors
   */
  def transform[A:reflect.ClassTag](dataset: RDD[(A, Vector)]): RDD[(A, Vector)] = {
    val bcIdf = dataset.context.broadcast(idf)
    dataset.mapPartitions { iter =>
      val thisIdf = bcIdf.value
      iter.map { case (a, v) =>
        val n = v.size
        v match {
          case sv: SparseVector =>
            val nnz = sv.indices.size
            val newValues = new Array[Double](nnz)
            var k = 0
            while (k < nnz) {
              newValues(k) = sv.values(k) * thisIdf(sv.indices(k))
              k += 1
            }
            (a, Vectors.sparse(n, sv.indices, newValues))
          case dv: DenseVector =>
            val newValues = new Array[Double](n)
            var j = 0
            while (j < n) {
              newValues(j) = dv.values(j) * thisIdf(j)
              j += 1
            }
            (a, Vectors.dense(newValues))
          case other =>
            throw new UnsupportedOperationException(
              s"Only sparse and dense vectors are supported but got ${other.getClass}.")
        }
      }
    }
  }

}