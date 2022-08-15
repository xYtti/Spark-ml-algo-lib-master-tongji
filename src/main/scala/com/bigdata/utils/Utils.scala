package com.bigdata.utils

import java.io.{File, FileInputStream, FileWriter, InputStreamReader, PrintWriter}
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util
import java.util.TimeZone
import java.util.Date
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.param.Params
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.functions.udf
import org.apache.spark.storage.StorageLevel
import org.yaml.snakeyaml.{DumperOptions, TypeDescription, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.nodes.Tag
import org.yaml.snakeyaml.representer.Representer

import scala.io.Source


object Utils {

  val dirPath = "hdfs:///tmp/ml/result"


  /**
   *
   * @param filename The resource name
   * @return
   */

  def getStream(filename: String): InputStreamReader = {

    val file = new File(filename)
    if (!file.exists() || file.isDirectory) {
      throw new Exception(s"Fail to find prorerty file[${file}]")
    }
    val inputStreamReader = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8)

    inputStreamReader

  }

  /**
   *
   * @param dateFmt         date format
   * @param utcMilliseconds "yyyy-MM-dd HH:mm:ss"
   * @return String date
   */
  def getDateStrFromUTC(dateFmt: String, utcMilliseconds: Long): String = {
    val sf = new SimpleDateFormat(dateFmt)
    sf.setTimeZone(TimeZone.getTimeZone("Asia/Shanghai"))
    sf.format(new Date(utcMilliseconds))
  }

/**
  /**
   * Convert DenseMatrix to 2-dimension array, stored in row major
   *
   * @param matrix Input matrix
   * @return 2-dimension array, stored in row major
   */

  def toRowMajorArray(matrix: DenseMatrix): Array[Array[Double]] = {
    val nRow = matrix.numRows
    val nCol = matrix.numCols
    val arr = new Array[Array[Double]](nRow).map(_ => new Array[Double](nCol))
    if (matrix.isTransposed) {
      var srcOffset = 0
      for {i <- 0 until nRow} {
        System.arraycopy(matrix.values, srcOffset, arr(i), 0, nCol)
        srcOffset += nCol
      }
    } else {
      matrix.values.indices.foreach(idx => {
        val j = math.floor(idx / nRow).toInt
        val i = idx % nRow
        arr(i)(j) = matrix.values(idx)
      })
    }
    arr
  }

  def writeMatrix(mat: DenseMatrix, path: String): Unit = {
    val file = new File(path)
    if (file.exists()) {
      file.delete()
    }
    val writer = new PrintWriter(path)
    val arr = toRowMajorArray(mat)
    arr.foreach(vec => writer.write(vec.mkString(",") + "\n"))
    writer.close()
  }

  def writeVector(vector: DenseVector, path: String): Unit = {
    val writer = new PrintWriter(path)
    vector.values.foreach(d => writer.write(d + "\n"))
    writer.close()
  }

  /**
   * Read matrix from local file.
   *
   * @param path File path
   * @return matrix in 2D-array format

  def readMatrix(path: String): Array[Array[Double]] = {
    val file = Source.fromFile(path)
    val arr = file.getLines().map(line => line.split(",").map(_.toDouble)).toArray
    file.close()
    arr
  }
  */

  /**
   * Read vector from local file.
   *
   * @param path File path
   */
  def readVector(path: String): Array[Double] = {
    val file = Source.fromFile(path)
    val arr = file.getLines().map(_.toDouble).toArray
    file.close()
    arr
  }

   def verifyVector(saveDataPath: String, verifiedDataPath: String): Boolean = {
    val ori = readMatrix(saveDataPath)
    val opt = readMatrix(verifiedDataPath)
    if (isEqualVector(ori, opt))
      return true
    else
      return false
  }
*/

  /**
   * Compare two vector
   *
   * @param tol tolerance
   */

  def saveYml[T](params: T, yaml: Yaml): Unit = {
    val folder = new File("report")
    if (!folder.exists()) {
      val mkdir = folder.mkdirs()
      println(s"Create dir report ${mkdir}")
    }
    val writer = new FileWriter(s"report/${params.algorithmName}_${
      getDateStrFromUTC("yyyyMMdd_HHmmss",
        System.currentTimeMillis())
    }.yml")
    yaml.dump(params, writer)
  }


  def saveAndVerifyRes[T](params: T, res: Double, sc: SparkContext): Unit = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val saveFile = new Path(params.saveDataPath)
    if (fs.exists(saveFile)) {
      fs.delete(saveFile, true)
    }
    sc.parallelize(res).saveAsTextFile(params.saveDataPath)
    if (params.isRaw == "yes") {
      val verifiedFile = new Path(params.verifiedDataPath)
      if (fs.exists(verifiedFile)) {
        val pri: Double = sc.textFile(params.saveDataPath)
        val opt: Double = sc.textFile(params.verifiedDataPath)
        if (math.abs(pri - opt) / pri <= 0.005) {
          println(s"${params.algorithmName} correct")
        }
        else {
          println(s"${params.algorithmName} incorrect")
        }
      }
    }
  }


  def saveMatrix(mat: DenseMatrix, saveDataPath: String, sc: SparkContext): Unit = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val saveFile = new Path(saveDataPath)
    if (fs.exists(saveFile)) {
      fs.delete(saveFile, true)
    }
    val result = mat.toArray
    val res = new Array[String](mat.numRows)
    for(i <- 0 until mat.numRows) {
      val row = result.slice(i * mat.numRows, (i + 1) * mat.numCols)
      res(i) = row+";"
    }
    sc.parallelize(res).repartition(100).saveAsTextFile(saveDataPath)
  }

  def saveVector(vector: DenseVector, saveDataPath: String, sc: SparkContext): Unit = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val saveFile = new Path(saveDataPath)
    if (fs.exists(saveFile)) {
      fs.delete(saveFile, true)
    }
    val res = vector.values
    sc.parallelize(res).saveAsTextFile(saveDataPath)
  }

  def saveArray(res: Array[Double], saveDataPath: String, sc: SparkContext): Unit = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val saveFile = new Path(saveDataPath)
    if (fs.exists(saveFile)) {
      fs.delete(saveFile, true)
    }
    sc.parallelize(res).saveAsTextFile(saveDataPath)
  }

  def readMatrix(saveDataPath: String, sc: SparkContext): Array[Array[Double]] = {
    val arr = sc.textFile(saveDataPath).map(line => line.split(";").map(_.toDouble)).toArray
    arr
  }

  def readVector(saveDataPath: String, sc: SparkContext): Array[Array[Double]] = {
    val arr = sc.textFile(saveDataPath).map(line => line.map(_.toDouble)).toArray
    arr
  }

  def isEqualMatrix(opensourceMatrix: Array[Array[Double]], boostkitMatrix: Array[Array[Double]]): Boolean = {
    if (opensourceMatrix.length != boostkitMatrix.length)
      return false
    for (i <- boostkitMatrix.indices) {
      if (opensourceMatrix(i).length != boostkitMatrix(i).length)
        return false
      for (j <- opensourceMatrix(i).indices) {
        if (math.abs(math.abs(opensourceMatrix(i)(j)) - math.abs(boostkitMatrix(i)(j))) > 1e-6)
          return false
      }
    }
    true
  }

  def isEqualVector(opensourceVector: Array[Double], boostkitVector: Array[Double]): Boolean = {
    if (opensourceVector.length != boostkitVector.length)
      return false
    for (i <- boostkitVector.indices) {
      if (math.abs(opensourceVector(i) - boostkitVector(i)) > 1e-6)
        return false
    }
    true
  }

  def verifyMatrix(saveDataPath: String, verifiedDataPath: String, sc: SparkContext): Boolean = {
    val pri = readMatrix(saveDataPath, sc)
    val opt = readMatrix(verifiedDataPath, sc)
    if (isEqualMatrix(pri, opt))
      return true
    else
      return false
  }

  def verifyVector(saveDataPath: String, verifiedDataPath: String, sc: SparkContext): Boolean = {
    val pri = readVector(saveDataPath, sc)
    val opt = readVector(verifiedDataPath, sc)
    if (isEqualVector(pri, opt))
      return true
    else
      return false
  }


}