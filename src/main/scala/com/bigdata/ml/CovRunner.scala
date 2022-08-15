package com.bigdata.ml

import com.bigdata.utils.Utils
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.hadoop.fs.{FileSystem, Path}
import org.yaml.snakeyaml.{DumperOptions, TypeDescription, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.nodes.Tag
import org.yaml.snakeyaml.representer.Representer

import java.io.{File, FileWriter}
import java.util
import scala.beans.BeanProperty

class CovConfig extends Serializable {

  @BeanProperty var cov: util.HashMap[String, util.HashMap[String, Object]] = _
}

class CovParams extends Serializable {
  @BeanProperty var numPartitions: Int = _


  @BeanProperty var dataPath: String = _
  @BeanProperty var datasetName: String = _
  @BeanProperty var datasetCpuName: String = _
  @BeanProperty var isRaw: String = "no"
  @BeanProperty var costTime: Double = _
  @BeanProperty var evaluation: String = "mismatch"
  @BeanProperty var algorithmName: String = _
  @BeanProperty var testcaseType: String = _
  @BeanProperty var saveDataPath: String = _
  @BeanProperty var verifiedDataPath: String = _

}

object CovRunner {

  def main(args: Array[String]): Unit = {

    try {
      val modelConfSplit = args(0).split("-")
      val (datasetName, platformName) = (modelConfSplit(0), modelConfSplit(1))
      val dataPath = args(1)
      val datasetCpuName = s"${datasetName}_${platformName}"

      val stream = Utils.getStream("conf/ml/cov/cov.yml")
      val representer = new Representer
      representer.addClassTag(classOf[CovParams], Tag.MAP)
      val options = new DumperOptions
      options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
      val yaml = new Yaml(new Constructor(classOf[CovConfig]), representer, options)
      val description = new TypeDescription(classOf[CovParams])
      yaml.addTypeDescription(description)
      val configs: CovConfig = yaml.load(stream).asInstanceOf[CovConfig]
      val paramsMap: util.HashMap[String, Object] = configs.cov.get(datasetCpuName)
      val params = new CovParams()
      params.setNumPartitions(paramsMap.get("numPartitions").asInstanceOf[Int])
      params.setDataPath(dataPath)
      params.setDatasetName(datasetName)
      params.setDatasetCpuName(datasetCpuName)
      params.setAlgorithmName("Cov")
      params.setTestcaseType(s"Cov_${datasetName}_${platformName}")
      params.setSaveDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/${datasetName}")

      val conf = new SparkConf().setAppName(s"Cov_${datasetName}_${platformName}")
      if (platformName == "raw") {
        params.setIsRaw("yes")
        params.setVerifiedDataPath(params.saveDataPath)
        params.setSaveDataPath(s"${params.saveDataPath}_raw")
      }
      val spark = SparkSession.builder.config(conf).getOrCreate()

      val costTime = new CovKernel().runJob(spark, params)
      params.setCostTime(costTime)

      Utils.saveYml[CovParams](params, yaml)
      println(s"Exec Successful: costTime: ${params.getCostTime}s")
    } catch {
      case e: Throwable =>
        println(s"Exec Failure: ${e.getMessage}")
        throw e
    }
  }
}

class CovKernel {

  def runJob(spark: SparkSession, params: CovParams): Double = {
    val sc = spark.sparkContext
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val startTime = System.currentTimeMillis()
    val data = sc.textFile(params.dataPath)
      .map(x => Vectors.dense(x.split(",").map(_.toDouble)))
      .repartition(params.numPartitions)
      .persist(StorageLevel.MEMORY_ONLY)
    val matrix = new RowMatrix(data)
    val covMat = matrix.computeCovariance().asInstanceOf[DenseMatrix]
    val costTime = (System.currentTimeMillis() - startTime) / 1000.0

    Utils.saveMatrix(covMat, params.saveDataPath, sc)
    if (params.isRaw == "yes") {
      val verifiedFile = new Path(params.verifiedDataPath)
      if (fs.exists(verifiedFile)) {
        if (Utils.verifyMatrix(params.saveDataPath, params.verifiedDataPath, sc))
          println(s"${params.algorithmName} correct")
        else
          println(s"${params.algorithmName} incorrect")
      }
    }

    costTime
  }
}