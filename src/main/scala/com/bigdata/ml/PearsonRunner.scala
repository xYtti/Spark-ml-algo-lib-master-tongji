package com.bigdata.ml

import java.io.{File, FileWriter}
import java.util
import scala.beans.BeanProperty
import com.bigdata.utils.Utils
import org.apache.spark.ml.stat
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, Vector, Vectors}
import org.apache.spark.ml.linalg.{Vectors => OldVectors}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.SparkConf
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.hadoop.fs.{FileSystem, Path}
import org.yaml.snakeyaml.{DumperOptions, TypeDescription, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.nodes.Tag
import org.yaml.snakeyaml.representer.Representer

class PearsonConfig extends Serializable {
  @BeanProperty var pearson: util.HashMap[String, util.HashMap[String, util.HashMap[String, util.HashMap[String, Object]]]] = _
}

class PearsonParams extends Serializable {
  @BeanProperty var pt: Int = _

  @BeanProperty var trainingDataPath: String = _
  @BeanProperty var outputDataPath: String = _
  @BeanProperty var apiName: String = _
  @BeanProperty var datasetName: String = _
  @BeanProperty var costTime: Double = _
  @BeanProperty var cpuName: String = _
  @BeanProperty var isRaw: String = _
  @BeanProperty var algorithmName: String = _
  @BeanProperty var testcaseType: String = _
  @BeanProperty var saveDataPath: String = _
  @BeanProperty var verifiedDataPath: String = _
}

object PearsonRunner {

  def main(args: Array[String]): Unit = {
    try {
      val modelConfSplit = args(0).split("_")
      val (dataStructure, datasetName) = (modelConfSplit(0), modelConfSplit(1))
      val dataPath = args(1)
      val dataPathSplit = dataPath.split(",")
      val (trainingDataPath, outputDataPath) = (dataPathSplit(0), dataPathSplit(1))
      val cpuName = args(2)
      val isRaw = args(3)
      val sparkConfSplit = args(4).split("_")
      val (master, deployMode, numExec, execCores, execMem) =
        (sparkConfSplit(0), sparkConfSplit(1), sparkConfSplit(2), sparkConfSplit(3), sparkConfSplit(4))

      val stream = (cpuName, isRaw) match {
        case ("aarch64", "no") =>
          Utils.getStream("conf/ml/pearson/pearson.yml")
        case ("x86_64", "no") =>
          Utils.getStream("conf/ml/pearson/pearson.yml")
        case ("x86_64", "yes") =>
          Utils.getStream("conf/ml/pearson/pearson_raw.yml")
      }
      val representer = new Representer
      representer.addClassTag(classOf[PearsonParams], Tag.MAP)
      val options = new DumperOptions
      options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
      val yaml = new Yaml(new Constructor(classOf[PearsonConfig]), representer, options)
      val description = new TypeDescription(classOf[PearsonParams])
      yaml.addTypeDescription(description)
      val configs: PearsonConfig = yaml.load(stream).asInstanceOf[PearsonConfig]
      val paramsMap: util.HashMap[String, Object] = configs.pearson.get(cpuName).get(dataStructure).get(datasetName)
      val params = new PearsonParams()
      params.setPt(paramsMap.getOrDefault("pt", "1000").asInstanceOf[Int])
      params.setTrainingDataPath(trainingDataPath)
      params.setOutputDataPath(outputDataPath)
      params.setApiName(dataStructure)
      params.setDatasetName(datasetName)
      params.setCpuName(cpuName)
      params.setIsRaw(isRaw)
      params.setAlgorithmName("Pearson")
      params.setTestcaseType(s"Pearson_${dataStructure}_${datasetName}")
      params.setSaveDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/${dataStructure}_${datasetName}")

      var appName = s"Pearson_${dataStructure}_${datasetName}"
      if (isRaw.equals("yes")){
        appName = s"Pearson_RAW_${dataStructure}_${datasetName}"
        params.setVerifiedDataPath(params.saveDataPath)
        params.setSaveDataPath(s"${params.saveDataPath}_raw")
      }
      params.setTestcaseType(appName)
      val conf = isRaw match {
        case "yes" =>
          new SparkConf().setAppName(s"Pearson_raw_${dataStructure}_${datasetName}").setMaster(master)
        case "no" =>
          new SparkConf().setAppName(s"Pearson_${dataStructure}_${datasetName}").setMaster(master)
      }
      val commonParas = Array (
        ("spark.submit.deployMode", deployMode),
        ("spark.executor.instances", numExec),
        ("spark.executor.cores", execCores),
        ("spark.executor.memory", execMem)
      )
      conf.setAll(commonParas)
      val spark = SparkSession.builder.config(conf).getOrCreate()
      println(s"Initialized spark session.")

      val costTime = dataStructure match {
        case "dataframe" =>
          new PearsonKernel().runDataframeJob(spark, params)
        case "rdd" =>
          new PearsonKernel().runRddJob(spark, params)
      }
      params.setCostTime(costTime)

      Utils.saveYml[PearsonParams](params, yaml)
      println(s"Exec Successful: costTime: ${costTime}s")
    } catch {
      case e: Throwable =>
        println(s"Exec Failure: ${e.getMessage}")
    }
  }
}

class PearsonKernel {

  def runDataframeJob(spark: SparkSession, params: PearsonParams): Double = {
    val pt = params.pt
    val sc = spark.sparkContext
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val startTime = System.currentTimeMillis()
    val data = spark.createDataFrame(
      sc
        .textFile(params.trainingDataPath)
        .map(x=>Row(Vectors.dense(x.split(",").map(_.toDouble))))
        .repartition(pt),
      StructType(List(StructField("matrix", VectorType)))
    ).persist(StorageLevel.MEMORY_ONLY)

    val mat = stat.Correlation.corr(data, "matrix")
    val costTime = (System.currentTimeMillis() - startTime) / 1000.0
    val pearsonMat = mat.collect()(0).getAs[DenseMatrix](0)
    //save result
    Utils.saveMatrix(pearsonMat, params.saveDataPath, sc)
    if (params.isRaw == "yes") {
      val verifiedFile = new Path(params.verifiedDataPath)
      if (fs.exists(verifiedFile)) {
        val ori = Utils.readMatrix(params.saveDataPath, sc)
        val opt = Utils.readMatrix(params.verifiedDataPath, sc)
        if (Utils.isEqualMatrix(ori, opt))
          println(s"${params.algorithmName} correct")
        else
          println(s"${params.algorithmName} incorrect")
      }
    }

    costTime
  }

  def runRddJob(spark: SparkSession, params: PearsonParams): Double = {

    val pt = params.pt
    val sc = spark.sparkContext
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val startTime = System.currentTimeMillis()
    val data = spark.createDataFrame(
      sc.textFile(params.trainingDataPath)
        .map(x=>Row(Vectors.dense(x.split(",").map(_.toDouble))))
        .repartition(pt),
      StructType(List(StructField("matrix", VectorType)))
    ).persist(StorageLevel.MEMORY_ONLY)

    val rdd = data.select("matrix").rdd.map{
      case Row(v: Vector) => OldVectors.fromML(v)
    }

    import org.apache.spark.mllib.stat.{Statistics => OldStatistics}
    val oldM = OldStatistics.corr(rdd, "pearson")
    val pearsonMat = oldM.asInstanceOf[DenseMatrix]
    val costTime = (System.currentTimeMillis() - startTime) / 1000.0
    //save result
    Utils.saveMatrix(pearsonMat, params.saveDataPath, sc)
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
