package com.bigdata.ml

import java.io.{File, FileWriter}
import java.util
import com.bigdata.utils.Utils
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.SPCA
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.storage.StorageLevel
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.yaml.snakeyaml.{DumperOptions, TypeDescription, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.nodes.Tag
import org.yaml.snakeyaml.representer.Representer

import scala.beans.BeanProperty

class SPCAConfig extends Serializable {

  @BeanProperty var spca: util.HashMap[String, util.HashMap[String, Object]] = _
}

class SPCAParams extends Serializable {

  @BeanProperty var pt: Int = _
  @BeanProperty var k: Int = _
  @BeanProperty var sep: String = _
  @BeanProperty var numCols: Int = _
  @BeanProperty var pcPath: String = _
  @BeanProperty var sigmaPath: String = _
  @BeanProperty var saveRes: Boolean = _

  @BeanProperty var inputDataPath: String = _
  @BeanProperty var apiName: String = _
  @BeanProperty var datasetName: String = _
  @BeanProperty var isRaw: String = _
  @BeanProperty var cpuName: String = _
  @BeanProperty var costTime: Double = _
  @BeanProperty var loadDataTime: Double = _
  @BeanProperty var algorithmName: String = _
  @BeanProperty var testcaseType: String = _
  @BeanProperty var saveDataPath: String = _
  @BeanProperty var verifiedDataPath: String = _

}

object SPCARunner {
  def main(args: Array[String]): Unit = {

    try {
      val modelConfSplit = args(0).split("-")
      val (datasetName, apiName) =
        (modelConfSplit(0), modelConfSplit(1))
      val inputDataPath = args(1)
      val cpuName = args(2)
      val isRaw = args(3)
      val sparkConfSplit = args(4).split("_")
      val (master, deployMode, numExec, execCores, execMem) =
        (sparkConfSplit(0), sparkConfSplit(1), sparkConfSplit(2), sparkConfSplit(3), sparkConfSplit(4))

      val stream = isRaw match {
        case "no" =>
          Utils.getStream("conf/ml/spca/spca.yml")
        case "yes" =>
          Utils.getStream("conf/ml/spca/spca_raw.yml")
      }
      val representer = new Representer
      representer.addClassTag(classOf[SPCAParams], Tag.MAP)
      val options = new DumperOptions
      options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
      val yaml = new Yaml(new Constructor(classOf[SPCAConfig]), representer, options)
      val description = new TypeDescription(classOf[SPCAParams])
      yaml.addTypeDescription(description)
      val configs: SPCAConfig = yaml.load(stream).asInstanceOf[SPCAConfig]
      val params = new SPCAParams()
      val paramsMap: util.HashMap[String, Object] = configs.spca.get(datasetName)
      params.setPt(paramsMap.getOrDefault("pt", "250").asInstanceOf[Int])
      params.setK(paramsMap.getOrDefault("k", "10").asInstanceOf[Int])
      params.setSep(paramsMap.getOrDefault("sep", " ").asInstanceOf[String])
      params.setNumCols(paramsMap.getOrDefault("numCols", "0").asInstanceOf[Int])
      params.setPcPath(paramsMap.getOrDefault("pcPath", null.asInstanceOf[String]).asInstanceOf[String])
      params.setSigmaPath(paramsMap.getOrDefault("sigmaPath", null.asInstanceOf[String]).asInstanceOf[String])
      params.setSaveRes(paramsMap.getOrDefault("saveRes", "false").asInstanceOf[Boolean])
      params.setInputDataPath(inputDataPath)
      params.setDatasetName(datasetName)
      params.setApiName(apiName)
      params.setCpuName(cpuName)
      params.setIsRaw(isRaw)
      params.setAlgorithmName("SPCA")
      params.setSaveDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/${datasetName}_${apiName}")
      var appName = s"SPCA_${datasetName}_${apiName}"
      if (isRaw == "yes") {
        appName = s"SPCA_${datasetName}_${apiName}_raw"
        params.setVerifiedDataPath(params.saveDataPath)
        params.setSaveDataPath(s"${params.saveDataPath}_raw")
      }
      params.setTestcaseType(appName)
      val conf = new SparkConf()
        .setAppName(appName).setMaster(master)
      val commonParas = Array (
        ("spark.submit.deployMode", deployMode),
        ("spark.executor.instances", numExec),
        ("spark.executor.cores", execCores),
        ("spark.executor.memory", execMem)
      )
      conf.setAll(commonParas)
      val spark = SparkSession.builder.config(conf).getOrCreate()

      val costTime = new SPCAKernel().runJob(spark, params)
      params.setCostTime(costTime)

      Utils.saveYml[SPCAParams](params, yaml)
      println(s"Exec Successful: costTime: ${costTime}s")
    } catch {
      case e: Throwable =>
        println(s"Exec Failure: ${e.getMessage}")
        throw e
    }
  }
}

class SPCAKernel {

  def runJob(spark: SparkSession, params: SPCAParams): Double = {

    import spark.implicits._
    val sc = spark.sparkContext
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val startTime = System.currentTimeMillis()
    val trainingData = if (params.isRaw == "yes"){
      val numColsBC = sc.broadcast(params.numCols)
      val sepBC = sc.broadcast(params.sep)
      val data = spark.createDataFrame(sc.textFile(params.inputDataPath, params.pt)
        .map(line => {
          val entry = line.split(sepBC.value)
          (entry(0).toInt, (entry(1).toInt, entry(2).toDouble))
        }).groupByKey()
        .map { case (_, vectorEntries) => Vectors.sparse(numColsBC.value, vectorEntries.toSeq) }
        .repartition(params.pt)
        .map(Tuple1.apply))
        .toDF("matrix").persist(StorageLevel.MEMORY_ONLY)
      data
    } else {
      val data = spark.createDataFrame(sc.textFile(params.inputDataPath, params.pt)
        .map(line => {
          val entry = line.split(params.sep)
          (entry(0).toInt, (entry(1).toInt, entry(2).toDouble))
        }).groupByKey()
        .map{case (_, vectorEntries) => Vectors.sparse(params.numCols, vectorEntries.toSeq)}
        .repartition(params.pt)
        .map(Tuple1.apply))
        .toDF("matrix")
        .persist(StorageLevel.MEMORY_ONLY)
      data
    }


    val loadDataTime = (System.currentTimeMillis() - startTime) / 1000.0
    params.setLoadDataTime(loadDataTime)

    val spca = if (params.isRaw == "no"){
      new SPCA()
        .setK(params.k)
        .setInputCol("matrix")
    } else {
      new PCA()
        .setK(params.k)
        .setInputCol("matrix")
    }

    val paramMap = ParamMap(spca.k -> params.k)
      .put(spca.inputCol, "matrix")
    val paramMaps: Array[ParamMap] = new Array[ParamMap](2)
    for (i <- 0 to paramMaps.size - 1) {
      paramMaps(i) = ParamMap(spca.k -> params.k)
        .put(spca.inputCol, "matrix")
    }
    val kPair = ParamPair(spca.k, params.k)
    val inputColPair = ParamPair(spca.inputCol, "matrix")
    val model = params.apiName match {
      case "fit" => spca.fit(trainingData)
      case "fit1" => spca.fit(trainingData, paramMaps)
      case "fit2" =>
        val models = spca.fit(trainingData, paramMaps)
        models(0)
      case "fit3" => spca.fit(trainingData, kPair, inputColPair)
    }
    val costTime = (System.currentTimeMillis() - startTime) / 1000.0
    params.setLoadDataTime(costTime)

    val spcaMat = model.pc
    Utils.saveMatrix(spcaMat, params.saveDataPath, sc)
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
