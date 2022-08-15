package com.bigdata.ml

import java.io.{File, FileWriter}
import java.util.HashMap
import com.bigdata.utils.Utils
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.apache.spark.sql.functions.udf
import org.apache.hadoop.fs.{FileSystem, Path}
import org.yaml.snakeyaml.{DumperOptions, TypeDescription, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.nodes.Tag
import org.yaml.snakeyaml.representer.Representer

import java.util
import scala.beans.BeanProperty

class LogRConfig extends Serializable {

  @BeanProperty var logR: util.HashMap[String, util.HashMap[String, Object]] = _
}

class LogRParams extends Serializable {

  @BeanProperty var numPartitions: Int = _
  @BeanProperty var numLabels: Int = _
  @BeanProperty var numFeatures: Int = _
  @BeanProperty var regParam: Double = _
  @BeanProperty var elasticNetParam: Double = _
  @BeanProperty var maxIter: Int = _
  @BeanProperty var tolerance: Double = _
  @BeanProperty var isSetBound: Boolean = _
  @BeanProperty var coefficientLowerBound: Double = _
  @BeanProperty var coefficientUpperBound: Double = _
  
  @BeanProperty var trainingDataPath: String = _
  @BeanProperty var testDataPath: String = _
  @BeanProperty var apiName: String = _
  @BeanProperty var datasetName: String = _
  @BeanProperty var datasetCpuName: String = _
  @BeanProperty var isRaw: String = "no"
  @BeanProperty var evaluation: Double = _
  @BeanProperty var costTime: Double = _
  @BeanProperty var loadDataTime: Double = _
  @BeanProperty var algorithmName: String = _
  @BeanProperty var testcaseType: String = _
  @BeanProperty var saveDataPath: String = _
  @BeanProperty var verifiedDataPath: String = _
}

object LogRRunner {
  def main(args: Array[String]): Unit = {
    try {
      val modelConfSplit = args(0).split("-")
      val (datasetName, apiName, cpuName) =
        (modelConfSplit(0), modelConfSplit(1), modelConfSplit(2))
      val dataPath = args(1)
      val dataPathSplit = dataPath.split(",")
      val (trainingDataPath, testDataPath) = (dataPathSplit(0), dataPathSplit(1))
      val datasetCpuName = s"${datasetName}_${cpuName}"

      val stream = Utils.getStream("conf/ml/logR/logR.yml")
      val representer = new Representer
      representer.addClassTag(classOf[LogRParams], Tag.MAP)
      val options = new DumperOptions
      options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
      val yaml = new Yaml(new Constructor(classOf[LogRConfig]), representer, options)
      val description = new TypeDescription(classOf[LogRParams])
      yaml.addTypeDescription(description)
      val config: LogRConfig = yaml.load(stream).asInstanceOf[LogRConfig]
      val paramsMap: util.HashMap[String, Object] = config.logR.get(datasetCpuName)
      val params = new LogRParams()
      params.setNumPartitions(paramsMap.get("numPartitions").asInstanceOf[Int])
      params.setMaxIter(paramsMap.get("maxIter").asInstanceOf[Int])
      params.setNumLabels(paramsMap.get("numLabels").asInstanceOf[Int])
      params.setNumFeatures(paramsMap.get("numFeatures").asInstanceOf[Int])
      params.setRegParam(paramsMap.get("regParam").asInstanceOf[Double])
      params.setElasticNetParam(paramsMap.get("elasticNetParam").asInstanceOf[Double])
      params.setTolerance(paramsMap.get("tolerance").asInstanceOf[Double])
      params.setIsSetBound(paramsMap.get("isSetBound").asInstanceOf[Boolean])
      params.setCoefficientLowerBound(paramsMap.get("coefficientLowerBound").asInstanceOf[Double])
      params.setCoefficientUpperBound(paramsMap.get("coefficientUpperBound").asInstanceOf[Double])
      params.setApiName(apiName)
      params.setTrainingDataPath(trainingDataPath)
      params.setTestDataPath(testDataPath)
      params.setDatasetName(datasetName)
      params.setDatasetCpuName(datasetCpuName)
      params.setAlgorithmName("LogR")
      params.setSaveDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/${datasetName}")
      if (cpuName == "raw") {
        params.setIsRaw("yes")
      }

      var appName = s"LogR_${datasetName}_${apiName}"
      if (cpuName.equals("raw")){
        appName = s"LogR_RAW_${datasetName}_${apiName}"
        params.setVerifiedDataPath(params.saveDataPath)
        params.setSaveDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/${datasetName}_raw")
      }
      params.setTestcaseType(appName)
      val conf = new SparkConf()
        .setAppName(appName)
      val spark = SparkSession.builder.config(conf).getOrCreate()
      val (res, costTime) = new LogRKernel().runJob(spark, params)
      params.setEvaluation(res)
      params.setCostTime(costTime)

      Utils.saveYml[LogRParams](params, yaml)
      println(s"Exec Successful: costTime: ${costTime}s; evaluation: ${res}")
    } catch {
      case e: Throwable =>
        println(s"Exec Failure: ${e.getMessage}")
        throw e
    }
  }
}

class LogRKernel {

  def runJob(spark: SparkSession, params: LogRParams): (Double, Double) = {

    import spark.implicits._
    val sc = spark.sparkContext
    val startTime = System.currentTimeMillis()
    val trainData = spark
      .read
      .format("libsvm")
      .option("vectorType", "dense")
      .option("numFeatures", params.numFeatures)
      .load(params.trainingDataPath)
      .repartition(params.numPartitions)
      .cache()

    val trainingData = if (params.datasetName == "rcv") {
      val label_convert = udf((x: Double) => if (x < 0.0) 0.0 else 1.0)
      trainData.withColumn("label", label_convert($"label"))
    } else {
      trainData
    }
    println(s"trainingData: ${trainingData.count()}")
    val loadDataTime = (System.currentTimeMillis() - startTime) / 1000.0

    val logR = if (params.isSetBound){
      new LogisticRegression()
        .setRegParam(params.regParam)
        .setElasticNetParam(params.elasticNetParam)
        .setMaxIter(params.maxIter)
        .setTol(params.tolerance)
        .setLowerBoundsOnCoefficients(new DenseMatrix(params.numLabels, params.numFeatures, Array.fill(params.numLabels * params.numFeatures)(params.coefficientLowerBound), true))
        .setUpperBoundsOnCoefficients(new DenseMatrix(params.numLabels, params.numFeatures, Array.fill(params.numLabels * params.numFeatures)(params.coefficientUpperBound), true))
    }
    else {
      new LogisticRegression()
        .setRegParam(params.regParam)
        .setElasticNetParam(params.elasticNetParam)
        .setMaxIter(params.maxIter)
        .setTol(params.tolerance)
    }

    val paramMap = ParamMap(logR.maxIter -> params.maxIter)
      .put(logR.regParam, params.regParam)
    val paramMaps: Array[ParamMap] = new Array[ParamMap](2)
    for (i <- 0 to paramMaps.size -1) {
      paramMaps(i) = ParamMap(logR.maxIter -> params.maxIter)
        .put(logR.regParam, params.regParam)
    }
    val maxIterParamPair = ParamPair(logR.maxIter, params.maxIter)
    val regParamPair = ParamPair(logR.regParam, params.regParam)
    val model = params.apiName match {
      case "fit" => logR.fit(trainingData)
      case "fit1" => logR.fit(trainingData, paramMap)
      case "fit2" =>
        val models = logR.fit(trainingData, paramMaps)
        models(0)
      case "fit3" => logR.fit(trainingData, maxIterParamPair, regParamPair)
    }
    val costTime = (System.currentTimeMillis() - startTime) / 1000.0
    params.setLoadDataTime(loadDataTime)

    val testData = spark
      .read
      .format("libsvm")
      .option("vectorType", "dense")
      .option("numFeatures", params.numFeatures)
      .load(params.testDataPath)
      .repartition(params.numPartitions)
      .cache()

    val testingData = if (params.datasetName == "rcv") {
      val label_convert = udf((x: Double) => if (x < 0.0) 0.0 else 1.0)
      testData.withColumn("label", label_convert($"label"))
    } else {
      testData
    }

    val predictions = model.transform(testingData)
    val res = predictions.filter($"label" === $"prediction").count().toDouble / predictions.count()

    Utils.saveAndVerifyRes[LogRParams](params, res, sc)

    (res, costTime)
  }
}

