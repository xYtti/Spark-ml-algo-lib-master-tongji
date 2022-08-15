package com.bigdata.ml
import java.io.{File, FileWriter}

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.yaml.snakeyaml.{DumperOptions, TypeDescription, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.nodes.Tag
import org.yaml.snakeyaml.representer.Representer

import scala.beans.BeanProperty
import java.util

import com.bigdata.utils.Utils
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.{Algo, Strategy}
import org.apache.spark.mllib.tree.impurity.{Gini, Variance}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel

class RFConfig extends Serializable {
  @BeanProperty var rf: util.HashMap[String, util.HashMap[String, util.HashMap[String, util.HashMap[String, Object]]]] = _
}

class RFParams extends Serializable {
  @BeanProperty var genericPt: Int = _
  @BeanProperty var maxMemoryInMB: Int = _
  @BeanProperty var pt: Int = _
  @BeanProperty var numCopiesInput: Int = _
  @BeanProperty var numTrees: Int = _
  @BeanProperty var maxDepth: Int = _
  @BeanProperty var maxBins: Int = _
  @BeanProperty var useNodeIdCache: Boolean = _
  @BeanProperty var checkpointInterval: Int = _
  @BeanProperty var numClasses: Int = _
  @BeanProperty var bcVariables: Boolean = _
  @BeanProperty var featureSubsetStrategy: String = _
  @BeanProperty var featuresType: String = _

  @BeanProperty var trainingDataPath: String = _
  @BeanProperty var testDataPath: String = _
  @BeanProperty var algorithmType: String = _
  @BeanProperty var apiName: String = _
  @BeanProperty var datasetName: String = _
  @BeanProperty var evaluation: Double = _
  @BeanProperty var costTime: Double = _
  @BeanProperty var cpuName: String = _
  @BeanProperty var isRaw: String = _
  @BeanProperty var algorithmName: String = _
  @BeanProperty var testcaseType: String = _
  @BeanProperty var saveDataPath: String = _
  @BeanProperty var verifiedDataPath: String = _
}

object RFRunner {
  def main(args: Array[String]): Unit = {
    try {
      val modelConfSplit = args(0).split("_")
      val (algorithmType, dataStructure, datasetName, apiName) =
        (modelConfSplit(0), modelConfSplit(1), modelConfSplit(2), modelConfSplit(3))
      val dataPath = args(1)
      val dataPathSplit = dataPath.split(",")
      val (trainingDataPath, testDataPath) = (dataPathSplit(0), dataPathSplit(1))
      val cpuName = args(2)
      val isRaw = args(3)
      val sparkConfSplit = args(4).split("_")
      val (master, deployMode, numExec, execCores, execMem) =
        (sparkConfSplit(0), sparkConfSplit(1), sparkConfSplit(2), sparkConfSplit(3), sparkConfSplit(4))

      val stream = (cpuName, isRaw) match {
        case ("aarch64", "no") =>
          Utils.getStream("conf/ml/rf/rf_arm.yml")
        case ("x86_64", "no") =>
          Utils.getStream("conf/ml/rf/rf_x86.yml")
        case ("x86_64", "yes") =>
          Utils.getStream("conf/ml/rf/rf_x86_raw.yml")
      }
      val representer = new Representer
      representer.addClassTag(classOf[RFParams], Tag.MAP)
      val options = new DumperOptions
      options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
      val yaml = new Yaml(new Constructor(classOf[RFConfig]), representer, options)
      val description = new TypeDescription(classOf[RFParams])
      yaml.addTypeDescription(description)
      val configs: RFConfig = yaml.load(stream).asInstanceOf[RFConfig]
      val paramsMap: util.HashMap[String, Object] = configs.rf.get(algorithmType).get(dataStructure).get(datasetName)
      val params = new RFParams()
      params.setGenericPt(paramsMap.getOrDefault("genericPt", "1000").asInstanceOf[Int])
      params.setMaxMemoryInMB(paramsMap.getOrDefault("maxMemoryInMB", "256").asInstanceOf[Int])
      params.setPt(paramsMap.getOrDefault("pt", "1000").asInstanceOf[Int])
      params.setNumCopiesInput(paramsMap.getOrDefault("numCopiesInput", "1").asInstanceOf[Int])
      params.setNumTrees(paramsMap.getOrDefault("numTrees", "20").asInstanceOf[Int])
      params.setMaxDepth(paramsMap.getOrDefault("maxDepth", "5").asInstanceOf[Int])
      params.setMaxBins(paramsMap.getOrDefault("maxBins", "32").asInstanceOf[Int])
      params.setUseNodeIdCache(paramsMap.getOrDefault("useNodeIdCache", "false").asInstanceOf[Boolean])
      params.setCheckpointInterval(paramsMap.getOrDefault("checkpointInterval", "10").asInstanceOf[Int])
      params.setNumClasses(paramsMap.get("numClasses").asInstanceOf[Int])
      params.setBcVariables(paramsMap.getOrDefault("bcVariables", "false").asInstanceOf[Boolean])
      params.setFeatureSubsetStrategy(paramsMap.getOrDefault("featureSubsetStrategy", "auto").asInstanceOf[String])
      params.setFeaturesType(paramsMap.getOrDefault("featuresType", "array").asInstanceOf[String])
      params.setTrainingDataPath(trainingDataPath)
      params.setTestDataPath(testDataPath)
      params.setAlgorithmType(algorithmType)
      params.setApiName(apiName)
      params.setDatasetName(datasetName)
      params.setCpuName(cpuName)
      params.setIsRaw(isRaw)
      params.setAlgorithmName("RF")
      params.setSaveDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/${algorithmType}_${datasetName}_${dataStructure}_${apiName}")

      var appName = s"RF_${algorithmType}_${datasetName}_${dataStructure}_${apiName}"
      if (isRaw.equals("yes")){
        appName = s"RF_RAW_${algorithmType}_${datasetName}_${dataStructure}_${apiName}"
        params.setVerifiedDataPath(params.saveDataPath)
        params.setSaveDataPath(s"${params.saveDataPath}_raw")
      }
      params.setTestcaseType(appName)
      val conf = new SparkConf().setAppName(appName).setMaster(master)
      val commonParas = Array (
        ("spark.submit.deployMode", deployMode),
        ("spark.executor.instances", numExec),
        ("spark.executor.cores", execCores),
        ("spark.executor.memory", execMem)
      )
      conf.setAll(commonParas)
      if ("no" == isRaw.asInstanceOf[String]) {
        conf.set("spark.boostkit.ml.rf.binnedFeaturesDataType",
          paramsMap.get("featuresType").asInstanceOf[String])
        conf.set("spark.boostkit.ml.rf.numTrainingDataCopies",
          paramsMap.get("numCopiesInput").asInstanceOf[Int].toString)
        conf.set("spark.boostkit.ml.rf.numPartsPerTrainingDataCopy",
          paramsMap.get("pt").asInstanceOf[Int].toString)
        conf.set("spark.boostkit.ml.rf.broadcastVariables",
          paramsMap.get("bcVariables").asInstanceOf[Boolean].toString)
      }
      val spark = SparkSession.builder.config(conf).getOrCreate()

      val (res, costTime) = dataStructure match {
        case "dataframe" => new RFKernel().runDataframeJob(spark, params)
        case "rdd" => new RFKernel().runRDDJob(spark, params)
      }
      params.setEvaluation(res)
      params.setCostTime(costTime)

      Utils.saveYml[RFParams](params, yaml)
      println(s"Exec Successful: costTime: ${costTime}s; evaluation: ${res}")
    } catch {
      case e: Throwable =>
        println(s"Exec Failure: ${e.getMessage}")
    }
  }
}

class RFKernel {
  def runDataframeJob(spark: SparkSession, params: RFParams): (Double, Double) = {
    val sc = spark.sparkContext
    val pt = params.pt
    val trainingDataPath = params.trainingDataPath
    val testDataPath = params.testDataPath
    val numTrees = params.numTrees
    val maxDepth = params.maxDepth
    val maxBins = params.maxBins
    val useNodeIdCache = params.useNodeIdCache
    val checkpointInterval = params.checkpointInterval
    val maxMemoryInMB = params.maxMemoryInMB
    val featureSubsetStrategy = params.featureSubsetStrategy
    val genericPt = params.genericPt

    val startTime = System.currentTimeMillis()

    val reader = spark.read.format("libsvm")
    if (params.datasetName == "mnist8m") {
      reader.option("numFeatures",784)
    } else if (params.datasetName == "higgs") {
      reader.option("numFeatures",28)
    } else if (params.datasetName == "epsilon") {
      reader.option("numFeatures", 2000)
    } else if (params.datasetName == "rcv") {
      reader.option("numFeatures", 47236)
    }

    val numPtTrainData = if ("no" == params.isRaw) genericPt else pt
    val trainingData = reader
      .load(trainingDataPath)
      .repartition(numPtTrainData)
      .persist(StorageLevel.MEMORY_AND_DISK_SER)

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(trainingData)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // for implementing different fit APIs
    val numTreesJY = 5
    val maxDepthJY = 3

    // Train a RandomForest model
    val rf = params.algorithmType match {
      case "classification" =>{
        val oldRf = new RandomForestClassifier()
          .setLabelCol("indexedLabel")
          .setFeaturesCol("features")
          .setNumTrees(numTreesJY)
          .setMaxDepth(maxDepthJY)
          .setMaxBins(maxBins)
          .setCacheNodeIds(useNodeIdCache)
          .setCheckpointInterval(checkpointInterval)
          .setMaxMemoryInMB(maxMemoryInMB)
        if (featureSubsetStrategy.nonEmpty)
          oldRf.setFeatureSubsetStrategy(featureSubsetStrategy)
        oldRf
      }
      case "regression" =>{
        val oldRf = new RandomForestRegressor()
          .setLabelCol("indexedLabel")
          .setFeaturesCol("features")
          .setNumTrees(numTreesJY)
          .setMaxDepth(maxDepthJY)
          .setMaxBins(maxBins)
          .setCacheNodeIds(useNodeIdCache)
          .setCheckpointInterval(checkpointInterval)
          .setMaxMemoryInMB(maxMemoryInMB)
        if (featureSubsetStrategy.nonEmpty)
          oldRf.setFeatureSubsetStrategy(featureSubsetStrategy)
        oldRf
      }
    }

    if (params.apiName == "fit"){
      rf.setNumTrees(numTrees)
      rf.setMaxDepth(maxDepth)
    }

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, rf, labelConverter))

    val paramMap = ParamMap(rf.maxDepth -> maxDepth)
      .put(rf.numTrees, numTrees)
    val maxDepth1 = maxDepthJY
    val maxDepth2 = maxDepth
    val firstParamPair = ParamPair(rf.maxDepth, maxDepth1)
    val otherParamPairs_1st = ParamPair(rf.maxDepth, maxDepth2)
    val otherParamPairs_2nd = ParamPair(rf.numTrees, numTrees)
    val paramMaps = new Array[ParamMap](2)
    for (i <- 0 until paramMaps.size){
      paramMaps(i) = ParamMap(rf.maxDepth -> maxDepth)
        .put(rf.numTrees, numTrees)
    }

    val model = params.apiName match {
      case "fit" => pipeline.fit(trainingData)
      case "fit1" => pipeline.fit(trainingData, paramMap)
      case "fit2" =>
        val models = pipeline.fit(trainingData, paramMaps)
        models(0)
      case "fit3" => pipeline.fit(trainingData, firstParamPair, otherParamPairs_1st, otherParamPairs_2nd)

    }

    val costTime = (System.currentTimeMillis() - startTime) / 1000.0

    val testData = reader
      .load(testDataPath)
      .repartition(genericPt)
      .persist(StorageLevel.MEMORY_AND_DISK_SER)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select (prediction, true label) and compute test error.
    val evaluator = params.algorithmType match {
      case "classification" =>
        new MulticlassClassificationEvaluator()
          .setLabelCol ("indexedLabel")
          .setPredictionCol ("prediction")
          .setMetricName ("accuracy")

      case "regression" =>
        new RegressionEvaluator()
          .setLabelCol ("indexedLabel")
          .setPredictionCol ("prediction")
          .setMetricName ("rmse")
    }
    val res = evaluator.evaluate(predictions)

    Utils.saveAndVerifyRes[RFParams](params, res, sc)

    (res, costTime)
  }

  def runRDDJob(spark: SparkSession, params: RFParams): (Double, Double) = {

    val sc = spark.sparkContext
    val pt = params.pt
    val trainingDataPath = params.trainingDataPath
    val testDataPath = params.testDataPath
    val numTrees = params.numTrees
    var maxDepth = params.maxDepth
    val maxBins = params.maxBins
    val useNodeIdCache = params.useNodeIdCache
    val checkpointInterval = params.checkpointInterval
    val maxMemoryInMB = params.maxMemoryInMB
    val featureSubsetStrategy = params.featureSubsetStrategy
    val genericPt = params.genericPt
    var numClasses = params.numClasses
    val startTime = System.currentTimeMillis()

    val numFeatures = params.datasetName match {
      case "mnist8m" => 784
      case "higgs" => 28
      case "epsilon" =>2000
      case "rcv" => 47236
    }

    val numPtTrainData = if ("no" == params.isRaw) genericPt else pt
    val trainingData = MLUtils.loadLibSVMFile(sc, trainingDataPath, numFeatures)
      .repartition(numPtTrainData)
      .persist(StorageLevel.MEMORY_AND_DISK_SER)
    val trainingLabelPositive = trainingData.map(i=> if (i.label < 0) {
      LabeledPoint(0.0, i.features)
    } else {
      LabeledPoint (i.label, i.features)
    })


    val model = params.algorithmType match {
      case "classification" =>
        val seed = "org.apache.spark.ml.classification.RandomForestClassifier".hashCode
        params.apiName match {
          case "train" =>
            val strategy = new Strategy (Algo.Classification, Gini, maxDepth = maxDepth,
              numClasses = numClasses, maxBins = maxBins, useNodeIdCache = useNodeIdCache,
              checkpointInterval = checkpointInterval, maxMemoryInMB = maxMemoryInMB)
            RandomForest.trainClassifier(trainingLabelPositive, strategy, numTrees, featureSubsetStrategy, seed)
          case "train1" =>
            val categoricalFeaturesInfo = Map[Int, Int]()
            RandomForest.trainClassifier(trainingLabelPositive, numClasses, categoricalFeaturesInfo,
              numTrees, featureSubsetStrategy, "gini", maxDepth, maxBins, seed)
          case "train2" =>
            val categoricalFeaturesInfo = new java.util.HashMap[java.lang.Integer, java.lang.Integer]()
            RandomForest.trainClassifier(trainingLabelPositive.toJavaRDD, numClasses, categoricalFeaturesInfo,
              numTrees, featureSubsetStrategy, "gini", maxDepth, maxBins, seed)
        }
      case "regression" =>
        val seed = "org.apache.spark.ml.regression.RandomForestRegressor".hashCode
        params.apiName match {
          case "train" =>
            val strategy = new Strategy (Algo.Regression, Variance, maxDepth = maxDepth,
              numClasses = 0, maxBins = maxBins, useNodeIdCache = useNodeIdCache,
              checkpointInterval = checkpointInterval, maxMemoryInMB = maxMemoryInMB)
            RandomForest.trainRegressor(trainingLabelPositive, strategy, numTrees, featureSubsetStrategy, seed)
          case "train1" =>
            val categoricalFeaturesInfo = Map[Int, Int]()
            RandomForest.trainRegressor(trainingLabelPositive, categoricalFeaturesInfo,
              numTrees, featureSubsetStrategy, "variance", maxDepth, maxBins, seed)
          case "train2" =>
            val categoricalFeaturesInfo = new java.util.HashMap[java.lang.Integer, java.lang.Integer]()
            RandomForest.trainRegressor(trainingLabelPositive.toJavaRDD, categoricalFeaturesInfo,
              numTrees, featureSubsetStrategy, "variance", maxDepth, maxBins, seed)
        }

    }

    val costTime = (System.currentTimeMillis() - startTime) / 1000.0

    val testData = MLUtils.loadLibSVMFile(sc, testDataPath)
      .repartition(genericPt)
      .persist(StorageLevel.MEMORY_AND_DISK_SER)
    val testLabelPositive = testData.map(i=> if (i.label < 0) {
      LabeledPoint(0.0, i.features)
    } else {
      LabeledPoint (i.label, i.features)
    })

    val labeleAndPreds = testLabelPositive.map{ point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val res = params.algorithmType match {
      case "classification" => 1.0 - labeleAndPreds.filter(r => r._1 == r._2).count.toDouble / testLabelPositive.count()
      case "regression" => math.sqrt(labeleAndPreds.map{ case(v, p) => math.pow((v - p), 2)}.mean())
    }

    Utils.saveAndVerifyRes[RFParams](params, res, sc)

    (res, costTime)
  }

}
