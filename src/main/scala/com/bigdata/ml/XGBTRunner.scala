package com.bigdata.ml

import com.bigdata.utils.Utils
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.yaml.snakeyaml.{DumperOptions, TypeDescription, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.nodes.Tag
import org.yaml.snakeyaml.representer.Representer

import java.io.{File, FileWriter, PrintWriter, StringWriter}
import scala.beans.BeanProperty
import java.util

class XGBTConfig extends Serializable{
  @BeanProperty var xgbt: util.HashMap[String,util.HashMap[String,util.HashMap[String,util.HashMap[String,Object]]]]=_
}
class XGBTParams extends Serializable{
  
  @BeanProperty var eta: Double = _
  @BeanProperty var gamma: Double = _
  @BeanProperty var min_child_weight: Int = _
  @BeanProperty var max_depth: Int = _
  @BeanProperty var allow_non_zero_for_missing: Boolean = _
  @BeanProperty var vectorType: String = _
  @BeanProperty var enable_bbgen: Boolean = _
  @BeanProperty var rabit_enable_tcp_no_delay: Boolean = _
  @BeanProperty var objective: String = _
  @BeanProperty var num_round: Int = _
  @BeanProperty var num_workers: Int = _
  @BeanProperty var nthread: Int = _
  @BeanProperty var tree_method: String = _
  @BeanProperty var grow_policy: String = _

  @BeanProperty var algorithmName: String = _
  @BeanProperty var testcaseType: String = _
  @BeanProperty var mode: String = _
  @BeanProperty var verbosity: Int = _
  @BeanProperty var tr_fname: String = _
  @BeanProperty var ts_fname: String = _
  @BeanProperty var num_class: Int = _
  @BeanProperty var evaluation: Double = _
  @BeanProperty var costTime: Double = _
  @BeanProperty var isRaw: String = _
  @BeanProperty var algType: String = _
  @BeanProperty var saveDataPath: String = _
  @BeanProperty var verifiedDataPath: String = _

}
object XGBTRunner {
  def main(args: Array[String]): Unit = {
    try{
      val modelConf = args(0)
      val modelConfSplit = modelConf.split("_")
      val (algType,datasetName)=(modelConfSplit(0),modelConfSplit(1))
      val dataPath = args(1)
      val dataPathSplit = dataPath.split(",")
      val (trainDataPath,testDataPath)=(dataPathSplit(0),dataPathSplit(1))
      val cpuName = args(2)
      val isRaw = args(3)
      val sparkConfSplit = args(4).split("_")
      val (master,deployMode)=(sparkConfSplit(0),sparkConfSplit(1))

      val stream = isRaw match{
        case "no" => Utils.getStream("conf/ml/xgbt/xgbt.yml")
        case "yes" => Utils.getStream("conf/ml/xgbt/xgbt_raw.yml")
      }
      val representer =new Representer
      representer.addClassTag(classOf[XGBTParams],Tag.MAP)
      val options = new DumperOptions
      options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
      val yaml = new Yaml(new Constructor(classOf[XGBTConfig]),representer,options)
      val description = new TypeDescription(classOf[XGBTParams])
      yaml.addTypeDescription(description)
      val configs: XGBTConfig = yaml.load(stream).asInstanceOf[XGBTConfig]
      val paramsMap: util.HashMap[String,Object] = configs.xgbt.get(cpuName).get(algType).get(datasetName)
      val params = new XGBTParams()
      params.setEta(paramsMap.get("eta").asInstanceOf[Double])
      params.setGamma(paramsMap.get("gamma").asInstanceOf[Int])
      params.setMin_child_weight(paramsMap.get("min_child_weight").asInstanceOf[Int])
      params.setMax_depth(paramsMap.get("max_depth").asInstanceOf[Int])
      params.setAllow_non_zero_for_missing(paramsMap.get("allow_non_zero_for_missing").asInstanceOf[Boolean])
      params.setVectorType(paramsMap.get("vectorType").asInstanceOf[String])
      params.setObjective(paramsMap.get("objective").asInstanceOf[String])
      params.setNum_round(paramsMap.get("num_round").asInstanceOf[Int])
      params.setNum_workers(paramsMap.get("num_workers").asInstanceOf[Int])
      params.setNthread(paramsMap.get("nthread").asInstanceOf[Int])
      params.setTree_method(paramsMap.get("tree_method").asInstanceOf[String])
      params.setGrow_policy(paramsMap.get("grow_policy").asInstanceOf[String])
      params.setTr_fname(trainDataPath)
      params.setTs_fname(testDataPath)
      params.setIsRaw(isRaw)
      params.setAlgType(algType)
      params.setAlgorithmName("xgboost")
      params.setSaveDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/${datasetName}_${algType}_${cpuName}")
      var appName = s"XGBT_${datasetName}_${algType}_${cpuName}"
      if(isRaw == "yes") {
        appName = s"XGBT_${datasetName}_${algType}_raw"
        params.setVerifiedDataPath(params.saveDataPath)
        params.setSaveDataPath(s"${params.saveDataPath}_raw")
      }
      params.setTestcaseType(appName)
      if(modelConf=="classification_mnist8m"){
        params.setNum_class(paramsMap.get("num_class").asInstanceOf[Int])
      }
      if(isRaw == "no"){
        params.setEnable_bbgen(paramsMap.get("enable_bbgen").asInstanceOf[Boolean])
        params.setRabit_enable_tcp_no_delay(paramsMap.get("rabit_enable_tcp_no_delay").asInstanceOf[Boolean])
      }


      val conf = new SparkConf().setAppName(appName).setMaster(master)
      val commonParas =Array(
        ("spark.submit.deployMode",deployMode)
      )
      conf.setAll(commonParas)
      val spark= SparkSession.builder().config(conf).getOrCreate()

      val (res, costTime) = new XGBTKernel().runJob(spark, params, paramsMap)
      params.setEvaluation(res)
      params.setCostTime(costTime)

      Utils.saveYml[XGBTParams](params, yaml)
      println(s"Exec Successful: costTime: ${costTime}s; evaluation: ${res}")
    }catch {
      case e: Throwable=>
        println(s"Exec Failure: ${e.getMessage}")
        val sw: StringWriter = new StringWriter()
        val pw: PrintWriter = new PrintWriter(sw)
        e.printStackTrace(pw)
        println("=============>>printStackTraceStr Exception: " + e.getClass + "\n===>" + sw.toString)
    }
  }
}


class XGBTKernel {
  def runJob(spark: SparkSession, params: XGBTParams, paramsMap: util.HashMap[String,Object]): (Double, Double) = {
    val sc = spark.sparkContext
    var paramsAnyMap: scala.Predef.Map[String,Any]=Map[String,Any]()
    paramsAnyMap += ("eta" -> paramsMap.get("eta"))
    paramsAnyMap += ("gamma" -> paramsMap.get("gamma"))
    paramsAnyMap += ("min_child_weight" -> paramsMap.get("min_child_weight"))
    paramsAnyMap += ("max_depth" -> paramsMap.get("max_depth"))
    paramsAnyMap += ("allow_non_zero_for_missing" -> paramsMap.get("allow_non_zero_for_missing"))
    paramsAnyMap += ("vectorType" -> paramsMap.get("vectorType"))
    paramsAnyMap += ("objective" -> paramsMap.get("objective"))
    paramsAnyMap += ("num_round" -> paramsMap.get("num_round"))
    paramsAnyMap += ("num_workers" -> paramsMap.get("num_workers"))
    paramsAnyMap += ("nthread" -> paramsMap.get("nthread"))
    paramsAnyMap += ("tree_method" -> paramsMap.get("tree_method"))
    paramsAnyMap += ("grow_policy" -> paramsMap.get("grow_policy"))
    if(modelConf=="classification_mnist8m") {
      paramsAnyMap += ("num_class" -> paramsMap.get("num_class"))
    }
    if (params.isRaw=="no"){
      paramsAnyMap += ("enable_bbgen" -> paramsMap.get("enable_bbgen"))
      paramsAnyMap += ("rabit_enable_tcp_no_delay" -> paramsMap.get("rabit_enable_tcp_no_delay"))
    }
    val start_time = System.currentTimeMillis()
    val XGBTrain = params.algType match {
      case "classification" =>
        new XGBoostClassifier(paramsAnyMap).setLabelCol("label").setFeaturesCol("features")
      case "regression" =>
        new XGBoostRegressor(paramsAnyMap).setLabelCol("label").setFeaturesCol("features")
    }

    val train_data = getTrainData(spark,params).persist(StorageLevel.MEMORY_AND_DISK_SER)

    val model = XGBTrain.fit(train_data)

    val fitEndTime = System.currentTimeMillis()
    val test_data = getTestData(spark,params).persist(StorageLevel.MEMORY_AND_DISK_SER)
    val predictions = model.transform(test_data)

    val evaluator = params.algType match {
      case "classification" =>
        new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
      case "regression" =>
        new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    }
    val res = evaluator.evaluate(predictions)
    println(s"Test Error =  ${(1.0-res)}")
    val costTime = (fitEndTime-start_time)/1000.0
    predictions.select("prediction","label","features").show(5)

    Utils.saveAndVerifyRes[XGBTParams](params, res, sc)

    (res, costTime)
  }

  def getTrainData(spark:SparkSession,config:XGBTParams): Dataset[Row] ={
    val tr_fname = config.getTr_fname
    println("tr_fname",tr_fname)
    var reader = spark.read.format("libsvm").option("vectorType",config.getVectorType)
    val tr_data = reader.load(tr_fname)
    tr_data
  }
  def getTestData(spark: SparkSession, config: XGBTParams): Dataset[Row] = {
    val ts_fname = config.getTs_fname
    println("ts_fname" , ts_fname)
    var reader= spark.read.format("libsvm").option("vectorType",config.getVectorType)
    val ts_data = reader.load(ts_fname)
    ts_data
  }

}