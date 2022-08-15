package com.bigdata.ml

import java.io.{File, FileWriter}
import java.util
import scala.beans.BeanProperty
import scala.collection.mutable
import com.bigdata.utils.Utils
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.SparkConf
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.yaml.snakeyaml.{DumperOptions, TypeDescription, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.nodes.Tag
import org.yaml.snakeyaml.representer.Representer

class ALSConfig extends Serializable {
  @BeanProperty var als: util.HashMap[String, util.HashMap[String, util.HashMap[String, util.HashMap[String, Object]]]] = _
}

class ALSParams extends Serializable {
  @BeanProperty var pt: Int = _
  @BeanProperty var nonnegative: Boolean = _
  @BeanProperty var implicitPrefs: Boolean = _
  @BeanProperty var numItemBlocks: Int = _
  @BeanProperty var numUserBlocks: Int = _
  @BeanProperty var numIterations: Int = _
  @BeanProperty var regParam: Double = _
  @BeanProperty var alpha: Double = _

  @BeanProperty var trainingDataPath: String = _
  @BeanProperty var apiName: String = _
  @BeanProperty var dataStructure: String = _
  @BeanProperty var datasetName: String = _
  @BeanProperty var evaluation: Double = _
  @BeanProperty var costTime: Double = _
  @BeanProperty var cpuName: String = _
  @BeanProperty var isRaw: String = _
  @BeanProperty var startTime: Long = _
  @BeanProperty var algorithmName: String = _
  @BeanProperty var testcaseType: String = _
  @BeanProperty var saveDataPath: String = _
  @BeanProperty var verifiedDataPath: String = _
}

object ALSRunner {
  def main(args: Array[String]): Unit = {
    try {
      val modelConfSplit = args(0).split("_")
      val (dataStructure, datasetName, apiName) = (modelConfSplit(0), modelConfSplit(1), modelConfSplit(2))
      val dataPath = args(1)
      val dataPathSplit = dataPath.split(",")
      val trainingDataPath = dataPathSplit(0)
      val cpuName = args(2)
      val isRaw = args(3)
      val sparkConfSplit = args(4).split("_")
      val (master, deployMode, numExec, execCores, execMem) =
        (sparkConfSplit(0), sparkConfSplit(1), sparkConfSplit(2), sparkConfSplit(3), sparkConfSplit(4))

      val stream = (cpuName, isRaw) match {
        case ("aarch64", "no") =>
          Utils.getStream("conf/ml/als/als.yml")
        case ("x86_64", "no") =>
          Utils.getStream("conf/ml/als/als.yml")
        case ("x86_64", "yes") =>
          Utils.getStream("conf/ml/als/als_raw.yml")
      }
      val representer = new Representer
      representer.addClassTag(classOf[ALSParams], Tag.MAP)
      val options = new DumperOptions
      options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
      val yaml = new Yaml(new Constructor(classOf[ALSConfig]), representer, options)
      val description = new TypeDescription(classOf[ALSParams])
      yaml.addTypeDescription(description)
      val configs: ALSConfig = yaml.load(stream).asInstanceOf[ALSConfig]
      val paramsMap: util.HashMap[String, Object] = configs.als.get(cpuName).get(dataStructure).get(datasetName)
      val params = new ALSParams()
      params.setPt(paramsMap.getOrDefault("pt", "1000").asInstanceOf[Int])
      params.setNumIterations(paramsMap.getOrDefault("numIterations", "200").asInstanceOf[Int])
      params.setNonnegative(paramsMap.getOrDefault("nonnegative", "false").asInstanceOf[Boolean])
      params.setImplicitPrefs(paramsMap.getOrDefault("implicitPrefs", "false").asInstanceOf[Boolean])
      params.setNumItemBlocks(paramsMap.getOrDefault("numItemBlocks", "228").asInstanceOf[Int])
      params.setNumUserBlocks(paramsMap.getOrDefault("numUserBlocks", "228").asInstanceOf[Int])
      params.setRegParam(paramsMap.getOrDefault("regParam", "0.0").asInstanceOf[Double])
      params.setAlpha(paramsMap.getOrDefault("alpha", "1.0").asInstanceOf[Double])
      params.setTrainingDataPath(trainingDataPath)
      params.setDataStructure(dataStructure)
      params.setDatasetName(datasetName)
      params.setApiName(apiName)
      params.setCpuName(cpuName)
      params.setIsRaw(isRaw)
      params.setAlgorithmName("ALS")
      params.setTestcaseType(s"ALS_${dataStructure}_${datasetName}_${apiName}")
      params.setSaveDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/${datasetName}_${dataStructure}_${apiName}_${cpuName}")

      if (isRaw.equals("yes")){
        params.setTestcaseType(s"ALS_raw_${dataStructure}_${datasetName}_${apiName}")
        params.setVerifiedDataPath(params.saveDataPath)
        params.setSaveDataPath(s"${params.saveDataPath}_raw")
      }
      val conf = isRaw match {
        case "yes" =>
          new SparkConf().setAppName(s"ALS_raw_${dataStructure}_${datasetName}").setMaster(master)
        case "no" =>
          new SparkConf().setAppName(s"ALS_${dataStructure}_${datasetName}").setMaster(master)
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
      val startTime = System.currentTimeMillis()
      params.setStartTime(startTime)
      val sc = spark.sparkContext

      import spark.implicits._
      val rawdata: RDD[SparseVector] = sc.objectFile(dataPath).repartition(params.pt)
      val (predictions, costTime) = dataStructure match {
        case "dataframe" =>
          val ratings = Vector2Rating(rawdata).toDF().cache()
          println("count: " + ratings.count())
          val mapTime = System.currentTimeMillis()
          println("map cost Time[seconds]: " + (mapTime - startTime).toDouble / 1000.0)
          new ALSKernel().runDataframeJob(spark, ratings, params)
        case "rdd" =>
          params.apiName = "run"
          val ratings: RDD[Rating] = Vector2Rating(rawdata).cache()
          println("count: " + ratings.count())
          val mapTime = System.currentTimeMillis()
          println("map cost Time[seconds]: " + (mapTime - startTime).toDouble / 1000.0)
          new ALSKernel().runRDDJob(spark, ratings, params)
      }
      params.setEvaluation(predictions)
      params.setCostTime(costTime)

      Utils.saveYml[ALSParams](params, yaml)
      println(s"Exec Successful: costTime: ${costTime}s; evaluation: ${predictions}")
    } catch {
      case e: Throwable =>
        println(s"Exec Failure: ${e.getMessage}")
    }


      def Vector2Rating(rawdata: RDD[SparseVector]) : RDD[Rating] = {
        val Ratingdata: RDD[Rating] = rawdata.zipWithIndex().flatMap{
          case (v, i) =>
            val arr = mutable.ArrayBuilder.make[Rating]
            arr.sizeHint(v.numActives)
            v.foreachActive{(ii, vi) =>
              arr += Rating(i.toInt, ii, vi.toFloat)
            }
            arr.result()
        }
        Ratingdata
      }
  }
}

class ALSKernel {

  def runDataframeJob(spark: SparkSession, ratings: DataFrame, params: ALSParams): (Double, Double) = {
    val sc = spark.sparkContext
    val numIterations = params.numIterations
    val nonnegative = params.nonnegative
    val implicitPrefs = params.implicitPrefs
    val numItemBlocks = params.numItemBlocks
    val numUserBlocks = params.numUserBlocks
    val regParam = params.regParam
    val alpha = params.alpha

    val als = new ALS()
      .setMaxIter(numIterations)
      .setUserCol("user")
      .setItemCol("product")
      .setRatingCol("rating")
      .setNonnegative(nonnegative)
      .setImplicitPrefs(implicitPrefs)
      .setNumItemBlocks(numItemBlocks)
      .setNumUserBlocks(numUserBlocks)
      .setRegParam(regParam)
      .setAlpha(alpha)

    val paramMap = ParamMap(als.maxIter -> params.numIterations)
      .put(als.regParam, params.regParam)

    val paramMaps: Array[ParamMap] = new Array[ParamMap](2)
    for (i <- 0 to paramMaps.size -1) {
      paramMaps(i) = ParamMap(als.maxIter -> params.numIterations)
        .put(als.regParam, params.regParam)
    }
    val maxIterParamPair = ParamPair(als.maxIter, params.numIterations)
    val regParamPair = ParamPair(als.regParam, params.regParam)
    val model = params.apiName match {
      case "fit" => als.fit(ratings)
      case "fit1" => als.fit(ratings, paramMap)
      case "fit2" =>
        val models = als.fit(ratings, paramMaps)
        models(0)
      case "fit3" => als.fit(ratings, maxIterParamPair, regParamPair)
    }

    val costTime = (System.currentTimeMillis() - params.startTime) / 1000.0

    model.setColdStartStrategy("drop")
    val predictions = model.transform(ratings)

    val res = if (params.implicitPrefs) {
      val p = predictions.select("rating", "prediction").rdd
        .map{ case Row(label: Double, prediction: Float) => (label, prediction) }
        .map{ case (r1, r2) =>
          val pr = if (r1 > 0.0) 1.0f else 0.0f
          val err = (pr - r2)
          err * err
        }.mean()
      println("implicitPrefs Mean Squared Error = " + p)
      p
    } else {
      val p = predictions.select("rating", "prediction").rdd
        .map{ case Row(label: Double, prediction: Float) => (label, prediction) }
        .map {t =>
          val err = (t._1 - t._2)
          err * err
        }.mean()
      println("Mean Squared Error = " + p)
      p
    }

    Utils.saveAndVerifyRes[ALSParams](params, res, sc)

    (res, costTime)
  }

  def runRDDJob(spark: SparkSession, ratings: RDD[Rating], params: ALSParams): (Double, Double) = {

    val sc = spark.sparkContext
    val numIterations = params.numIterations
    val nonnegative = params.nonnegative
    val implicitPrefs = params.implicitPrefs
    val numItemBlocks = params.numItemBlocks
    val numUserBlocks = params.numUserBlocks
    val regParam = params.regParam
    val alpha = params.alpha

    import org.apache.spark.mllib.recommendation.ALS
    val rank = 10
    val model = ALS.train(ratings, rank, numIterations, regParam, numUserBlocks)
    val costTime = (System.currentTimeMillis() - params.startTime) / 1000.0

    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(usersProducts).map{ case Rating(user, product, rate) =>
        ((user, product), rate)
      }
    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)

    val res = if (implicitPrefs) {
      val p = ratesAndPreds.map{ case ((user, product), (r1, r2)) => (r1, r2)}
        .map{ case (r1, r2) =>
          val pr = if (r1 > 0.0) 1.0f else 0.0f
          val err = (pr - r2)
          err * err
        }.mean()
      println("implicitPrefs Mean Squared Error = " + p)
      p
    } else {
      val p = ratesAndPreds.map{ case ((user, product), (r1, r2)) => (r1, r2)}
        .map {t =>
          val err = (t._1 - t._2)
          err * err
        }.mean()
      println("Mean Squared Error = " + p)
      p
    }

    Utils.saveAndVerifyRes[ALSParams](params, res, sc)

    (res, costTime)
  }

}
