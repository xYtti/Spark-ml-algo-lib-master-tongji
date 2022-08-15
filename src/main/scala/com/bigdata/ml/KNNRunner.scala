package com.bigdata.ml

import java.io.{File, FileWriter, PrintWriter}
import java.util
import com.bigdata.utils.Utils
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.neighbors.KNN
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{Row, SparkSession}
import org.yaml.snakeyaml.{DumperOptions, TypeDescription, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.nodes.Tag
import org.yaml.snakeyaml.representer.Representer

import scala.beans.BeanProperty

class KNNConfig extends Serializable {

  @BeanProperty var knn: util.HashMap[String, util.HashMap[String, util.HashMap[String, Object]]] = _
}

class KNNParams extends Serializable {

  @BeanProperty var pt: Int = _
  @BeanProperty var k: Int = _
  @BeanProperty var testNum: Int = _
  @BeanProperty var testBatchSize: Int = _
  @BeanProperty var featuresCol: String = _
  @BeanProperty var distanceCol: String = _
  @BeanProperty var neighborsCol: String = _
  @BeanProperty var topTreeSizeRate: Double = _
  @BeanProperty var topTreeLeafSize: Int = _
  @BeanProperty var subTreeLeafSize: Int = _

  @BeanProperty var inputDataPath: String = _
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

object KNNRunner {
  def main(args: Array[String]): Unit = {

    try {
      val datasetName = args(0)
      val inputDataPath = args(1)
      val cpuName = args(2)
      val isRaw = args(3)
      val sparkConfSplit = args(4).split("_")
      val (master, deployMode, numExec, execCores, execMem) =
        (sparkConfSplit(0), sparkConfSplit(1), sparkConfSplit(2), sparkConfSplit(3), sparkConfSplit(4))

      val stream = isRaw match {
        case "no" =>
          Utils.getStream("conf/ml/knn/knn.yml")
        case "yes" =>
          Utils.getStream("conf/ml/knn/knn_raw.yml")
      }
      val representer = new Representer
      representer.addClassTag(classOf[KNNParams], Tag.MAP)
      val options = new DumperOptions
      options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
      val yaml = new Yaml(new Constructor(classOf[KNNConfig]), representer, options)
      val description = new TypeDescription(classOf[KNNParams])
      yaml.addTypeDescription(description)
      val configs: KNNConfig = yaml.load(stream).asInstanceOf[KNNConfig]
      val paramsMap: util.HashMap[String, Object] = configs.knn.get(cpuName).get(datasetName)
      val params = new KNNParams()
      params.setPt(paramsMap.getOrDefault("pt", "200").asInstanceOf[Int])
      params.setK(paramsMap.getOrDefault("k", "10").asInstanceOf[Int])
      params.setTestNum(paramsMap.getOrDefault("testNum", "100000").asInstanceOf[Int])
      params.setTestBatchSize(paramsMap.getOrDefault("testBatchSize", "10").asInstanceOf[Int])
      params.setTopTreeSizeRate(paramsMap.getOrDefault("topTreeSizeRate", "10.0").asInstanceOf[Double])
      params.setTopTreeLeafSize(paramsMap.getOrDefault("topTreeLeafSize", "10").asInstanceOf[Int])
      params.setSubTreeLeafSize(paramsMap.getOrDefault("subTreeLeafSize", "30").asInstanceOf[Int])
      params.setFeaturesCol("features")
      params.setDistanceCol("distances")
      params.setNeighborsCol("neighbors")
      params.setInputDataPath(inputDataPath)
      params.setDatasetName(datasetName)
      params.setCpuName(cpuName)
      params.setIsRaw(isRaw)
      params.setAlgorithmName("KNN")
      params.setSaveDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/${cpuName}_${datasetName}")
      params.setVerifiedDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/forceRes")

      var appName = s"KNN_${datasetName}"
      if (isRaw == "yes") {
        appName = s"KNN_${datasetName}_raw"
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
      val spark = SparkSession.builder.config(conf).getOrCreate()

      val costTime = if (isRaw == "no") {
        new KNNKernel().runJob(spark, params)
      } else {
        new KNNKernel().runRawJob(spark, params)
      }
      params.setCostTime(costTime)

      Utils.saveYml[KNNParams](params, yaml)
      println(s"Exec Successful: costTime: ${costTime}s")
    } catch {
      case e: Throwable =>
        println(s"Exec Failure: ${e.getMessage}")
        throw e
    }
  }
}

class KNNKernel {

  def runJob(spark: SparkSession, params: KNNParams): Double = {

    import spark.implicits._
    val sc = spark.sparkContext
    val startTime = System.currentTimeMillis()
    val dataPath = params.inputDataPath
    val featuresCol = params.featuresCol
    val testNum = params.testNum
    val pt = params.pt
    val neighborsCol = params.neighborsCol
    val distanceCol = params.distanceCol
    val testBatchSize = params.testBatchSize
    val k = params.k
    var groundTruthHDFSPath = params.verifiedDataPath

    //read data
    val rawData = sc.textFile(dataPath)
      .map(line => {
        val arr = line.split("\t")
        val id = arr(0).toLong
        val feature = Vectors.dense(arr(1).split(",").map(_.toDouble))
        (id, feature)
      }).toDF("id", featuresCol).cache()

    //split train/test datasets
    val trainDataDF = rawData.filter($"id" >= testNum).repartition(pt).cache()
    val testDataDF = rawData.filter($"id" < testNum).repartition(pt).cache()
    trainDataDF.count()
    testDataDF.count()

    val loadDataTime = (System.currentTimeMillis() - startTime) / 1000.0
    params.setLoadDataTime(loadDataTime)

    //fit
    val model = new KNN()
      .setFeaturesCol(featuresCol)
      .setAuxiliaryCols(Array("id"))
      .fit(trainDataDF)

    //transform
    val testResults = model
        .setNeighborsCol(neighborsCol)
        .setDistanceCol(distanceCol)
        .setK(k)
        .setTestBatchSize(testBatchSize)
        .transform(testDataDF).cache()
    testResults.count()

    val costTime = (System.currentTimeMillis() - startTime) / 1000.0

    // 与groudtruth对比
    val trueResult = sc.textFile(groundTruthHDFSPath).map(line => {
      val arr = line.split("\t")
      val id = arr(0).toLong
      val neighbors = arr(1).split(",").map(_.toInt)
        .filter(neighborIdx => neighborIdx >= testNum).take(k)
      (id, neighbors)
    }).filter(_._2.length == k)
    val combinedData = trueResult.toDF("id", "trueNeighbors")
      .join(testResults.selectExpr("id", "neighbors", "distances"), "id")
      .map(r => {
        val trueNN = r.getAs[Seq[Int]]("trueNeighbors").toArray
        val myNN = r.getAs[Seq[Row]]("neighbors").map(_.getAs[Long]("id").toInt).toArray
        val myDistancesSize = r.getAs[Seq[Double]]("distances").toSet.size
        (r.getAs[Long]("id"), trueNN, myNN, myDistancesSize)
      })
      .filter(_._4 == k)
      .cache()
    //    val actualTotalNum = combinedData.count()

    val incorrectCases = combinedData.map{case (id, trueNN, myNN, _) => {
      val myNNSet = myNN.toSet
      var isEqual = true
      Range(0, k - 1).foreach(i => {
        if(!myNNSet.contains(trueNN(i)))
          isEqual = false
      })
      (id, isEqual, trueNN, myNN)
    }}.filter(!_._2).collect()

    //    println("---------- incorrect cases -----------")
    //    incorrectCases.foreach(x => {
    //      println(s"id=${x._1}\ttrue=" + x._3.mkString(",") + s"\tmy=" + x._4.mkString(","))
    //    })

    if(incorrectCases.length == 0)
      println("KNN correct")
    else
      println("KNN incorrect")

    costTime
  }

  def runRawJob(spark: SparkSession, params: KNNParams): Double = {

    import spark.implicits._
    val sc = spark.sparkContext
    val startTime = System.currentTimeMillis()
    val dataPath = params.inputDataPath
    val featuresCol = params.featuresCol
    val testNum = params.testNum
    val pt = params.pt
    val neighborsCol = params.neighborsCol
    val distanceCol = params.distanceCol
    val k = params.k
    val topTreeSizeRate = params.topTreeSizeRate
    val topTreeLeafSize = params.topTreeLeafSize
    val subTreeLeafSize = params.subTreeLeafSize

    //read data
    val rawData = sc.textFile(dataPath)
      .map(line => {
        val arr = line.split("\t")
        val id = arr(0).toLong
        val feature = Vectors.dense(arr(1).split(",").map(_.toDouble))
        (id, feature)
      }).toDF("id", featuresCol).cache()

    //split train/test datasets
    val trainDataDF = rawData.filter($"id" >= testNum).repartition(pt).cache()
    val testDataDF = rawData.filter($"id" < testNum).repartition(pt).cache()
    trainDataDF.count()
    testDataDF.count()

    val loadDataTime = (System.currentTimeMillis() - startTime) / 1000.0
    params.setLoadDataTime(loadDataTime)

    //fit
    import org.apache.spark.ml.knn.KNN
    val model = new KNN()
      .setTopTreeSize((pt * topTreeSizeRate).toInt)
      .setTopTreeLeafSize(topTreeLeafSize)
      .setSubTreeLeafSize(subTreeLeafSize)
      .setBalanceThreshold(0.0)
      .setFeaturesCol(featuresCol)
      .setAuxCols(Array("id"))
      .fit(trainDataDF)

    //transform
    val testResults = model
      .setBufferSize(Double.MaxValue)
      .setNeighborsCol(neighborsCol)
      .setDistanceCol(distanceCol)
      .setK(k)
      .transform(testDataDF).cache()
    testResults.count()

    val costTime = (System.currentTimeMillis() - startTime) / 1000.0

    costTime
  }

  def writeForceRes(params: KNNParams): Unit ={
    val spark = SparkSession
      .builder()
      .appName("writeResults")
      .getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    val dataPath = params.inputDataPath
    val featuresCol = params.featuresCol
    val neighborsCol = params.neighborsCol
    val distanceCol = params.distanceCol
    val pt = params.pt
    val k = params.k
    val testNum = params.testNum
    val testBatchSize = params.testBatchSize
    var groundTruthLocalPath = params.verifiedDataPath

    val rawData = sc.textFile(dataPath)
      .map(line => {
        val arr = line.split("\t")
        val id = arr(0).toLong
        val feature = Vectors.dense(arr(1).split(",").map(_.toDouble))
        (id, feature)
      }).cache()

    // split train/test datasets
    val trainData = rawData.filter(_._1 >= testNum).repartition(pt).cache()
    val testData = rawData.filter(_._1 < testNum).repartition(pt).cache()
    println(s"-------- split data, trainNum=${trainData.count()}, testNum=${testData.count()} ----------")
    rawData.unpersist(blocking = true)

    // search in batch
    for(startIdx <- 0 until testNum by testBatchSize) {
      val exeTime = System.currentTimeMillis()
      val endIdx = math.min(startIdx + testBatchSize, testNum)
      val queryLocal = testData.filter(x => x._1 >= startIdx && x._1 < endIdx).collect()
      val queryBd = sc.broadcast(queryLocal)
      val neighbors = trainData.mapPartitions(iter => {
        val curTrainData = iter.toArray
        Iterator(queryBd.value.map{case (queryIdx, queryVector) => {
          val distances = curTrainData.map{case (trainIdx, trainVector) =>
            (trainIdx, euclideanDistance(trainVector, queryVector))}
            .sortBy(t => (t._2, t._1)).take(k)
          (queryIdx, distances)
        }})
      }).treeReduce((arr1, arr2) => {
        arr1.indices.toArray.map(i => {
          (arr1(i)._1, (arr1(i)._2 ++ arr2(i)._2).sortBy(t => (t._2, t._1)).take(k))
        })
      }, depth = 3)
      val writer = new PrintWriter(s"${groundTruthLocalPath}/part-${startIdx}-${endIdx}")
      neighbors.foreach{case(queryIdx, queryNN) => {
        writer.write(queryIdx + "\t" + queryNN.map(_._1).mkString(",") + "\n")
      }}
      writer.close()
      println(s"------ $startIdx-$endIdx done, time=${(System.currentTimeMillis() - exeTime) / 60000.0} ---------")
      queryBd.destroy()
    }
    spark.stop()
  }

  def euclideanDistance(v1: Vector, v2: Vector): Double = {
    euclideanDistance(v1.toArray, v2.toArray)
  }

  def euclideanDistance(v1: Array[Double], v2: Array[Double]): Double = {
    math.sqrt(v1.indices.map(i => math.pow(v1(i) - v2(i), 2)).sum)
  }

}
