//scalastyle:off
package com.bigdata.ml

import java.io.{File, FileWriter}
import java.util.HashMap
import com.bigdata.utils.Utils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.ml.recommendation.{SimRank, SimRankOpenSource}
import org.yaml.snakeyaml.{DumperOptions, TypeDescription, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.nodes.Tag
import org.yaml.snakeyaml.representer.Representer

import java.util
import scala.beans.BeanProperty

class SimRankConfig extends Serializable {

  @BeanProperty var simrank: util.HashMap[String, util.HashMap[String, Object]] = _
}

class SimRankParams extends Serializable {

  @BeanProperty var numPartitions: Int = _
  @BeanProperty var damp: Double = _
  @BeanProperty var maxIter: Int = _
  @BeanProperty var saveResult: Boolean = false
  @BeanProperty var savePath: String = _

  @BeanProperty var isRaw: String = _
  @BeanProperty var cpuName: String = _
  @BeanProperty var dataPath: String = _
  @BeanProperty var datasetName: String = _
  @BeanProperty var caseName: String = _
  @BeanProperty var costTime: Double = _
  @BeanProperty var loadDataTime: Double = _
  @BeanProperty var algorithmType: String = _
  @BeanProperty var algorithmName: String = _
  @BeanProperty var saveDataPath: String = _
  @BeanProperty var verifiedDataPath: String = _
}

object SimRankRunner {
  def main(args: Array[String]): Unit = {
    try {
      val caseName = args(0)
      val caseArray = caseName.split("-")
      val (datasetName, algorithmType, cpuName) = (caseArray(0), caseArray(1), caseArray(2))
      val dataPath = args(1)
      val isRaw = args(2)

      val stream = Utils.getStream("conf/ml/simrank/simrank.yml")
      val representer = new Representer
      representer.addClassTag(classOf[SimRankParams], Tag.MAP)
      val options = new DumperOptions
      options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
      val yaml = new Yaml(new Constructor(classOf[SimRankConfig]), representer, options)
      val description = new TypeDescription(classOf[SimRankParams])
      yaml.addTypeDescription(description)
      val config: SimRankConfig = yaml.load(stream).asInstanceOf[SimRankConfig]
      val paramsMap: util.HashMap[String, Object] = config.simrank.get(caseName)
      val params = new SimRankParams()
      params.setNumPartitions(paramsMap.get("numPartitions").asInstanceOf[Int])
      params.setMaxIter(paramsMap.get("maxIter").asInstanceOf[Int])
      params.setDamp(paramsMap.get("damp").asInstanceOf[Double])
      params.setSaveResult(paramsMap.get("saveResult").asInstanceOf[Boolean])
      params.setSavePath(paramsMap.get("savePath").asInstanceOf[String])
      params.setAlgorithmType(algorithmType)
      params.setCpuName(cpuName)
      params.setIsRaw(isRaw)
      params.setDataPath(dataPath)
      params.setDatasetName(datasetName)
      params.setCaseName(caseName)
      params.setAlgorithmName("simrank")
      params.setSaveDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/${datasetName}_${cpuName}")
      if (isRaw.equals("yes")){
        params.setVerifiedDataPath(params.saveDataPath)
        params.setSaveDataPath(s"${params.saveDataPath}_raw")
      }

      val conf = new SparkConf()
        .setAppName(s"SimRank_${caseName}")
      val spark = SparkSession.builder.config(conf).getOrCreate()

      val costTime = new SimRankKernel().runJob(spark, params)

      Utils.saveYml[SimRankParams](params, yaml)
      println(s"Exec Successful: costTime: ${costTime}s")
    } catch {
      case e: Throwable =>
        println(s"Exec Failure: ${e.getMessage}")
        throw e
    }
  }
}

class SimRankKernel {

  def runJob(spark: SparkSession, params: SimRankParams): Double = {
    val sc = spark.sparkContext
    val fs = FileSystem.get(sc.hadoopConfiguration)
    import spark.implicits._

    val startTime = System.currentTimeMillis()
    val userCol = "user"
    val itemCol = "item"
    val df = spark.sparkContext.objectFile[Rating[Int]](params.getDataPath).repartition(params.getNumPartitions)
      .map(row => {
        ("user-" + row.user.toString, "item-" + row.item.toString)
      }).toDF(userCol, itemCol)

    val loadDataTime = (System.currentTimeMillis() - startTime) / 1000.0
    params.setLoadDataTime(loadDataTime)

    if (params.getIsRaw.equals("no")) {
      val simrank = new SimRank()
        .setDamp(params.getDamp)
        .setNumIter(params.getMaxIter)
        .setUserCol(userCol)
        .setItemCol(itemCol)

      val simrankRes = simrank.computeSimilarity(df)
      val resItem = simrankRes.itemSimilarity.foreach(_ => {})
      val resUser = simrankRes.userSimilarity.foreach(_ => {})
      val costTime = (System.currentTimeMillis() - startTime) / 1000.0
      params.setCostTime(costTime)
      val saveItemPath = s"${params.saveDataPath}_item"
      val saveUserPath = s"${params.saveDataPath}_user"
      Utils.saveArray(resItem, saveItemPath, sc)
      Utils.saveArray(resUser, saveUserPath, sc)

    } else {
      val simrankRes = new SimRankOpenSource().execute(df, (userCol, itemCol), params.getDamp, params.getMaxIter)
      val resItem = simrankRes._1.foreach(_ => {})
      val resUser = simrankRes._2.foreach(_ => {})
      val costTime = (System.currentTimeMillis() - startTime) / 1000.0
      params.setCostTime(costTime)
      val saveItemPath = s"${params.saveDataPath}_item"
      val saveUserPath = s"${params.saveDataPath}_user"
      Utils.saveArray(resItem, saveItemPath, sc)
      Utils.saveArray(resUser, saveUserPath, sc)
      val verifiedItemFile = new Path(s"${params.verifiedDataPath}_item")
      val verifiedUserFile = new Path(s"${params.verifiedDataPath}_user")
      if (fs.exists(verifiedItemFile) && fs.exists(verifiedUserFile)) {
        if (Utils.verifyVector(saveItemPath, s"${params.verifiedDataPath}_item", sc)
        && Utils.verifyVector(saveUserPath, s"${params.verifiedDataPath}_item", sc))
          println(s"${params.algorithmName} correct")
        else
          println(s"${params.algorithmName} incorrect")
      }
    }
    params.getCostTime
  }
}

