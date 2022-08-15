package com.bigdata.ml

import com.bigdata.utils.Utils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Encoders, SparkSession}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.{IDF, IDFModel}
import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.{DumperOptions, TypeDescription, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.nodes.Tag
import org.yaml.snakeyaml.representer.Representer

import java.io.{File, FileInputStream, FileOutputStream, FileWriter, ObjectInputStream, ObjectOutputStream}
import java.util
import scala.beans.BeanProperty

class IDFConfig extends Serializable {
  @BeanProperty var idf: util.HashMap[String, util.HashMap[String, util.HashMap[String, Object]]] = _
}

class IDFParams extends Serializable {
  @BeanProperty var pt: Int = _
  @BeanProperty var combineStrategy: String = _
  @BeanProperty var fetchMethod: String = _
  @BeanProperty var orcFormat: Boolean = _


  @BeanProperty var dataPath: String = _
  @BeanProperty var apiName: String = _
  @BeanProperty var datasetName: String = _
  @BeanProperty var costTime: Double = _
  @BeanProperty var isRaw: String = "no"
  @BeanProperty var algorithmName: String = _
  @BeanProperty var testcaseType: String = _
  @BeanProperty var saveDataPath: String = _
  @BeanProperty var verifiedDataPath: String = _
}

object IDFRunner{
  def main(args: Array[String]): Unit = {
    try {
      val platformName = args(0)
      val datasetName = args(1)
      val dataPath = args(2)
      val modelPath = args(3)
      val checkModel = java.lang.Boolean.parseBoolean(args(4))

      val stream = Utils.getStream("conf/ml/idf/idf.yml")
      val representer = new Representer
      representer.addClassTag(classOf[IDFParams], Tag.MAP)
      val options = new DumperOptions
      options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
      val yaml = new Yaml(new Constructor(classOf[IDFConfig]), representer, options)
      val description = new TypeDescription(classOf[IDFParams])
      yaml.addTypeDescription(description)
      val configs: IDFConfig = yaml.load(stream).asInstanceOf[IDFConfig]
      val paramsMap: util.HashMap[String, Object] = configs.idf.get(platformName).get(datasetName)
      val params = new IDFParams()
      params.setPt(paramsMap.get("pt").asInstanceOf[Int])
      params.setCombineStrategy(paramsMap.get("combineStrategy").asInstanceOf[String])
      params.setFetchMethod(paramsMap.get("fetchMethod").asInstanceOf[String])
      params.setOrcFormat(paramsMap.get("orcFormat").asInstanceOf[Boolean])
      params.setDataPath(dataPath)
      params.setDatasetName(datasetName)
      params.setAlgorithmName("IDF")
      params.setTestcaseType(s"IDF_${datasetName}_${platformName}")
      params.setSaveDataPath(s"hdfs:///tmp/ml/result/${params.algorithmName}/${datasetName}")
      if (platformName == "pri") {
        params.setIsRaw("yes")
        params.setVerifiedDataPath(params.saveDataPath)
        params.setSaveDataPath(s"${params.saveDataPath}_raw")
      }

      val conf = new SparkConf().setAppName(s"IDF_${datasetName}_${platformName}")
      conf.set("spark.driver.maxResultSize", "256G")
      if (platformName == "opt") {
        conf.set("spark.sophon.ml.idf.combineStrategy", params.combineStrategy)
        conf.set("spark.sophon.ml.idf.fetchMethod", params.fetchMethod)
      }
      val spark = SparkSession.builder().config(conf).getOrCreate()
      val costTime = new IDFKernel().runJob(spark, params)
      params.setCostTime(costTime)

      Utils.saveYml[IDFParams](params, yaml)
      println(s"Exec Successful: costTime: ${params.getCostTime}s")
    } catch {
      case e: Throwable =>
        println(s"Exec Failure: ${e.getMessage}")
        throw e
    }
  }

}


class IDFKernel {
  def runJob(spark: SparkSession, params: IDFParams): Double = {
    val sc = spark.sparkContext
    val startTime = System.currentTimeMillis()
    val orcData = spark.read.schema(Encoders.product[DocSchema].schema).format("orc").load(params.dataPath)
    val data = if (pt > 0){
      orcData.select("tf").repartition(params.pt)
    } else {
      orcData.select("tf")
    }
    val idf = new IDF().setInputCol("tf").setOutputCol("tf_idf")
    val model = idf.fit(data)
    val costTime = (System.currentTimeMillis() - startTime) / 1000.0
    val res = model.idf.toArray

    val fs = FileSystem.get(sc.hadoopConfiguration)
    val saveFile = new Path(params.saveDataPath)
    if (fs.exists(saveFile)) {
      fs.delete(saveFile, true)
    }
    sc.parallelize(res).saveAsTextFile(params.saveDataPath)
    if (params.isRaw == "yes") {
      val verifiedFile = new Path(params.verifiedDataPath)
      if (fs.exists(verifiedFile)) {
        val pri = sc.textFile(params.saveDataPath).map(line => line.map(_.toDouble)).toArray
        val opt = sc.textFile(params.verifiedDataPath).map(line => line.map(_.toDouble)).toArray
        if (Utils.isEqualVector(pri, opt)) {
          println(s"${params.algorithmName} correct")
        }
        else {
          println(s"${params.algorithmName} incorrect")
        }
      }
    }
    costTime
  }
  case class DocSchema(id: Long, tf: Vector)
}