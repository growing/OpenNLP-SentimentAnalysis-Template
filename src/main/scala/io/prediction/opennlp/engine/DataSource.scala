package io.prediction.opennlp.engine

import io.prediction.controller.{EmptyEvaluationInfo, EmptyParams, PDataSource}
import io.prediction.data.storage.Storage
import io.prediction.opennlp.engine.Sentiment.Sentiment
import opennlp.maxent.BasicEventStream
import opennlp.model.OnePassDataIndexer
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.util.Random

class DataSource(val dsp: DataSourceParams) extends PDataSource[
  TrainingData,
  EmptyParams,
  Query,
  Sentiment] {

  val Separator = " "

  override def readTraining(sc: SparkContext): TrainingData = {
    val trainingTreeStrings = allPhraseAndSentiments(sc)
    phraseAndSentimentsToTrainingData(trainingTreeStrings)
  }

  override def readEval(
    sc: SparkContext): Seq[(TrainingData, EmptyParams, RDD[(Query, Sentiment)])] = {
    val shuffled = Random.shuffle(allPhraseAndSentiments(sc))
    val (trainingSet, testingSet) =
      shuffled.splitAt((shuffled.size*0.9).toInt)

    val trainingData = phraseAndSentimentsToTrainingData(trainingSet)

    val qna = testingSet.map { line =>
      val lastSpace = line.lastIndexOf(Separator)
      (Query(line.substring(0, lastSpace)), Sentiment(line.substring(lastSpace + 1).toInt))
    }

    Seq((trainingData, EmptyParams(), sc.parallelize(qna)))
  }

  private def allPhraseAndSentiments(sc: SparkContext): Seq[String] = {
    val events = Storage.getPEvents().find(appId = dsp.appId, entityType = Some("phrase"))(sc)

    events.map { event =>
      val phrase = event.properties.get[String]("phrase")
      val sentiment = event.properties.get[String]("sentiment")

      s"$phrase $sentiment"
    }.collect().toSeq
  }

  private def phraseAndSentimentsToTrainingData(phraseAndSentiments: Seq[String]) = {
    val eventStream = new BasicEventStream(new SeqDataStream(phraseAndSentiments), Separator)
    val dataIndexer = new OnePassDataIndexer(eventStream, dsp.cutoff)

    TrainingData(dataIndexer)
  }
}
