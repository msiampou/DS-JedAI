
package model.structures

case class CandidateSet() {
  private var _map = Map[Int, Int]()

  def increment(key: Int, a: Int = 1): Unit = {
    val value = _map.get(key)
    value match {
      case None =>
        add(key, a)
      case Some(x) =>
        add(key, x+a)
    }
  }

  def clear(): Unit = {
    _map = _map.empty
  }

  def get(key: Int): Int = {
    val value = _map.get(key)
    value match {
      case None => 0
      case Some(x) => x
    }
  }

  def add(key: Int, value: Int): Unit = _map += (key -> value)

  def remove(key: Int): Unit = _map -= key

}