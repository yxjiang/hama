package org.apache.hama.ml.perception;

import org.apache.hadoop.io.Writable;

/**
 * MLPMessage is used to hold the parameters that needs to be sent between the
 * tasks.
 */
public abstract class MLPMessage implements Writable {
  protected boolean terminated;

  public MLPMessage(boolean terminated) {
    setTerminated(terminated);
  }

  public void setTerminated(boolean terminated) {
    this.terminated = terminated;
  }

  public boolean isTerminated() {
    return terminated;
  }

}
