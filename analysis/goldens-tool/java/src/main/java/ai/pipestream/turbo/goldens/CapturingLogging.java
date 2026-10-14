package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.ILoggerFactory;
import org.slf4j.IMarkerFactory;
import org.slf4j.Marker;
import org.slf4j.event.Level;
import org.slf4j.helpers.BasicMarkerFactory;
import org.slf4j.helpers.LegacyAbstractLogger;
import org.slf4j.helpers.MessageFormatter;
import org.slf4j.helpers.NOPMDCAdapter;
import org.slf4j.spi.MDCAdapter;
import org.slf4j.spi.SLF4JServiceProvider;

/**
 * The SLF4J provider of the generator: every logger is off, except one whose debug lines a golden
 * records while it {@link #capture captures} them. OpenNLP's {@code QNMinimizer} reports its
 * iterations (the objective value, its change rate and the training accuracy) and why it stopped
 * only through debug logging, so this is how they reach the goldens. The messages are formatted as
 * SLF4J formats them, with {@code String.valueOf} of each argument.
 *
 * <p>Found through {@code META-INF/services/org.slf4j.spi.SLF4JServiceProvider}. Capturing is not
 * thread safe; the generator runs one annotator at a time.</p>
 */
public final class CapturingLogging implements SLF4JServiceProvider {

  private static final Map<String, CapturingLogger> LOGGERS = new ConcurrentHashMap<>();
  private static String capturedLogger;
  private static List<String> captured;

  private final ILoggerFactory loggerFactory = name -> LOGGERS.computeIfAbsent(name, CapturingLogger::new);
  private final IMarkerFactory markerFactory = new BasicMarkerFactory();
  private final MDCAdapter mdcAdapter = new NOPMDCAdapter();

  /**
   * Runs {@code action} with the debug level of the logger named {@code logger} on and returns
   * the messages it logged, in order.
   */
  static List<String> capture(String logger, Runnable action) {
    capturedLogger = logger;
    captured = new ArrayList<>();
    try {
      action.run();
      return captured;
    } finally {
      capturedLogger = null;
      captured = null;
    }
  }

  @Override
  public ILoggerFactory getLoggerFactory() {
    return loggerFactory;
  }

  @Override
  public IMarkerFactory getMarkerFactory() {
    return markerFactory;
  }

  @Override
  public MDCAdapter getMDCAdapter() {
    return mdcAdapter;
  }

  @Override
  public String getRequestedApiVersion() {
    return "2.0.99";
  }

  @Override
  public void initialize() {}

  /** A logger that is off unless it is the one being captured, which is on at debug and above. */
  private static final class CapturingLogger extends LegacyAbstractLogger {

    private static final long serialVersionUID = 1L;

    CapturingLogger(String name) {
      this.name = name;
    }

    private boolean capturing() {
      return name.equals(capturedLogger);
    }

    @Override
    public boolean isTraceEnabled() {
      return false;
    }

    @Override
    public boolean isDebugEnabled() {
      return capturing();
    }

    @Override
    public boolean isInfoEnabled() {
      return capturing();
    }

    @Override
    public boolean isWarnEnabled() {
      return capturing();
    }

    @Override
    public boolean isErrorEnabled() {
      return capturing();
    }

    @Override
    protected String getFullyQualifiedCallerName() {
      return null;
    }

    @Override
    protected void handleNormalizedLoggingCall(Level level, Marker marker, String pattern, Object[] arguments,
        Throwable throwable) {
      if (capturing()) {
        captured.add(MessageFormatter.basicArrayFormat(pattern, arguments));
      }
    }
  }
}
