package ai.pipestream.turboembed.conformance;

import ai.pipestream.turboembed.TurboBridge;
import ai.pipestream.turboembed.TurboRuntime;

/** Fresh-process class loading check, without native initialization. */
public final class LoaderProbe {
    private LoaderProbe() {}

    public static void main(String[] args) {
        TurboRuntime runtime = args.length == 0 ? TurboRuntime.load()
                : TurboRuntime.load(TurboBridge.valueOf(args[0]));
        System.out.println(runtime.kind());
    }
}
